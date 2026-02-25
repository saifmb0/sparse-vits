#!/usr/bin/env python3
"""
tune_block_sizes.py — Grid-search BLOCK_M × BLOCK_N for triton_ragged_attention
=================================================================================
Sweeps all power-of-2 tile sizes (16, 32, 64, 128) and reports, for each combo:
  • Kernel latency  (ms)   — raw timed triton_ragged_attention call
  • End-to-end throughput  (images/s) — full model forward at BS=32, 50% prune
  • Max abs diff vs PyTorch — numerical equivalence (must stay < 0.01)
  • Pass/Fail               — prediction match with default BLOCK_M=32/BLOCK_N=32

Designed for a T4 16 GB GPU (Turing SM75).  Larger tile sizes are viable here
compared to the GTX 1650 default config used during development.

Usage:
    python tune_block_sizes.py [--imagenet-samples N] [--batch-size BS]

Results are written to results/block_size_tuning.json and printed as a table.
"""

import sys, os, gc, time, json, argparse, itertools, functools
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DEVICE, IMG_SIZE, WARMUP_ITERS, BENCH_ITERS
from models.deit_base import load_deit, split_deit, get_dtype
from models.pruning import threshold_prune_mask
from kernels.pack_tokens import triton_pack_tokens
import kernels.ragged_attention as _rag_mod
from kernels.ragged_attention import triton_ragged_attention as _orig_ragged_attn

# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--imagenet-samples", type=int, default=64,
                   help="Number of ImageNet-1K validation images to stream (default 64)")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Batch size for throughput measurement (default 32)")
    p.add_argument("--prune-ratio", type=float, default=0.5,
                   help="Prune ratio for throughput tests (default 0.5)")
    p.add_argument("--block-sizes", type=int, nargs="+",
                   default=[16, 32, 64, 128],
                   help="Power-of-2 tile sizes to test (default: 16 32 64 128)")
    p.add_argument("--no-equiv", action="store_true",
                   help="Skip numerical equivalence check (faster)")
    return p.parse_args()


# ── Monkey-patch helper ──────────────────────────────────────────────────────

def patch_block_sizes(block_m: int, block_n: int):
    """
    Replace kernels.ragged_attention.triton_ragged_attention with a version
    that hard-wires the given block sizes.  This transparently affects all
    model code that calls the kernel via the module reference.
    """
    @functools.wraps(_orig_ragged_attn)
    def _patched(q, k, v, cu_seqlens, **kwargs):
        return _orig_ragged_attn(q, k, v, cu_seqlens,
                                 block_m=block_m, block_n=block_n)
    _rag_mod.triton_ragged_attention = _patched


def restore_original():
    _rag_mod.triton_ragged_attention = _orig_ragged_attn


# ── Data loading ─────────────────────────────────────────────────────────────

def load_images(n_samples: int, dtype):
    """Stream n_samples pairs from ImageNet-1K val and stack into a tensor."""
    from benchmarks.bench5_accuracy import get_imagenet_val
    val_data = get_imagenet_val(max_samples=n_samples)
    images = torch.stack([v[0] for v in val_data]).to(DEVICE)          # [N, 3, 224, 224]
    labels = torch.tensor([v[1] for v in val_data], device=DEVICE)     # [N]
    return images, labels


# ── Kernel micro-benchmark ───────────────────────────────────────────────────

def bench_kernel_latency(block_m: int, block_n: int,
                          num_heads: int = 6, head_dim: int = 64,
                          seq_lens=(40, 80, 100, 60)) -> float:
    """
    Directly time triton_ragged_attention with realistic pruned seq-lens.
    Returns median latency in ms over BENCH_ITERS runs.
    """
    Total = sum(seq_lens)
    B = len(seq_lens)
    q = torch.randn(Total, num_heads, head_dim, device=DEVICE, dtype=torch.float16)
    k = torch.randn(Total, num_heads, head_dim, device=DEVICE, dtype=torch.float16)
    v = torch.randn(Total, num_heads, head_dim, device=DEVICE, dtype=torch.float16)
    boundaries = [0] + list(itertools.accumulate(seq_lens))
    cu_seqlens = torch.tensor(boundaries, dtype=torch.int32, device=DEVICE)

    # warmup — forces JIT compilation for this (block_m, block_n)
    for _ in range(max(WARMUP_ITERS, 5)):
        _ = _orig_ragged_attn(q, k, v, cu_seqlens, block_m=block_m, block_n=block_n)
    torch.cuda.synchronize()

    latencies = []
    for _ in range(BENCH_ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        _orig_ragged_attn(q, k, v, cu_seqlens, block_m=block_m, block_n=block_n)
        end.record()
        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end))

    latencies.sort()
    return latencies[len(latencies) // 2]   # median


# ── Throughput measurement ────────────────────────────────────────────────────

def bench_throughput(model_fn, images: torch.Tensor,
                     batch_size: int) -> float:
    """
    Returns images/sec for model_fn(images[start:end]) over BENCH_ITERS batches.
    """
    N = images.shape[0]
    batch = images[:min(batch_size, N)]

    # warmup
    for _ in range(max(WARMUP_ITERS, 3)):
        with torch.inference_mode():
            model_fn(batch)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(BENCH_ITERS):
        with torch.inference_mode():
            model_fn(batch)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return (BENCH_ITERS * len(batch)) / elapsed


# ── Numerical equivalence ─────────────────────────────────────────────────────

def check_equivalence(triton_model_fn, pytorch_model_fn, images: torch.Tensor):
    """
    Returns (max_abs_diff, predictions_match) comparing Triton vs PyTorch.
    Uses the first 4 images only (following bench5 convention).
    """
    imgs = images[:4]
    with torch.inference_mode():
        out_triton  = triton_model_fn(imgs)
        out_pytorch = pytorch_model_fn(imgs)

    diff = (out_triton.float() - out_pytorch.float()).abs()
    max_diff = diff.max().item()
    pred_match = (out_triton.argmax(-1) == out_pytorch.argmax(-1)).all().item()
    return max_diff, bool(pred_match)


# ── Model builders ────────────────────────────────────────────────────────────

def build_pytorch_baseline(prune_ratio: float):
    from baselines.pytorch_pruned import build_pytorch_pruned_model
    m = build_pytorch_pruned_model()
    return lambda imgs: m(imgs, fixed_ratio=prune_ratio)


def build_triton_model(prune_ratio: float):
    from models.triton_ragged_deit import build_triton_ragged_model
    m = build_triton_ragged_model()
    return lambda imgs: m(imgs, fixed_ratio=prune_ratio)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    os.makedirs("results", exist_ok=True)

    # ── validate block sizes ─────────────────────────────────────────
    for bs in args.block_sizes:
        assert bs >= 16 and (bs & (bs - 1)) == 0, \
            f"Block size {bs} is not a power of 2 >= 16"

    combos = [(m, n) for m in args.block_sizes for n in args.block_sizes]

    print("=" * 70)
    print("Ragged Attention Block-Size Tuning")
    print(f"  Device           : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM             : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  ImageNet samples : {args.imagenet_samples}")
    print(f"  Batch size       : {args.batch_size}")
    print(f"  Prune ratio      : {args.prune_ratio}")
    print(f"  Block combos     : {combos}")
    print("=" * 70)

    dtype = get_dtype()

    print(f"\nStreaming {args.imagenet_samples} ImageNet-1K validation images...")
    images, labels = load_images(args.imagenet_samples, dtype)

    # Build PyTorch baseline once (used for equivalence reference)
    if not args.no_equiv:
        print("Building PyTorch padded reference model...")
        pytorch_fn = build_pytorch_baseline(args.prune_ratio)

    # Reference output from default (32×32) for prediction-match comparison
    ref_block_m, ref_block_n = 32, 32
    print(f"\nBuilding reference Triton model (BLOCK_M={ref_block_m}, BLOCK_N={ref_block_n})...")
    patch_block_sizes(ref_block_m, ref_block_n)
    ref_triton_fn = build_triton_model(args.prune_ratio)
    with torch.inference_mode():
        ref_logits = ref_triton_fn(images[:4])
    ref_preds = ref_logits.argmax(-1)
    restore_original()
    del ref_triton_fn
    torch.cuda.empty_cache()

    # ── results table header ─────────────────────────────────────────
    header = f"{'BLOCK_M':>8} {'BLOCK_N':>8} {'Kernel(ms)':>12} {'Throughput':>12} {'MaxDiff':>10} {'PredMatch':>10} {'Status':>8}"
    separator = "─" * len(header)
    print(f"\n{separator}")
    print(header)
    print(separator)

    results = []
    for block_m, block_n in combos:
        label = f"M={block_m} N={block_n}"
        try:
            # 1. Kernel micro-latency (no monkey-patch needed — called directly)
            kernel_ms = bench_kernel_latency(block_m, block_n)

            # 2. End-to-end throughput (patch module-level reference)
            patch_block_sizes(block_m, block_n)
            triton_fn = build_triton_model(args.prune_ratio)
            tput = bench_throughput(triton_fn, images, args.batch_size)

            # 3. Numerical equivalence
            max_diff = float("nan")
            pred_match_pytorch = None
            if not args.no_equiv:
                max_diff, pred_match_pytorch = check_equivalence(
                    triton_fn, pytorch_fn, images)

            # 4. Prediction match vs. reference (BLOCK_M=32, BLOCK_N=32)
            with torch.inference_mode():
                this_logits = triton_fn(images[:4])
            this_preds = this_logits.argmax(-1)
            ref_match = (this_preds == ref_preds).all().item()

            restore_original()
            del triton_fn
            torch.cuda.empty_cache()
            gc.collect()

            status = "OK"
            if not args.no_equiv and max_diff > 0.01:
                status = "WARN"

            row = {
                "block_m": block_m,
                "block_n": block_n,
                "kernel_latency_ms": round(kernel_ms, 4),
                "throughput_img_s": round(tput, 2),
                "max_abs_diff_vs_pytorch": round(max_diff, 6) if not args.no_equiv else None,
                "predictions_match_pytorch": pred_match_pytorch,
                "predictions_match_ref": ref_match,
                "status": status,
            }
            results.append(row)

            diff_str  = f"{max_diff:.6f}" if not args.no_equiv else "  skipped"
            pmatch_str = str(pred_match_pytorch) if not args.no_equiv else "  skipped"
            print(f"{block_m:>8} {block_n:>8} {kernel_ms:>12.4f} {tput:>12.1f} "
                  f"{diff_str:>10} {pmatch_str:>10} {status:>8}")

        except Exception as exc:
            restore_original()
            torch.cuda.empty_cache()
            gc.collect()
            print(f"{block_m:>8} {block_n:>8}   FAILED: {exc}")
            results.append({
                "block_m": block_m, "block_n": block_n,
                "status": "ERROR", "error": str(exc),
            })

    print(separator)

    # ── find best configs ─────────────────────────────────────────────
    valid = [r for r in results if r.get("status") == "OK"]
    if valid:
        fastest_kernel = min(valid, key=lambda r: r["kernel_latency_ms"])
        fastest_e2e    = max(valid, key=lambda r: r["throughput_img_s"])
        print(f"\nBest kernel latency : M={fastest_kernel['block_m']}  "
              f"N={fastest_kernel['block_n']}  "
              f"→ {fastest_kernel['kernel_latency_ms']:.4f} ms")
        print(f"Best throughput     : M={fastest_e2e['block_m']}  "
              f"N={fastest_e2e['block_n']}  "
              f"→ {fastest_e2e['throughput_img_s']:.1f} img/s")

    # ── save results ─────────────────────────────────────────────────
    out_path = "results/block_size_tuning.json"
    with open(out_path, "w") as f:
        json.dump({
            "device": torch.cuda.get_device_name(0),
            "batch_size": args.batch_size,
            "prune_ratio": args.prune_ratio,
            "imagenet_samples": args.imagenet_samples,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
