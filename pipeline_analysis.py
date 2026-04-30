#!/usr/bin/env python3
"""
pipeline_analysis.py
====================
Three focused analyses complementing micro_benchmark.py and e2e_benchmark.py.

1. correctness    — kernel-level and E2E numerical equivalence tests
2. stage-breakdown — per-stage CUDA-event timing (pack / late-blocks / head)
3. profile        — torch.profiler Chrome trace for the Triton Ragged pipeline

All results go to --results-dir (default: results/RTX4000Ada).

Usage
-----
    python pipeline_analysis.py                       # all three
    python pipeline_analysis.py --mode correctness
    python pipeline_analysis.py --mode stage-breakdown
    python pipeline_analysis.py --mode profile
"""

import argparse
import gc
import json
import os
import statistics
import sys

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── shared infrastructure (import from sibling scripts) ──────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from micro_benchmark import (
    build_kernel_registry, _build_synthetic, cuda_event_bench,
    attn_sdpa_math, attn_triton_ragged, triton_jit_warmup,
    NUM_HEADS, HEAD_DIM, DEVICE, DTYPE, CONFIGS, WARMUP,
)
from e2e_benchmark import (
    _load_deit, _make_batch, _threshold_prune, _gather_pad,
    triton_pack_tokens,
    PaddedPipeline, TritonRaggedPipeline,
    MODEL_NAME, FIXED_RATIO,
)

try:
    from flash_attn import flash_attn_varlen_func as _fa2_fn
    _FA2_KERNEL = True
except ImportError:
    _FA2_KERNEL = False

try:
    from e2e_benchmark import FA2VarlenPipeline
    _FA2_PIPELINE = True
except Exception:
    _FA2_PIPELINE = False

# ── constants ─────────────────────────────────────────────────────────────────
CORRECTNESS_TOL   = 1e-2   # max-abs threshold for float16 (matches bench5)
STAGE_BATCH_SIZES = [1, 8, 32, 128]
STAGE_ITERS       = 100
PROFILE_BATCH     = 8
PROFILE_WARMUP    = 2
PROFILE_ACTIVE    = 5


# ═════════════════════════════════════════════════════════════════════════════
# 1. Correctness
# ═════════════════════════════════════════════════════════════════════════════

def run_correctness(results_dir="results/RTX4000Ada"):
    """
    1a. Kernel-level: compare Triton (and FA2 if available) outputs against
        SDPA-math (exact softmax, no approximation) on synthetic QKV tensors
        at each micro-benchmark config.

    1b. E2E-level: run the full Padded and Triton pipelines on the same random
        images and compare final logits and top-1 predictions.
    """
    print("\n" + "=" * 70)
    print("Correctness Tests")
    print("=" * 70)

    results = {
        "gpu":       torch.cuda.get_device_name(0),
        "tolerance": CORRECTNESS_TOL,
        "kernel":    {},
        "e2e":       {},
    }

    # ── 1a: kernel-level ────────────────────────────────────────────────────
    print("\n[1/2] Kernel — Triton vs SDPA-math reference")
    print("-" * 50)

    for bs, seq_len, _ in CONFIGS:
        q, k, v, cu, _ = _build_synthetic(bs, seq_len)
        triton_jit_warmup(q, k, v, cu, label=f"BS={bs} L={seq_len}")
        del q, k, v, cu
        torch.cuda.empty_cache()

    all_kernel_pass = True
    for bs, seq_len, label in CONFIGS:
        torch.manual_seed(42)
        q, k, v, cu, max_len = _build_synthetic(bs, seq_len)

        with torch.inference_mode():
            ref = attn_sdpa_math(q, k, v, cu, max_len).float()
            tri = attn_triton_ragged(q, k, v, cu, max_len).float()

        tri_max  = (ref - tri).abs().max().item()
        tri_mean = (ref - tri).abs().mean().item()
        tri_pass = tri_max < CORRECTNESS_TOL

        entry = {
            "triton_max_abs_diff":  round(tri_max,  6),
            "triton_mean_abs_diff": round(tri_mean, 6),
            "triton_passed":        tri_pass,
        }

        line = (f"  {label:<25}  Triton  max={tri_max:.2e}  mean={tri_mean:.2e}"
                f"  {'PASS' if tri_pass else 'FAIL'}")

        if _FA2_KERNEL:
            fa2 = _fa2_fn(
                q, k, v,
                cu_seqlens_q=cu, cu_seqlens_k=cu,
                max_seqlen_q=max_len, max_seqlen_k=max_len,
                causal=False,
            ).float()
            fa2_max  = (ref - fa2).abs().max().item()
            fa2_mean = (ref - fa2).abs().mean().item()
            fa2_pass = fa2_max < CORRECTNESS_TOL
            entry.update({
                "fa2_max_abs_diff":  round(fa2_max,  6),
                "fa2_mean_abs_diff": round(fa2_mean, 6),
                "fa2_passed":        fa2_pass,
            })
            line += f"  |  FA2  max={fa2_max:.2e}  {'PASS' if fa2_pass else 'FAIL'}"

        print(line)
        results["kernel"][label] = entry
        if not tri_pass:
            all_kernel_pass = False

        del q, k, v, cu
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n  → Kernel: {'ALL PASS' if all_kernel_pass else 'FAILURES DETECTED'}")

    # ── 1b: E2E-level ───────────────────────────────────────────────────────
    print("\n[2/2] E2E — Padded vs Triton pipeline logit comparison")
    print("-" * 50)

    deit        = _load_deit()
    padded_pipe = PaddedPipeline(deit).to(DEVICE, dtype=DTYPE).eval()
    triton_pipe = TritonRaggedPipeline(deit).to(DEVICE, dtype=DTYPE).eval()
    if _FA2_PIPELINE:
        fa2_pipe = FA2VarlenPipeline(deit).to(DEVICE, dtype=DTYPE).eval()
    del deit
    torch.cuda.empty_cache()

    all_e2e_pass = True
    for bs in [1, 4, 16]:
        torch.manual_seed(7)
        images = _make_batch(bs)

        with torch.inference_mode():
            logits_pad = padded_pipe(images).float()
            logits_tri = triton_pipe(images).float()

        tri_max   = (logits_pad - logits_tri).abs().max().item()
        tri_mean  = (logits_pad - logits_tri).abs().mean().item()
        preds_ok  = (logits_pad.argmax(1) == logits_tri.argmax(1)).all().item()
        tri_pass  = tri_max < CORRECTNESS_TOL

        entry = {
            "triton_max_abs_diff":    round(tri_max,  6),
            "triton_mean_abs_diff":   round(tri_mean, 6),
            "triton_predictions_match": bool(preds_ok),
            "triton_passed":          tri_pass,
        }

        line = (f"  BS={bs:<3}  max={tri_max:.2e}  mean={tri_mean:.2e}"
                f"  preds={'match' if preds_ok else 'DIFFER'}"
                f"  {'PASS' if tri_pass else 'FAIL'}")

        if _FA2_PIPELINE:
            with torch.inference_mode():
                logits_fa2 = fa2_pipe(images).float()
            fa2_max  = (logits_pad - logits_fa2).abs().max().item()
            fa2_ok   = (logits_pad.argmax(1) == logits_fa2.argmax(1)).all().item()
            entry.update({
                "fa2_max_abs_diff":      round(fa2_max, 6),
                "fa2_predictions_match": bool(fa2_ok),
            })
            line += f"  |  FA2  max={fa2_max:.2e}  preds={'match' if fa2_ok else 'DIFFER'}"

        print(line)
        results["e2e"][f"BS={bs}"] = entry
        if not tri_pass:
            all_e2e_pass = False

        del images
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n  → E2E: {'ALL PASS' if all_e2e_pass else 'FAILURES DETECTED'}")

    del padded_pipe, triton_pipe
    if _FA2_PIPELINE:
        del fa2_pipe
    torch.cuda.empty_cache()

    return results


# ═════════════════════════════════════════════════════════════════════════════
# 2. Stage breakdown
# ═════════════════════════════════════════════════════════════════════════════

def _bench_stage(fn, *args, warmup=20, iters=STAGE_ITERS):
    """CUDA-event timing for a single pipeline stage. Returns median_ms."""
    with torch.inference_mode():
        for _ in range(warmup):
            fn(*args)
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            fn(*args)
            ends[i].record()
        torch.cuda.synchronize()
        times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
        return statistics.median(times)


def run_stage_breakdown(results_dir="results/RTX4000Ada"):
    """
    Times each pipeline stage independently with pre-computed fixed inputs,
    so measurements are free of inter-stage synchronisation bubbles.

    Stages
    ------
    front_early  : patch embed + pos embed + first PRUNE_AFTER blocks
                   (identical for both pipelines)
    pack / gather: triton_pack_tokens  vs  _gather_pad + additive mask build
    late_blocks  : remaining transformer blocks — ragged vs padded attention
    head         : CLS token extract + LayerNorm + linear classifier
    """
    print("\n" + "=" * 70)
    print("Stage Breakdown")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}  |  pruning: {int(FIXED_RATIO*100)}%  |  iters: {STAGE_ITERS}\n")

    deit        = _load_deit()
    padded_pipe = PaddedPipeline(deit).to(DEVICE, dtype=DTYPE).eval()
    triton_pipe = TritonRaggedPipeline(deit).to(DEVICE, dtype=DTYPE).eval()
    del deit
    torch.cuda.empty_cache()

    results = {
        "gpu":         torch.cuda.get_device_name(0),
        "model":       MODEL_NAME,
        "fixed_ratio": FIXED_RATIO,
        "batch_sizes": STAGE_BATCH_SIZES,
        "padded":      {},
        "triton":      {},
    }

    for bs in STAGE_BATCH_SIZES:
        print(f"[BS={bs}]")
        images = _make_batch(bs)

        # Pre-compute fixed inputs for each stage
        with torch.inference_mode():
            x    = triton_pipe.early(triton_pipe.front(images))
            mask = _threshold_prune(x, FIXED_RATIO)

            x_pad, attn_m = _gather_pad(x, mask)
            B, S, D = x_pad.shape
            amask = x_pad.new_zeros(B, 1, 1, S)
            amask.masked_fill_(~attn_m.unsqueeze(1).unsqueeze(2), float("-inf"))
            x_pad_out = x_pad.clone()
            for blk in padded_pipe.late_blocks:
                x_pad_out = blk(x_pad_out, amask)

            packed, cu = triton_pack_tokens(x, mask)
            packed_out = packed.clone()
            for blk in triton_pipe.ragged_blocks:
                packed_out = blk(packed_out, cu)

        # Capture loop variables explicitly to avoid closure issues
        _cu = cu

        def run_front_early(img):
            return triton_pipe.early(triton_pipe.front(img))

        def run_gather(x_, m_):
            xp, am = _gather_pad(x_, m_)
            B_, S_, _ = xp.shape
            msk = xp.new_zeros(B_, 1, 1, S_)
            msk.masked_fill_(~am.unsqueeze(1).unsqueeze(2), float("-inf"))
            return xp, msk

        def run_pack(x_, m_):
            return triton_pack_tokens(x_, m_)

        def run_padded_late(xp_, am_):
            for blk in padded_pipe.late_blocks:
                xp_ = blk(xp_, am_)
            return xp_

        def run_triton_late(p_, c_):
            for blk in triton_pipe.ragged_blocks:
                p_ = blk(p_, c_)
            return p_

        def run_padded_head(xp_):
            return padded_pipe.head(padded_pipe.norm(xp_[:, 0]))

        def run_triton_head(p_, c_=_cu):
            return triton_pipe.head(triton_pipe.norm(p_[c_[:-1].long()]))

        t_front  = _bench_stage(run_front_early, images)
        t_gather = _bench_stage(run_gather,      x, mask)
        t_pack   = _bench_stage(run_pack,        x, mask)
        t_p_late = _bench_stage(run_padded_late, x_pad,    amask)
        t_t_late = _bench_stage(run_triton_late, packed,   cu)
        t_p_head = _bench_stage(run_padded_head, x_pad_out)
        t_t_head = _bench_stage(run_triton_head, packed_out)

        p_total = t_front + t_gather + t_p_late + t_p_head
        t_total = t_front + t_pack   + t_t_late + t_t_head

        results["padded"][bs] = {
            "front_early_ms": round(t_front,  3),
            "gather_ms":      round(t_gather, 3),
            "late_blocks_ms": round(t_p_late, 3),
            "head_ms":        round(t_p_head, 3),
        }
        results["triton"][bs] = {
            "front_early_ms": round(t_front,  3),
            "pack_ms":        round(t_pack,   3),
            "late_blocks_ms": round(t_t_late, 3),
            "head_ms":        round(t_t_head, 3),
        }

        print(f"  Padded  front={t_front:.2f}  gather={t_gather:.2f}  "
              f"late={t_p_late:.2f}  head={t_p_head:.2f}  ≈{p_total:.2f} ms")
        print(f"  Triton  front={t_front:.2f}  pack={t_pack:.2f}    "
              f"late={t_t_late:.2f}  head={t_t_head:.2f}  ≈{t_total:.2f} ms")
        print(f"  Late-blocks speedup: {t_p_late / t_t_late:.2f}×\n")

        del images, x, mask, x_pad, attn_m, amask, x_pad_out, packed, cu, packed_out
        torch.cuda.empty_cache()
        gc.collect()

    return results


def plot_stage_breakdown(results, results_dir):
    batch_sizes = results["batch_sizes"]
    n = len(batch_sizes)

    clr = {
        "front_early": "#aec7e8",
        "gather":      "#d62728",
        "pack":        "#2ca02c",
        "late_blocks": "#ff7f0e",
        "head":        "#bcbd22",
    }

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, bs in zip(axes, batch_sizes):
        p = results["padded"][bs]
        t = results["triton"][bs]

        p_stages = [
            (p["front_early_ms"], clr["front_early"]),
            (p["gather_ms"],      clr["gather"]),
            (p["late_blocks_ms"], clr["late_blocks"]),
            (p["head_ms"],        clr["head"]),
        ]
        t_stages = [
            (t["front_early_ms"], clr["front_early"]),
            (t["pack_ms"],        clr["pack"]),
            (t["late_blocks_ms"], clr["late_blocks"]),
            (t["head_ms"],        clr["head"]),
        ]

        bar_w = 0.35
        x_pos = [0.0, bar_w + 0.15]

        for xi, stages in zip(x_pos, [p_stages, t_stages]):
            bottom = 0.0
            for val, color in stages:
                ax.bar(xi, val, bar_w, bottom=bottom, color=color,
                       edgecolor="white", linewidth=0.5)
                if val >= 0.5:
                    ax.text(xi, bottom + val / 2, f"{val:.1f}",
                            ha="center", va="center", fontsize=7)
                bottom += val

        ax.set_xticks(x_pos)
        ax.set_xticklabels(["Padded", "Triton"], fontsize=10)
        ax.set_title(f"BS={bs}", fontsize=10, fontweight="bold")
        ax.set_ylabel("Latency (ms)" if bs == batch_sizes[0] else "", fontsize=10)
        ax.grid(True, axis="y", alpha=0.25)

    legend_handles = [
        Patch(color=clr["front_early"], label="front + early blocks"),
        Patch(color=clr["gather"],      label="gather + mask  [Padded]"),
        Patch(color=clr["pack"],        label="pack  [Triton]"),
        Patch(color=clr["late_blocks"], label="late blocks  (attn + MLP)"),
        Patch(color=clr["head"],        label="head"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=8,
               bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(
        f"Pipeline Stage Breakdown — DeiT-B, {int(results['fixed_ratio']*100)}% pruned\n"
        "Stages timed independently with fixed inputs, CUDA-event timing",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 0.82, 1.0])
    os.makedirs(results_dir, exist_ok=True)
    out = os.path.join(results_dir, "stage_breakdown.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✓  Saved {out}")


# ═════════════════════════════════════════════════════════════════════════════
# 3. Profiling
# ═════════════════════════════════════════════════════════════════════════════

def run_profile(results_dir="results/RTX4000Ada"):
    """
    Runs torch.profiler over the Triton Ragged pipeline and exports a Chrome
    trace JSON.  Open it at https://ui.perfetto.dev or chrome://tracing.

    Also prints the top 15 CUDA ops by self_cuda_time_total so the dominant
    kernels are visible without needing a trace viewer.
    """
    from torch.profiler import profile, ProfilerActivity, schedule

    print("\n" + "=" * 70)
    print("torch.profiler — Triton Ragged pipeline")
    print("=" * 70)
    print(f"BS={PROFILE_BATCH}, {int(FIXED_RATIO*100)}% pruned\n")

    deit        = _load_deit()
    triton_pipe = TritonRaggedPipeline(deit).to(DEVICE, dtype=DTYPE).eval()
    del deit
    torch.cuda.empty_cache()

    images = _make_batch(PROFILE_BATCH)

    with torch.inference_mode():
        for _ in range(10):
            triton_pipe(images)
    torch.cuda.synchronize()

    os.makedirs(results_dir, exist_ok=True)
    trace_path = os.path.join(results_dir, "profile_trace.json")

    n_steps = PROFILE_WARMUP + PROFILE_ACTIVE

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=0, warmup=PROFILE_WARMUP, active=PROFILE_ACTIVE),
        on_trace_ready=lambda p: p.export_chrome_trace(trace_path),
        record_shapes=True,
        with_stack=False,
    ) as prof:
        with torch.inference_mode():
            for _ in range(n_steps):
                triton_pipe(images)
                prof.step()

    torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))
    print(f"\n✓  Chrome trace → {trace_path}")
    print("    View at  https://ui.perfetto.dev  (drag-and-drop the JSON)")

    del images, triton_pipe
    torch.cuda.empty_cache()

    return {"trace_path": trace_path}


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", default="results/RTX4000Ada",
                        help="directory for JSON + PNG output")
    parser.add_argument("--mode",
                        choices=["correctness", "stage-breakdown", "profile", "all"],
                        default="all",
                        help="which analysis to run (default: all)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA not available")

    os.makedirs(args.results_dir, exist_ok=True)

    if args.mode in ("correctness", "all"):
        r = run_correctness(args.results_dir)
        path = os.path.join(args.results_dir, "correctness.json")
        with open(path, "w") as f:
            json.dump(r, f, indent=2)
        print(f"✓  Saved {path}")

    if args.mode in ("stage-breakdown", "all"):
        r = run_stage_breakdown(args.results_dir)
        path = os.path.join(args.results_dir, "stage_breakdown.json")
        with open(path, "w") as f:
            json.dump(r, f, indent=2)
        print(f"✓  Saved {path}")
        plot_stage_breakdown(r, args.results_dir)

    if args.mode in ("profile", "all"):
        r = run_profile(args.results_dir)
        path = os.path.join(args.results_dir, "profile_summary.json")
        with open(path, "w") as f:
            json.dump(r, f, indent=2)
        print(f"✓  Saved {path}")


if __name__ == "__main__":
    main()
