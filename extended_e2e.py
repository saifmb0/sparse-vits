#!/usr/bin/env python3
"""
extended_e2e.py
===============
Extended E2E benchmarks on RTX 4000 Ada:

1. model-scaling  — DeiT-Ti / DeiT-S / DeiT-B, fixed 50% pruning
2. sparsity-sweep — DeiT-B, pruning ratio 0–90%, fixed BS=32
3. high-res       — DeiT-B at 384×384 (577 tokens) vs 224×224 (197 tokens),
                    50% pruning, comparing padded pipeline vs Triton ragged.
                    At 384×384 the padded pipeline allocates a 577-token buffer
                    even after pruning to 288 tokens; Triton processes 288 only.

Note on multi-pruning throughput: every pruning algorithm (EViT, ATS,
DynamicViT, Threshold-L2) produces a boolean keep-mask of fixed size
(1-ratio)×S.  The Triton kernel depends only on the count of kept tokens,
not their identity, so throughput is identical across methods at the same
(ratio, BS).  A multi-method throughput benchmark is therefore uninformative;
correctness across mask patterns is covered by pipeline_analysis.py.

Usage
-----
    python extended_e2e.py                            # all modes
    python extended_e2e.py --mode model-scaling
    python extended_e2e.py --mode sparsity-sweep
    python extended_e2e.py --mode high-res
    python extended_e2e.py --results-dir results/RTX4000Ada
"""

import argparse
import gc
import json
import os
import sys

import timm
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

torch.backends.cudnn.enabled = False   # matches e2e_benchmark.py

from e2e_benchmark import (
    _make_batch, _threshold_prune, _gather_pad,
    triton_pack_tokens,
    PaddedPipeline, TritonRaggedPipeline,
    cuda_event_bench,
    DEVICE, DTYPE, WARMUP_ITERS, BENCH_ITERS, PRUNE_AFTER,
)

try:
    from e2e_benchmark import FA2VarlenPipeline
    from flash_attn import flash_attn_varlen_func  # noqa — verify availability
    _FA2 = True
except Exception:
    _FA2 = False

# ── constants ─────────────────────────────────────────────────────────────────
# 384×384 / patch16: (384/16)² + 1 CLS = 576 + 1 = 577 tokens
HIGHRES_MODEL     = "deit_base_patch16_384"
HIGHRES_IMG_SIZE  = 384
HIGHRES_TOKENS    = 577   # 576 patches + 1 CLS
HIGHRES_RATIO     = 0.5   # 50% pruning → ~288 actual tokens, max=577
HIGHRES_BS        = [1, 4, 8, 16, 32, 64, 128]

# 224×224 baseline for direct comparison in the same plot
BASELINE_MODEL    = "deit_base_patch16_224"
BASELINE_TOKENS   = 197   # 196 patches + 1 CLS

SCALING_MODELS = [
    ("deit_tiny_patch16_224",  "DeiT-Ti"),
    ("deit_small_patch16_224", "DeiT-S"),
    ("deit_base_patch16_224",  "DeiT-B"),
]
SCALING_BS    = [1, 4, 8, 16, 32, 64, 128]
SCALING_RATIO = 0.5

SWEEP_RATIOS  = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SWEEP_BS      = 32
SWEEP_MODEL   = "deit_base_patch16_224"


# ── helpers ───────────────────────────────────────────────────────────────────

def _load(model_name):
    m = timm.create_model(model_name, pretrained=True)
    return m.to(DEVICE, dtype=DTYPE).eval()


def _build_pipelines(deit):
    padded = PaddedPipeline(deit).to(DEVICE, dtype=DTYPE).eval()
    triton = TritonRaggedPipeline(deit).to(DEVICE, dtype=DTYPE).eval()
    fa2    = FA2VarlenPipeline(deit).to(DEVICE, dtype=DTYPE).eval() if _FA2 else None
    return padded, triton, fa2


# Ratio-parameterised forwards — the existing pipeline classes hard-code
# FIXED_RATIO, so we re-implement the forward here with an explicit ratio arg.

@torch.inference_mode()
def _padded_fwd(pipe, images, ratio):
    x     = pipe.early(pipe.front(images))
    mask  = _threshold_prune(x, ratio)
    x_pad, attn_m = _gather_pad(x, mask)
    B, S, D = x_pad.shape
    amask = x_pad.new_zeros(B, 1, 1, S)
    amask.masked_fill_(~attn_m.unsqueeze(1).unsqueeze(2), float("-inf"))
    for blk in pipe.late_blocks:
        x_pad = blk(x_pad, amask)
    return pipe.head(pipe.norm(x_pad[:, 0]))


@torch.inference_mode()
def _triton_fwd(pipe, images, ratio):
    x      = pipe.early(pipe.front(images))
    mask   = _threshold_prune(x, ratio)
    packed, cu = triton_pack_tokens(x, mask)
    for blk in pipe.ragged_blocks:
        packed = blk(packed, cu)
    return pipe.head(pipe.norm(packed[cu[:-1].long()]))


@torch.inference_mode()
def _fa2_fwd(pipe, images, ratio):
    x      = pipe.early(pipe.front(images))
    mask   = _threshold_prune(x, ratio)
    packed, cu = triton_pack_tokens(x, mask)
    max_seqlen = int((cu[1:] - cu[:-1]).max().item())
    for blk in pipe.fa2_blocks:
        packed = blk(packed, cu, max_seqlen)
    return pipe.head(pipe.norm(packed[cu[:-1].long()]))


def _run(fn, images):
    """Thin wrapper so cuda_event_bench receives a model-style callable."""
    class _W(nn.Module):
        def forward(self, x): return fn(x)
    return cuda_event_bench(_W(), images, warmup=WARMUP_ITERS, iters=BENCH_ITERS)


# ═════════════════════════════════════════════════════════════════════════════
# 1. Model scaling
# ═════════════════════════════════════════════════════════════════════════════

def run_model_scaling(results_dir):
    print("\n" + "=" * 70)
    print("Model Scaling — DeiT-Ti / DeiT-S / DeiT-B")
    print("=" * 70)
    print(f"Pruning: {int(SCALING_RATIO*100)}%  |  BS: {SCALING_BS}\n")

    results = {
        "gpu":         torch.cuda.get_device_name(0),
        "fixed_ratio": SCALING_RATIO,
        "batch_sizes": SCALING_BS,
        "models":      {},
    }

    for model_name, label in SCALING_MODELS:
        print(f"\n── {label} ──")
        deit = _load(model_name)
        padded_pipe, triton_pipe, fa2_pipe = _build_pipelines(deit)
        del deit
        torch.cuda.empty_cache()

        entry = {"padded": [], "triton": [], "padded_ms": [], "triton_ms": []}
        if _FA2:
            entry.update({"fa2": [], "fa2_ms": []})

        for bs in SCALING_BS:
            images = _make_batch(bs)

            tp_p, ms_p, _, _ = _run(lambda img, p=padded_pipe: _padded_fwd(p, img, SCALING_RATIO), images)
            tp_t, ms_t, _, _ = _run(lambda img, p=triton_pipe: _triton_fwd(p, img, SCALING_RATIO), images)

            entry["padded"].append(round(tp_p, 1))
            entry["triton"].append(round(tp_t, 1))
            entry["padded_ms"].append(round(ms_p, 3))
            entry["triton_ms"].append(round(ms_t, 3))

            line = f"  BS={bs:<4} Padded={tp_p:7.1f}  Triton={tp_t:7.1f}  ({tp_t/tp_p:.2f}×)"

            if _FA2:
                tp_f, ms_f, _, _ = _run(lambda img, p=fa2_pipe: _fa2_fwd(p, img, SCALING_RATIO), images)
                entry["fa2"].append(round(tp_f, 1))
                entry["fa2_ms"].append(round(ms_f, 3))
                line += f"  FA2={tp_f:7.1f}"

            print(line)
            del images
            torch.cuda.empty_cache()
            gc.collect()

        results["models"][label] = entry

        del padded_pipe, triton_pipe
        if _FA2:
            del fa2_pipe
        torch.cuda.empty_cache()
        gc.collect()

    return results


def plot_model_scaling(results, results_dir):
    bs_list = results["batch_sizes"]
    models  = results["models"]

    colors = {"DeiT-Ti": "#2ca02c", "DeiT-S": "#1f77b4", "DeiT-B": "#ff7f0e"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for label, data in models.items():
        c = colors.get(label, "gray")
        ax1.plot(bs_list, data["triton"], "D-",  color=c, label=f"{label} Triton", lw=2, ms=7)
        ax1.plot(bs_list, data["padded"], "s--", color=c, label=f"{label} Padded", lw=1, ms=5, alpha=0.5)
        if "fa2" in data:
            ax1.plot(bs_list, data["fa2"], "^:", color=c, lw=1, ms=5, alpha=0.5,
                     label=f"{label} FA2")
            fa2_speedups = [t / f for t, f in zip(data["triton"], data["fa2"])]
            ax2.plot(bs_list, fa2_speedups, "D-", color=c, label=label, lw=2, ms=7)

    ax1.set_xlabel("Batch size", fontsize=12)
    ax1.set_ylabel("Throughput (img/s)", fontsize=11)
    ax1.set_title(f"Throughput — {int(results['fixed_ratio']*100)}% pruned", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1.2, label="break-even")
    ax2.set_xlabel("Batch size", fontsize=12)
    ax2.set_ylabel("Speedup  Triton / FA2 varlen", fontsize=11)
    ax2.set_title("Triton vs FA2 varlen Speedup by Model", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"Model Scaling — DeiT-Ti/S/B, {int(results['fixed_ratio']*100)}% pruned\n"
        "RTX 4000 Ada (SM89), CUDA-event timing",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    os.makedirs(results_dir, exist_ok=True)
    out = os.path.join(results_dir, "model_scaling.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✓  Saved {out}")


# ═════════════════════════════════════════════════════════════════════════════
# 2. Sparsity sweep
# ═════════════════════════════════════════════════════════════════════════════

def run_sparsity_sweep(results_dir):
    print("\n" + "=" * 70)
    print(f"Sparsity Sweep — {SWEEP_MODEL}, BS={SWEEP_BS}")
    print("=" * 70)
    print(f"Ratios: {SWEEP_RATIOS}\n")

    results = {
        "gpu":        torch.cuda.get_device_name(0),
        "model":      SWEEP_MODEL,
        "batch_size": SWEEP_BS,
        "ratios":     SWEEP_RATIOS,
        "padded":     [], "triton":    [],
        "padded_ms":  [], "triton_ms": [],
    }
    if _FA2:
        results.update({"fa2": [], "fa2_ms": []})

    deit = _load(SWEEP_MODEL)
    padded_pipe, triton_pipe, fa2_pipe = _build_pipelines(deit)
    del deit
    torch.cuda.empty_cache()

    images = _make_batch(SWEEP_BS)

    for ratio in SWEEP_RATIOS:
        n_kept = round(197 * (1.0 - ratio))
        tp_p, ms_p, _, _ = _run(lambda img, r=ratio: _padded_fwd(padded_pipe, img, r), images)
        tp_t, ms_t, _, _ = _run(lambda img, r=ratio: _triton_fwd(triton_pipe, img, r), images)

        results["padded"].append(round(tp_p, 1))
        results["triton"].append(round(tp_t, 1))
        results["padded_ms"].append(round(ms_p, 3))
        results["triton_ms"].append(round(ms_t, 3))

        line = (f"  {int(ratio*100):2d}% pruned  ~{n_kept:3d} tokens"
                f"  Padded={tp_p:7.1f}  Triton={tp_t:7.1f}  ({tp_t/tp_p:.2f}×)")

        if _FA2:
            tp_f, ms_f, _, _ = _run(lambda img, r=ratio: _fa2_fwd(fa2_pipe, img, r), images)
            results["fa2"].append(round(tp_f, 1))
            results["fa2_ms"].append(round(ms_f, 3))
            line += f"  FA2={tp_f:7.1f}"

        print(line)
        torch.cuda.empty_cache()
        gc.collect()

    del images, padded_pipe, triton_pipe
    if _FA2:
        del fa2_pipe
    torch.cuda.empty_cache()

    return results


def plot_sparsity_sweep(results, results_dir):
    ratios   = results["ratios"]
    pct      = [int(r * 100) for r in ratios]
    padded   = results["padded"]
    triton   = results["triton"]
    speedups = [t / p for t, p in zip(triton, padded)]

    # Theoretical speedup: attention is O(N²), linear layers O(N).
    # Late blocks = 8 transformer layers, each with O(N²) attn + O(N) MLP.
    # Rough split: ~20% attn, ~80% linear → weighted theoretical speedup.
    theo = [1.0 / (0.2 * (1 - r)**2 + 0.8 * (1 - r)) for r in ratios]
    # Normalise so theo[0] == 1 (no pruning, no speedup)
    theo = [t / theo[0] for t in theo]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(pct, padded, "s--", color="#d62728", label="PyTorch Padded (SDPA)", lw=2, ms=7)
    ax1.plot(pct, triton, "D-",  color="#2ca02c", label="Triton Ragged (ours)",  lw=2, ms=7)
    if _FA2 and "fa2" in results:
        ax1.plot(pct, results["fa2"], "^-", color="#9467bd",
                 label="FlashAttention-2 (varlen)", lw=2, ms=7)
    ax1.set_xlabel("Pruning ratio (%)", fontsize=12)
    ax1.set_ylabel("Throughput (img/s)", fontsize=11)
    ax1.set_title(f"Sparsity–Throughput — DeiT-B, BS={results['batch_size']}",
                  fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(pct, speedups, "s--", color="#d62728", label="Triton / Padded SDPA", lw=1.5, ms=6, alpha=0.7)
    ax2.plot(pct, theo,     "k--", label="Theoretical (0.2·N² + 0.8·N)", lw=1.5, alpha=0.5)
    if _FA2 and "fa2" in results:
        fa2_speedups = [t / f for t, f in zip(results["triton"], results["fa2"])]
        ax2.plot(pct, fa2_speedups, "D-", color="#2ca02c", label="Triton / FA2 varlen", lw=2, ms=7)
    ax2.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax2.set_xlabel("Pruning ratio (%)", fontsize=12)
    ax2.set_ylabel("Speedup (Triton as numerator)", fontsize=11)
    ax2.set_title("Triton Speedup vs Pruning Ratio", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"Sparsity Sweep — DeiT-B, BS={results['batch_size']}\n"
        "RTX 4000 Ada (SM89), CUDA-event timing",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    os.makedirs(results_dir, exist_ok=True)
    out = os.path.join(results_dir, "sparsity_sweep.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✓  Saved {out}")


# ═════════════════════════════════════════════════════════════════════════════
# 3. High-resolution (384×384 vs 224×224)
# ═════════════════════════════════════════════════════════════════════════════

def run_high_res(results_dir):
    print("\n" + "=" * 70)
    print(f"High-resolution E2E — 384×384 vs 224×224, {int(HIGHRES_RATIO*100)}% pruned")
    print("=" * 70)
    print(f"  224² baseline: {BASELINE_TOKENS} tokens → ~{round(BASELINE_TOKENS*(1-HIGHRES_RATIO))} kept")
    print(f"  384² high-res: {HIGHRES_TOKENS} tokens → ~{round(HIGHRES_TOKENS*(1-HIGHRES_RATIO))} kept")
    print(f"  Padded buffers: 197 vs 577 tokens.  BS: {HIGHRES_BS}\n")

    results = {
        "gpu":          torch.cuda.get_device_name(0),
        "fixed_ratio":  HIGHRES_RATIO,
        "batch_sizes":  HIGHRES_BS,
        "resolutions":  {},
    }

    configs = [
        (BASELINE_MODEL, "224²", 224),
        (HIGHRES_MODEL,  "384²", HIGHRES_IMG_SIZE),
    ]

    for model_name, res_label, img_size in configs:
        print(f"\n── DeiT-B {res_label} ──")
        deit = _load(model_name)
        padded_pipe, triton_pipe, fa2_pipe = _build_pipelines(deit)
        del deit
        torch.cuda.empty_cache()

        entry = {
            "padded": [], "triton": [],
            "padded_ms": [], "triton_ms": [],
        }
        if _FA2:
            entry.update({"fa2": [], "fa2_ms": []})

        for bs in HIGHRES_BS:
            images = torch.randn(bs, 3, img_size, img_size, device=DEVICE, dtype=DTYPE)

            tp_p, ms_p, _, _ = _run(
                lambda img, p=padded_pipe: _padded_fwd(p, img, HIGHRES_RATIO), images)
            tp_t, ms_t, _, _ = _run(
                lambda img, p=triton_pipe: _triton_fwd(p, img, HIGHRES_RATIO), images)

            entry["padded"].append(round(tp_p, 1))
            entry["triton"].append(round(tp_t, 1))
            entry["padded_ms"].append(round(ms_p, 3))
            entry["triton_ms"].append(round(ms_t, 3))

            line = (f"  BS={bs:<4}  Padded={tp_p:7.1f}  Triton={tp_t:7.1f}"
                    f"  ({tp_t/tp_p:.2f}×)")

            if _FA2:
                tp_f, ms_f, _, _ = _run(
                    lambda img, p=fa2_pipe: _fa2_fwd(p, img, HIGHRES_RATIO), images)
                entry["fa2"].append(round(tp_f, 1))
                entry["fa2_ms"].append(round(ms_f, 3))
                line += f"  FA2={tp_f:7.1f}"

            print(line)
            del images
            torch.cuda.empty_cache()
            gc.collect()

        results["resolutions"][res_label] = entry

        del padded_pipe, triton_pipe
        if _FA2:
            del fa2_pipe
        torch.cuda.empty_cache()
        gc.collect()

    return results


def plot_high_res(results, results_dir):
    bs_list = results["batch_sizes"]
    resolutions = results["resolutions"]

    res_colors = {"224²": "#1f77b4", "384²": "#d62728"}

    fig, axes = plt.subplots(1, len(resolutions), figsize=(7 * len(resolutions), 5), squeeze=False)

    speedup_data = {}
    padded_speedup_data = {}
    for col, (res_label, data) in enumerate(resolutions.items()):
        ax = axes[0][col]
        c  = res_colors.get(res_label, "gray")

        ax.plot(bs_list, data["padded"], "s--", color=c,
                label="Padded (SDPA)", lw=2, ms=7)
        ax.plot(bs_list, data["triton"], "D-", color=c,
                label="Triton Ragged (ours)", lw=2, ms=7, alpha=0.9,
                markerfacecolor="white", markeredgewidth=2)
        if _FA2 and "fa2" in data:
            ax.plot(bs_list, data["fa2"], "^-", color=c,
                    label="FA2 (varlen)", lw=2, ms=7, linestyle=":")

        ax.set_xlabel("Batch size", fontsize=12)
        ax.set_ylabel("Throughput (img/s)", fontsize=11)
        ax.set_title(f"DeiT-B {res_label}, {int(results['fixed_ratio']*100)}% pruned",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        padded_speedup_data[res_label] = [t / p for t, p in zip(data["triton"], data["padded"])]
        if _FA2 and "fa2" in data:
            speedup_data[res_label] = [t / f for t, f in zip(data["triton"], data["fa2"])]

    # Speedup comparison subplot (separate figure) — primary: Triton vs FA2
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for res_label, speedups in speedup_data.items():
        c = res_colors.get(res_label, "gray")
        ax2.plot(bs_list, speedups, "D-", color=c,
                 label=f"Triton / FA2 — DeiT-B {res_label}",
                 lw=2, ms=7)
    for res_label, speedups in padded_speedup_data.items():
        c = res_colors.get(res_label, "gray")
        ax2.plot(bs_list, speedups, "s--", color=c,
                 label=f"Triton / Padded — DeiT-B {res_label}",
                 lw=1.2, ms=5, alpha=0.5)
    ax2.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax2.set_xlabel("Batch size", fontsize=12)
    ax2.set_ylabel("Speedup (Triton as numerator)", fontsize=11)
    ax2.set_title("Triton Speedup: 384×384 vs 224×224 Inputs\nRTX 4000 Ada (SM89)",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    out2 = os.path.join(results_dir, "high_res_speedup.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"✓  Saved {out2}")

    plt.tight_layout()
    os.makedirs(results_dir, exist_ok=True)
    out = os.path.join(results_dir, "high_res.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓  Saved {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", default="results/RTX4000Ada")
    parser.add_argument("--mode",
                        choices=["model-scaling", "sparsity-sweep", "high-res", "all"],
                        default="all")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA not available")

    os.makedirs(args.results_dir, exist_ok=True)

    if args.mode in ("model-scaling", "all"):
        r = run_model_scaling(args.results_dir)
        path = os.path.join(args.results_dir, "model_scaling.json")
        with open(path, "w") as f:
            json.dump(r, f, indent=2)
        print(f"✓  Saved {path}")
        plot_model_scaling(r, args.results_dir)

    if args.mode in ("sparsity-sweep", "all"):
        r = run_sparsity_sweep(args.results_dir)
        path = os.path.join(args.results_dir, "sparsity_sweep.json")
        with open(path, "w") as f:
            json.dump(r, f, indent=2)
        print(f"✓  Saved {path}")
        plot_sparsity_sweep(r, args.results_dir)

    if args.mode in ("high-res", "all"):
        r = run_high_res(args.results_dir)
        path = os.path.join(args.results_dir, "high_res.json")
        with open(path, "w") as f:
            json.dump(r, f, indent=2)
        print(f"✓  Saved {path}")
        plot_high_res(r, args.results_dir)


if __name__ == "__main__":
    main()
