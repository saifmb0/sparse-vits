"""
Ragged Attention Benchmark Chart Generator
==========================================
Generates all paper figures from raw JSON benchmark data.

Usage:
    python plot_benchmarks.py --data_root ./  --output_dir ./figures --hardware GTX1650
    python plot_benchmarks.py --data_root ./  --output_dir ./figures --hardware A100
    python plot_benchmarks.py --data_root ./  --output_dir ./figures --hardware all

Directory structure expected:
    <data_root>/
        GTX1650/bench1_throughput.json
        GTX1650/bench2_sparsity.json
        ...
        T4/...
        A100/...
"""

import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ─────────────────────────────────────────────
# GLOBAL STYLE CONFIG — edit here to change aesthetics
# ─────────────────────────────────────────────
STYLE = {
    "font_family": "DejaVu Sans",
    "title_size": 13,
    "label_size": 11,
    "tick_size": 9,
    "legend_size": 8,
    "linewidth": 1.8,
    "markersize": 6,
    "dpi": 200,
    "fig_format": "png",  # or "pdf" for camera-ready
}

# Color + marker map per variant — consistent across all figures
VARIANT_STYLE = {
    "Unpruned":          {"color": "#222222", "marker": "o",  "ls": "-",  "label": "Unpruned DeiT-S"},
    "pytorch_pruned":    {"color": "#C0392B", "marker": "s",  "ls": "--", "label": "Pruned · PyTorch (pad)"},
    "triton_ragged":     {"color": "#27AE60", "marker": "D",  "ls": "-",  "label": "Triton Ragged (ours)"},
    "dynamicvit_pytorch":{"color": "#8E44AD", "marker": "^",  "ls": "--", "label": "DynamicViT · PyTorch"},
    "dynamicvit_triton": {"color": "#9B59B6", "marker": "^",  "ls": "-",  "label": "DynamicViT · Triton (ours)"},
    "evit_pytorch":      {"color": "#1A6E3C", "marker": "v",  "ls": "--", "label": "EViT · PyTorch (pad)"},
    "evit_triton":       {"color": "#2ECC71", "marker": "v",  "ls": "-",  "label": "EViT · Triton (ours)"},
    "ats_pytorch":       {"color": "#D4A017", "marker": "*",  "ls": "--", "label": "ATS · PyTorch (pad)"},
    "ats_triton":        {"color": "#F39C12", "marker": "*",  "ls": "-",  "label": "ATS · Triton (ours)"},
    "tome_pytorch":      {"color": "#7F8C8D", "marker": "P",  "ls": "--", "label": "ToMe · PyTorch (merge)"},
    "theoretical":       {"color": "#2C3E50", "marker": None, "ls": "--", "label": "Theoretical Ideal"},
}

plt.rcParams.update({
    "font.family": STYLE["font_family"],
    "axes.titlesize": STYLE["title_size"],
    "axes.labelsize": STYLE["label_size"],
    "xtick.labelsize": STYLE["tick_size"],
    "ytick.labelsize": STYLE["tick_size"],
    "legend.fontsize": STYLE["legend_size"],
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load(path):
    with open(path) as f:
        return json.load(f)

def savefig(fig, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.{STYLE['fig_format']}")
    fig.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")

def vstyle(key):
    s = VARIANT_STYLE.get(key, {"color": "gray", "marker": "o", "ls": "-", "label": key})
    return s

def filter_zeros(xs, ys):
    """Remove (x,y) pairs where y==0 (OOM markers)."""
    xs, ys = np.array(xs), np.array(ys)
    mask = ys > 0
    return xs[mask], ys[mask]


# ─────────────────────────────────────────────
# BENCH 1: Throughput vs Batch Size
# ─────────────────────────────────────────────

def plot_bench1(data_path, output_dir, hw_tag):
    d = load(data_path)
    bs = d["batch_sizes"]

    keys = ["unpruned", "pytorch_pruned", "triton_ragged",
            "dynamicvit_pytorch", "dynamicvit_triton",
            "evit_pytorch", "evit_triton",
            "ats_pytorch", "ats_triton", "tome_pytorch"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for key in keys:
        if key not in d:
            continue
        s = vstyle(key)
        x, y = filter_zeros(bs, d[key])
        ax.plot(x, y, color=s["color"], marker=s["marker"],
                linestyle=s["ls"], linewidth=STYLE["linewidth"],
                markersize=STYLE["markersize"], label=s["label"])

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Throughput (images / sec)")
    ax.set_title(f"Benchmark 1: Batch-Size Scaling  [{hw_tag}]")
    ax.legend(loc="upper left", ncol=2, framealpha=0.9)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    savefig(fig, output_dir, f"bench1_throughput_{hw_tag}")


# ─────────────────────────────────────────────
# BENCH 2: Sparsity vs Speedup
# ─────────────────────────────────────────────

def plot_bench2(data_path, output_dir, hw_tag):
    d = load(data_path)
    ratios = [r * 100 for r in d["prune_ratios"]]  # to percent

    keys = ["theoretical", "pytorch_pruned", "triton_ragged",
            "dynamicvit_pytorch", "dynamicvit_triton",
            "evit_pytorch", "evit_triton",
            "ats_pytorch", "ats_triton", "tome_pytorch"]

    fig, ax = plt.subplots(figsize=(7, 5))

    for key in keys:
        if key not in d:
            continue
        s = vstyle(key)
        lw = 2.2 if key == "theoretical" else STYLE["linewidth"]
        mk = None if key == "theoretical" else s["marker"]
        ax.plot(ratios, d[key], color=s["color"], marker=mk,
                linestyle=s["ls"], linewidth=lw,
                markersize=STYLE["markersize"], label=s["label"])

    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Pruning Ratio (%)")
    ax.set_ylabel("Speedup (×)")
    ax.set_title(f"Benchmark 2: Sparsity vs. Real Speedup  [{hw_tag}]")
    ax.legend(loc="upper left", ncol=2, framealpha=0.9)

    savefig(fig, output_dir, f"bench2_sparsity_{hw_tag}")


# ─────────────────────────────────────────────
# BENCH 3: VRAM Footprint
# ─────────────────────────────────────────────

def plot_bench3(data_path, output_dir, hw_tag):
    d = load(data_path)
    bs = d["batch_sizes"]

    keys = ["unpruned", "pytorch_pruned", "triton_ragged",
            "dynamicvit_pytorch", "dynamicvit_triton",
            "evit_pytorch", "evit_triton",
            "ats_pytorch", "ats_triton", "tome_pytorch"]

    fig, ax = plt.subplots(figsize=(7, 5))

    for key in keys:
        if key not in d:
            continue
        vals = d[key]
        # -1 = OOM, treat as missing
        x = [b for b, v in zip(bs, vals) if v > 0]
        y = [v for v in vals if v > 0]
        if not x:
            continue
        s = vstyle(key)
        ax.plot(x, y, color=s["color"], marker=s["marker"],
                linestyle=s["ls"], linewidth=STYLE["linewidth"],
                markersize=STYLE["markersize"], label=s["label"])

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Peak VRAM (MB)")
    ax.set_title(f"Benchmark 3: Peak VRAM Allocation  [{hw_tag}]")
    ax.legend(loc="upper left", ncol=2, framealpha=0.9)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    savefig(fig, output_dir, f"bench3_vram_{hw_tag}")


# ─────────────────────────────────────────────
# BENCH 4: Model Scaling (Throughput + Speedup bar)
# ─────────────────────────────────────────────

def plot_bench4_throughput(data_path, output_dir, hw_tag):
    d = load(data_path)
    bs = d["batch_sizes"]
    models = list(d["models"].keys())

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.5), sharey=False)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        m = d["models"][model]
        for key in ["unpruned", "padded", "ragged"]:
            if key not in m:
                continue
            label_map = {"unpruned": "Unpruned", "padded": "Pruned · PyTorch (pad)", "ragged": "Triton Ragged (ours)"}
            color_map = {"unpruned": "#2980B9", "padded": "#C0392B", "ragged": "#27AE60"}
            marker_map = {"unpruned": "o", "padded": "s", "ragged": "D"}
            x, y = filter_zeros(bs, m[key])
            ax.plot(x, y, color=color_map[key], marker=marker_map[key],
                    linewidth=STYLE["linewidth"], markersize=STYLE["markersize"],
                    label=label_map[key])
        ax.set_title(model)
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Throughput (img/s)")
        ax.legend(fontsize=7)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    fig.suptitle(f"Throughput Scaling by Model Size  [{hw_tag}]  (50% prune)", fontsize=STYLE["title_size"])
    fig.tight_layout()
    savefig(fig, output_dir, f"bench4_model_scaling_throughput_{hw_tag}")


def plot_bench4_speedup_bar(data_path, output_dir, hw_tag):
    """Bar chart: ragged vs padded speedup over unpruned at a fixed BS."""
    d = load(data_path)
    bs_list = d["batch_sizes"]
    models = list(d["models"].keys())

    # Pick BS=32 if available, else midpoint
    target_bs = 32
    if target_bs not in bs_list:
        target_bs = bs_list[len(bs_list) // 2]
    idx = bs_list.index(target_bs)

    padded_speedups, ragged_speedups, unpruned_vals = [], [], []
    for model in models:
        m = d["models"][model]
        base = m["unpruned"][idx] if m["unpruned"][idx] > 0 else 1.0
        padded_speedups.append(m["padded"][idx] / base if m["padded"][idx] > 0 else 0)
        ragged_speedups.append(m["ragged"][idx] / base if m["ragged"][idx] > 0 else 0)
        unpruned_vals.append(base)

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars_pad = ax.bar(x - width/2, padded_speedups, width, color="#C0392B", label="Pruned · PyTorch (pad)", alpha=0.9)
    bars_rag = ax.bar(x + width/2, ragged_speedups, width, color="#27AE60", label="Pruned · Triton (ours)", alpha=0.9)

    ax.axhline(1.0, color="gray", linewidth=1.0, linestyle="--", label="Unpruned baseline")
    ax.bar_label(bars_pad, fmt="%.2f×", padding=2, fontsize=8)
    ax.bar_label(bars_rag, fmt="%.2f×", padding=2, fontsize=8)

    model_labels = [f"{m}\n({v:.0f} img/s)" for m, v in zip(models, unpruned_vals)]
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.set_ylabel("Speedup over Unpruned")
    ax.set_title(f"Ragged vs Padded Speedup at BS={target_bs}  (50% prune)  [{hw_tag}]")
    ax.legend()

    savefig(fig, output_dir, f"bench4_model_scaling_speedup_{hw_tag}")


# ─────────────────────────────────────────────
# BENCH 5: Numerical Equivalence + Pareto
# ─────────────────────────────────────────────

def plot_bench5_numerical(data_path, output_dir, hw_tag):
    d = load(data_path)["numerical_equivalence"]
    methods = list(d.keys())
    max_errs = [d[m]["max_abs_diff"] for m in methods]
    mean_errs = [d[m]["mean_abs_diff"] for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width/2, max_errs,  width, color="#E74C3C", label="Max Abs Error",  alpha=0.9)
    ax.bar(x + width/2, mean_errs, width, color="#2ECC71", label="Mean Abs Error", alpha=0.9)
    ax.axhline(0.1, color="#F39C12", linewidth=1.5, linestyle="--", label="fp16 tolerance")

    for i, (mx, mn) in enumerate(zip(max_errs, mean_errs)):
        ax.text(i - width/2, mx + 0.0003, f"✓", ha="center", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Max Absolute Logit Difference")
    ax.set_title(f"Triton Ragged vs PyTorch Padded — Numerical Equivalence  [{hw_tag}]")
    ax.legend()

    savefig(fig, output_dir, f"bench5_numerical_equiv_{hw_tag}")


def plot_bench5_pareto(data_path, output_dir, hw_tag):
    """
    Pareto frontier: Top-1 Accuracy vs Throughput.
    Each variant becomes one curve across sparsity ratios.
    """
    d = load(data_path)["pareto"]
    ratios = d["ratios"]
    variants = d["variants"]

    # Color map for pareto variants
    pareto_colors = {
        "Unpruned DeiT-S":        ("#2980B9",  "o",  "-"),
        "Threshold-L2 · PyTorch": ("#C0392B",  "s",  "--"),
        "Threshold-L2 · Triton":  ("#E74C3C",  "s",  "-"),
        "EViT · PyTorch":         ("#1A6E3C",  "v",  "--"),
        "EViT · Triton":          ("#2ECC71",  "v",  "-"),
        "ATS · PyTorch":          ("#D4A017",  "*",  "--"),
        "ATS · Triton":           ("#F39C12",  "*",  "-"),
        "ToMe · PyTorch":         ("#7F8C8D",  "P",  "--"),
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    for vname, vdata in variants.items():
        tp = vdata["throughput"]
        acc = vdata["top1"]
        color, marker, ls = pareto_colors.get(vname, ("#999", "o", "-"))

        # annotate sparsity at each point
        ax.plot(tp, acc, color=color, marker=marker, linestyle=ls,
                linewidth=STYLE["linewidth"], markersize=STYLE["markersize"],
                label=vname)

        # label sparsity % on points (skip 0% clutter if crowded)
        for x_pt, y_pt, r in zip(tp, acc, ratios):
            if r in [0.0, 0.5, 0.75, 0.9]:
                ax.annotate(f"{int(r*100)}%", (x_pt, y_pt),
                            textcoords="offset points", xytext=(4, 2),
                            fontsize=6.5, color=color)

    ax.set_xlabel(f"Throughput (img/s, BS=32)")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title(f"Accuracy vs. Throughput Pareto Frontier  (ImageNet-1K val)  [{hw_tag}]")
    ax.legend(loc="lower left", framealpha=0.9, ncol=1)

    savefig(fig, output_dir, f"bench5_pareto_{hw_tag}")


# ─────────────────────────────────────────────
# BENCH 6: Kernel Microbenchmark + Pipeline
# ─────────────────────────────────────────────

def plot_bench6_microbench(data_path, output_dir, hw_tag):
    d = load(data_path)["kernel_microbenchmark"]
    configs = d["configs"]
    kernels = d["kernels"]

    x = np.arange(len(configs))
    n = len(kernels)
    width = 0.8 / n

    kernel_colors = {
        "SDPA (math, padded)":       "#5D6D7E",
        "SDPA (efficient, padded)":  "#85929E",
        "SDPA (flash, padded)":      "#AEB6BF",
        "PyTorch SDPA (Primary)":    "#2E86C1",
        "FlashAttention-2 (varlen)": "#E67E22",
        "NestedTensor + SDPA":       "#E74C3C",
        "NestedTensor (Secondary)":  "#E74C3C",
        "NestedTensor SDPA":         "#E74C3C",
        "Triton Ragged (ours)":      "#27AE60",
    }

    fig, ax = plt.subplots(figsize=(13, 5))

    for i, (kname, kvals) in enumerate(kernels.items()):
        offset = (i - n/2 + 0.5) * width
        color = kernel_colors.get(kname, "#999999")
        bars = ax.bar(x + offset, kvals, width * 0.92, label=kname, color=color, alpha=0.88)
        ax.bar_label(bars, fmt="%.2f", padding=1, fontsize=6.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Attention Kernel Micro-benchmark (lower is better)  [{hw_tag}]")
    ax.legend(loc="upper left", framealpha=0.9)

    savefig(fig, output_dir, f"bench6_kernel_microbench_{hw_tag}")


def plot_bench6_pipeline(data_path, output_dir, hw_tag):
    d = load(data_path)["pipeline_comparison"]
    bs = d["batch_sizes"]
    pipelines = d["pipelines"]

    pipeline_colors = {
        "PyTorch Padded (SDPA)":     ("#C0392B", "s", "--"),
        "Triton Ragged (ours)":      ("#27AE60", "D", "-"),
        "FlashAttention-2 (varlen)": ("#E67E22", "^", "-"),
        "NestedTensor SDPA":         ("#E74C3C", "v", ":"),
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    for pname, pvals in pipelines.items():
        color, marker, ls = pipeline_colors.get(pname, ("#999", "o", "-"))
        x, y = filter_zeros(bs, pvals)
        ax.plot(x, y, color=color, marker=marker, linestyle=ls,
                linewidth=STYLE["linewidth"], markersize=STYLE["markersize"],
                label=pname)

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Throughput (img/s)")
    ax.set_title(f"End-to-End Pipeline: Padded vs Triton Ragged  [{hw_tag}]  (Threshold-L2, 50% prune)")
    ax.legend()
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    savefig(fig, output_dir, f"bench6_pipeline_comparison_{hw_tag}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

HARDWARE_TAGS = ["GTX1650", "T4", "A100"]

BENCH_FUNCS = {
    "bench1_throughput.json":   [plot_bench1],
    "bench2_sparsity.json":     [plot_bench2],
    "bench3_vram.json":         [plot_bench3],
    "bench4_model_scaling.json":[plot_bench4_throughput, plot_bench4_speedup_bar],
    "bench5_accuracy.json":     [plot_bench5_numerical, plot_bench5_pareto],
    "bench6_sota_baselines.json":[plot_bench6_microbench, plot_bench6_pipeline],
}

def run(data_root, output_dir, hardware):
    tags = HARDWARE_TAGS if hardware == "all" else [hardware]
    for hw in tags:
        hw_dir = os.path.join(data_root, hw)
        if not os.path.isdir(hw_dir):
            print(f"  [skip] {hw_dir} not found")
            continue
        print(f"\n── {hw} ──")
        out = os.path.join(output_dir, hw)
        for fname, funcs in BENCH_FUNCS.items():
            fpath = os.path.join(hw_dir, fname)
            if not os.path.isfile(fpath):
                print(f"  [skip] {fname}")
                continue
            for fn in funcs:
                try:
                    fn(fpath, out, hw)
                except Exception as e:
                    print(f"  [error] {fn.__name__} on {fname}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  default="./results",           help="Root dir containing GTX1650/, T4/, A100/")
    parser.add_argument("--output_dir", default="./figures",   help="Output directory for figures")
    parser.add_argument("--hardware",   default="all",         choices=["GTX1650", "T4", "A100", "all"])
    parser.add_argument("--format",     default="png",         choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    STYLE["fig_format"] = args.format
    run(args.data_root, args.output_dir, args.hardware)
    print("\nDone.")