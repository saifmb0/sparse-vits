#!/usr/bin/env python3
"""
Generate all paper figures from aggregated benchmark results.

Figures produced:
  figures/sparsity_sweep.png   - Speedup vs pruning ratio (single run)
  figures/e2e_benchmark.png    - E2E throughput vs batch size (100-run aggregated)
  figures/stage_breakdown.png  - Per-stage latency stacked bars (100-run aggregated)
  figures/high_res_speedup.png - Speedup at 224² vs 384² (single run)
  figures/model_scaling.png    - Throughput across DeiT scales (single run)
  figures/micro_benchmark.png  - Kernel latency comparison (100-run aggregated)
"""

import json
import glob
import os
import numpy as np
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

os.makedirs("figures", exist_ok=True)

# ── Consistent style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "axes.spines.top":  False,
    "axes.spines.right": False,
})

COLORS = {
    "PyTorch Padded (SDPA)":     "#d62728",
    "Triton Ragged (ours)":      "#2ca02c",
    "FlashAttention-2 (varlen)": "#7b2d8b",
}
MARKERS = {
    "PyTorch Padded (SDPA)":     "s",
    "Triton Ragged (ours)":      "D",
    "FlashAttention-2 (varlen)": "^",
}
LABELS = {
    "PyTorch Padded (SDPA)":     "Padded SDPA",
    "Triton Ragged (ours)":      "Triton Ragged (ours)",
    "FlashAttention-2 (varlen)": "FA2 varlen",
}


# ── Aggregation helper ────────────────────────────────────────────────────────
def aggregate(base_dir, json_name, data_key):
    """Aggregate multi-run JSON results into per-config statistics."""
    files = glob.glob(os.path.join(base_dir, "*", json_name))
    if not files:
        raise FileNotFoundError(f"No files matching {base_dir}/*/{{json_name}}")
    raw = defaultdict(lambda: defaultdict(list))
    config_labels = None
    for fpath in files:
        with open(fpath) as f:
            d = json.load(f)
        if config_labels is None:
            config_labels = d.get("batch_sizes") or d.get("configs")
        target = d[data_key]
        for name, vals in target.items():
            for i, v in enumerate(vals):
                if v > 0:
                    raw[name][i].append(v)
    agg = {}
    for name, configs in raw.items():
        agg[name] = []
        for i in sorted(configs):
            v = configs[i]
            agg[name].append({
                "median": float(np.median(v)),
                "p5":     float(np.percentile(v, 5)),
                "p95":    float(np.percentile(v, 95)),
                "n":      len(v),
            })
    return config_labels, agg


def medians(agg, name):
    return np.array([s["median"] for s in agg[name]])

def lo(agg, name):
    return np.array([s["median"] - s["p5"] for s in agg[name]])

def hi(agg, name):
    return np.array([s["p95"] - s["median"] for s in agg[name]])


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 – Sparsity sweep
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating sparsity_sweep.png …")
with open("results/Old/RTX/sparsity_sweep.json") as f:
    ss = json.load(f)

ratios  = [int(r * 100) for r in ss["ratios"]]
padded  = np.array(ss["padded"])
triton  = np.array(ss["triton"])
fa2     = np.array(ss["fa2"])
ref_pad = padded  # speedup relative to padded at each ratio

fig, ax = plt.subplots(figsize=(3.5, 2.6))

ax.plot(ratios, padded  / padded,  "s--", color=COLORS["PyTorch Padded (SDPA)"],
        label=LABELS["PyTorch Padded (SDPA)"],    linewidth=1.5, markersize=5, zorder=3)
ax.plot(ratios, triton  / padded,  "D-",  color=COLORS["Triton Ragged (ours)"],
        label=LABELS["Triton Ragged (ours)"],     linewidth=1.5, markersize=5, zorder=3)
ax.plot(ratios, fa2     / padded,  "^-",  color=COLORS["FlashAttention-2 (varlen)"],
        label=LABELS["FlashAttention-2 (varlen)"], linewidth=1.5, markersize=5, zorder=3)

ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
ax.set_xlabel("Pruning ratio (%)")
ax.set_ylabel("Speedup over padded SDPA")
ax.set_title("Speedup vs. Pruning Ratio\nBS=32, DeiT-B, RTX 4000 Ada", fontweight="bold")
ax.set_xticks(ratios)
ax.legend(loc="upper left")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f×"))

plt.tight_layout()
fig.savefig("figures/sparsity_sweep.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Print key numbers
for i, r in enumerate(ratios):
    print(f"  {r:3d}%  padded={padded[i]:.1f}  triton={triton[i]:.1f}  "
          f"fa2={fa2[i]:.1f}  speedup_triton={triton[i]/padded[i]:.2f}×")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 – E2E throughput (100-run aggregated)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nGenerating e2e_benchmark.png …")
bs_labels, tp_agg = aggregate("results/e2e_benchmark", "e2e_benchmark.json", "pipelines")
_,          lat_agg = aggregate("results/e2e_benchmark", "e2e_benchmark.json", "latency_ms")

order = ["PyTorch Padded (SDPA)", "Triton Ragged (ours)", "FlashAttention-2 (varlen)"]

fig, ax = plt.subplots(figsize=(3.5, 2.8))

for name in order:
    if name not in tp_agg:
        continue
    med = medians(tp_agg, name)
    err_lo = lo(tp_agg, name)
    err_hi = hi(tp_agg, name)
    ax.errorbar(bs_labels, med,
                yerr=[err_lo, err_hi],
                marker=MARKERS[name], color=COLORS[name],
                label=LABELS[name],
                linewidth=1.5, markersize=5,
                capsize=3, capthick=1.0, elinewidth=0.8,
                linestyle="-" if name != "PyTorch Padded (SDPA)" else "--",
                zorder=3)

ax.set_xscale("log", base=2)
ax.set_xticks(bs_labels)
ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
ax.set_xlabel("Batch size")
ax.set_ylabel("Throughput (img/s, ↑ better)")
ax.set_title("E2E Throughput — DeiT-B, 50% pruned\n"
             "RTX 4000 Ada, median +/- 5/95-pct of 100 runs", fontweight="bold")
ax.legend(loc="upper left")

plt.tight_layout()
fig.savefig("figures/e2e_benchmark.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("  E2E throughput medians (img/s):")
print(f"  {'BS':>5} | {'Padded':>8} | {'Triton':>8} | {'FA2':>8} | T/Pad | T/FA2")
print("  " + "-" * 55)
pad_med = medians(tp_agg, "PyTorch Padded (SDPA)")
tri_med = medians(tp_agg, "Triton Ragged (ours)")
fa2_med = medians(tp_agg, "FlashAttention-2 (varlen)")
for i, bs in enumerate(bs_labels):
    print(f"  {bs:>5} | {pad_med[i]:>8.1f} | {tri_med[i]:>8.1f} | {fa2_med[i]:>8.1f} | "
          f"{tri_med[i]/pad_med[i]:.2f}× | {tri_med[i]/fa2_med[i]:.2f}×")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 – Stage breakdown (100-run aggregated)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nGenerating stage_breakdown.png …")
sb_files = glob.glob("results/stage_breakdown/*/stage_breakdown.json")
if not sb_files:
    raise FileNotFoundError("No stage_breakdown JSON files found")

# Aggregate stage breakdown across all runs
sb_raw = {
    "padded": defaultdict(lambda: defaultdict(list)),
    "triton": defaultdict(lambda: defaultdict(list)),
}
sb_batch_sizes = None
for fpath in sb_files:
    with open(fpath) as f:
        d = json.load(f)
    if sb_batch_sizes is None:
        sb_batch_sizes = [str(b) for b in d["batch_sizes"]]
    for bs in sb_batch_sizes:
        for stage, v in d["padded"][bs].items():
            sb_raw["padded"][bs][stage].append(v)
        for stage, v in d["triton"][bs].items():
            sb_raw["triton"][bs][stage].append(v)

sb_agg = {"padded": {}, "triton": {}}
for pipeline in ["padded", "triton"]:
    for bs in sb_batch_sizes:
        sb_agg[pipeline][bs] = {
            stage: float(np.median(vals))
            for stage, vals in sb_raw[pipeline][bs].items()
        }

# Plot grouped stacked bars
fig, ax = plt.subplots(figsize=(5.5, 2.8))

n_bs = len(sb_batch_sizes)
x = np.arange(n_bs)
w = 0.35

pad_colors = {
    "front_early_ms": "#4878cf",
    "gather_ms":      "#e87d2a",
    "late_blocks_ms": "#d62728",
    "head_ms":        "#888888",
}
tri_colors = {
    "front_early_ms": "#4878cf",
    "pack_ms":        "#f7c100",
    "late_blocks_ms": "#2ca02c",
    "head_ms":        "#888888",
}

pad_stages = ["front_early_ms", "gather_ms", "late_blocks_ms", "head_ms"]
tri_stages = ["front_early_ms", "pack_ms",   "late_blocks_ms", "head_ms"]

pad_labels_nice = {
    "front_early_ms": "Front + early blocks",
    "gather_ms":      "Gather (padded)",
    "late_blocks_ms": "Late blocks (attention)",
    "head_ms":        "Classification head",
}
tri_labels_nice = {
    "front_early_ms": "Front + early blocks",
    "pack_ms":        "Pack (Triton)",
    "late_blocks_ms": "Late blocks (attention)",
    "head_ms":        "Classification head",
}

# Draw padded bars
bottoms = np.zeros(n_bs)
for stage in pad_stages:
    vals = np.array([sb_agg["padded"][bs][stage] for bs in sb_batch_sizes])
    ax.bar(x - w/2, vals, w, bottom=bottoms,
           color=pad_colors[stage],
           label=pad_labels_nice[stage] if stage != "front_early_ms" else None,
           edgecolor="white", linewidth=0.4)
    bottoms += vals

# Draw triton bars
bottoms = np.zeros(n_bs)
for stage in tri_stages:
    vals = np.array([sb_agg["triton"][bs][stage] for bs in sb_batch_sizes])
    ax.bar(x + w/2, vals, w, bottom=bottoms,
           color=tri_colors[stage],
           edgecolor="white", linewidth=0.4)
    bottoms += vals

ax.set_xticks(x)
ax.set_xticklabels([f"BS={bs}" for bs in sb_batch_sizes])
ax.set_ylabel("Latency (ms)")
ax.set_title("Per-Stage Latency — DeiT-B, 50% pruned\n"
             "RTX 4000 Ada, medians of 100 runs", fontweight="bold")

# Manual legend
legend_elements = [
    Patch(facecolor="#4878cf", label="Front + early blocks"),
    Patch(facecolor=pad_colors["late_blocks_ms"], label="Late blocks — Padded"),
    Patch(facecolor=tri_colors["late_blocks_ms"], label="Late blocks — Triton"),
    Patch(facecolor=pad_colors["gather_ms"],      label="Gather (padded)"),
    Patch(facecolor=tri_colors["pack_ms"],        label="Pack (Triton)"),
    Patch(facecolor="#888888",                    label="Class. head"),
]
ax.legend(handles=legend_elements, fontsize=7, loc="upper left",
          ncol=2, framealpha=0.9)

# Annotate pipeline labels
total_pad = {bs: sum(sb_agg["padded"][bs].values()) for bs in sb_batch_sizes}
total_tri = {bs: sum(sb_agg["triton"][bs].values()) for bs in sb_batch_sizes}
for i, bs in enumerate(sb_batch_sizes):
    ax.text(i - w/2, total_pad[bs] + 0.5, "Padded", ha="center",
            va="bottom", fontsize=6, color="#333333", rotation=0)
    ax.text(i + w/2, total_tri[bs] + 0.5, "Triton", ha="center",
            va="bottom", fontsize=6, color="#333333", rotation=0)

plt.tight_layout()
fig.savefig("figures/stage_breakdown.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("  Stage breakdown medians (ms):")
for bs in sb_batch_sizes:
    p = sb_agg["padded"][bs]
    t = sb_agg["triton"][bs]
    lb_speedup = p["late_blocks_ms"] / t["late_blocks_ms"]
    print(f"  BS={bs:3s}: padded_late={p['late_blocks_ms']:.1f} "
          f"triton_late={t['late_blocks_ms']:.1f} speedup={lb_speedup:.2f}×")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 – High-resolution speedup
# ═══════════════════════════════════════════════════════════════════════════════
print("\nGenerating high_res_speedup.png …")
with open("results/Old/RTX/high_res.json") as f:
    hr = json.load(f)

hr_bs = hr["batch_sizes"]
resolutions = list(hr["resolutions"].keys())  # ["224²", "384²"]

fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.6), sharey=False)

for ax, res in zip(axes, resolutions):
    rdata = hr["resolutions"][res]
    padded_tp = np.array(rdata["padded"])
    triton_tp = np.array(rdata["triton"])
    fa2_tp    = np.array(rdata["fa2"])

    ax.plot(hr_bs, triton_tp / padded_tp, "D-",  color=COLORS["Triton Ragged (ours)"],
            label="Triton / Padded", linewidth=1.5, markersize=5, zorder=3)
    ax.plot(hr_bs, fa2_tp    / padded_tp, "^--", color=COLORS["FlashAttention-2 (varlen)"],
            label="FA2 / Padded",   linewidth=1.5, markersize=5, zorder=3)
    ax.plot(hr_bs, triton_tp / fa2_tp,    "s:",  color="#1f77b4",
            label="Triton / FA2",   linewidth=1.5, markersize=4, zorder=3)

    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xscale("log", base=2)
    ax.set_xticks(hr_bs)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Speedup" if res == resolutions[0] else "")
    res_str = res.replace("²", "²")
    ax.set_title(f"{res_str} inputs", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f×"))
    if res == resolutions[0]:
        ax.legend(loc="upper left", fontsize=7)

fig.suptitle("Speedup vs. Padded SDPA — DeiT-B, 50% pruned, RTX 4000 Ada",
             fontweight="bold", fontsize=9, y=1.01)
plt.tight_layout()
fig.savefig("figures/high_res_speedup.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("  High-res speedup (Triton/Padded) at large batch sizes:")
for res in resolutions:
    rdata = hr["resolutions"][res]
    padded_tp = np.array(rdata["padded"])
    triton_tp = np.array(rdata["triton"])
    max_speedup = np.max(triton_tp / padded_tp)
    res_str = res.replace("²", "²")
    print(f"    {res_str}: max speedup = {max_speedup:.2f}×  "
          f"  (BS=1: {triton_tp[0]/padded_tp[0]:.2f}×)")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5 – Model scaling
# ═══════════════════════════════════════════════════════════════════════════════
print("\nGenerating model_scaling.png …")
with open("results/Old/RTX/model_scaling.json") as f:
    ms = json.load(f)

ms_bs    = ms["batch_sizes"]
model_order = ["DeiT-Ti", "DeiT-S", "DeiT-B"]
pipeline_order = ["padded", "triton", "fa2"]
pipe_labels = {"padded": "Padded SDPA", "triton": "Triton Ragged (ours)", "fa2": "FA2 varlen"}
pipe_colors = {
    "padded": COLORS["PyTorch Padded (SDPA)"],
    "triton": COLORS["Triton Ragged (ours)"],
    "fa2":    COLORS["FlashAttention-2 (varlen)"],
}
pipe_markers = {"padded": "s", "triton": "D", "fa2": "^"}
pipe_styles  = {"padded": "--", "triton": "-", "fa2": "-"}

model_linestyles = {"DeiT-Ti": "-", "DeiT-S": "--", "DeiT-B": ":"}
model_alphas     = {"DeiT-Ti": 1.0, "DeiT-S": 0.80, "DeiT-B": 0.60}

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=False)

for ax, model in zip(axes, model_order):
    mdata = ms["models"][model]
    ax.plot(ms_bs, mdata["padded"], "s--", color=COLORS["PyTorch Padded (SDPA)"],
            label=LABELS["PyTorch Padded (SDPA)"], linewidth=1.5, markersize=4)
    ax.plot(ms_bs, mdata["triton"], "D-",  color=COLORS["Triton Ragged (ours)"],
            label=LABELS["Triton Ragged (ours)"],  linewidth=1.5, markersize=4)
    ax.plot(ms_bs, mdata["fa2"],    "^-",  color=COLORS["FlashAttention-2 (varlen)"],
            label=LABELS["FlashAttention-2 (varlen)"], linewidth=1.5, markersize=4)

    ax.set_xscale("log", base=2)
    ax.set_xticks(ms_bs)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Batch size")
    if model == model_order[0]:
        ax.set_ylabel("Throughput (img/s)")
    ax.set_title(model, fontweight="bold")
    if model == model_order[0]:
        ax.legend(fontsize=6.5, loc="upper left")

    # Annotate peak throughput speedup (peak Triton / peak Padded)
    pad_arr = np.array(mdata["padded"])
    tri_arr = np.array(mdata["triton"])
    peak = np.max(tri_arr) / np.max(pad_arr)
    ax.text(0.97, 0.05, f"Peak: {peak:.2f}x\nvs padded",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.5, color=COLORS["Triton Ragged (ours)"],
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

fig.suptitle("Throughput Across DeiT Scales — 50% pruned, RTX 4000 Ada",
             fontweight="bold", fontsize=9, y=1.01)
plt.tight_layout()
fig.savefig("figures/model_scaling.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("  Model scaling peak throughput (img/s):")
print(f"  {'Model':<10} | {'Padded':>8} | {'Triton':>8} | {'FA2':>8} | T/Pad")
print("  " + "-" * 50)
for model in model_order:
    mdata = ms["models"][model]
    pad_peak = max(mdata["padded"])
    tri_peak = max(mdata["triton"])
    fa2_peak = max(mdata["fa2"])
    print(f"  {model:<10} | {pad_peak:>8.0f} | {tri_peak:>8.0f} | {fa2_peak:>8.0f} | "
          f"{tri_peak/pad_peak:.2f}×")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6 – Micro-benchmark (100-run aggregated)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nGenerating micro_benchmark.png …")
cfg_labels, kern_agg = aggregate(
    "results/micro_benchmark", "micro_benchmark.json", "kernels")

# Split into BS=32 and BS=64 groups (3 configs each)
bs32_idx = [0, 1, 2]
bs64_idx = [3, 4, 5]
bs32_labels = ["0%", "50%", "80%"]
bs64_labels = ["0%", "50%", "80%"]

# Define kernel display order and style
kernel_order = [
    "SDPA (math, padded)",
    "SDPA (efficient, padded)",
    "SDPA (flash, padded)",
    "NestedTensor + SDPA",
    "FlashAttention-2 (varlen)",
    "Triton Ragged (ours)",
]
k_colors = {
    "SDPA (math, padded)":       "#aec7e8",
    "SDPA (efficient, padded)":  "#ffbb78",
    "SDPA (flash, padded)":      "#98df8a",
    "NestedTensor + SDPA":       "#c5b0d5",
    "FlashAttention-2 (varlen)": COLORS["FlashAttention-2 (varlen)"],
    "Triton Ragged (ours)":      COLORS["Triton Ragged (ours)"],
}
k_markers = {
    "SDPA (math, padded)":       "o",
    "SDPA (efficient, padded)":  "v",
    "SDPA (flash, padded)":      "^",
    "NestedTensor + SDPA":       "p",
    "FlashAttention-2 (varlen)": "^",
    "Triton Ragged (ours)":      "D",
}
k_labels = {
    "SDPA (math, padded)":       "SDPA math",
    "SDPA (efficient, padded)":  "SDPA efficient",
    "SDPA (flash, padded)":      "SDPA flash",
    "NestedTensor + SDPA":       "NestedTensor+SDPA",
    "FlashAttention-2 (varlen)": "FA2 varlen",
    "Triton Ragged (ours)":      "Triton (ours)",
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=False)

for ax, idxs, bs_label, title in [
    (ax1, bs32_idx, bs32_labels, "BS = 32"),
    (ax2, bs64_idx, bs64_labels, "BS = 64"),
]:
    x = np.arange(len(idxs))
    for kname in kernel_order:
        if kname not in kern_agg:
            continue
        med = np.array([kern_agg[kname][i]["median"] for i in idxs])
        lo_err = np.array([kern_agg[kname][i]["median"] - kern_agg[kname][i]["p5"] for i in idxs])
        hi_err = np.array([kern_agg[kname][i]["p95"] - kern_agg[kname][i]["median"] for i in idxs])
        ax.errorbar(x, med,
                    yerr=[lo_err, hi_err],
                    marker=k_markers[kname],
                    color=k_colors[kname],
                    label=k_labels[kname],
                    linewidth=1.5, markersize=5,
                    capsize=3, capthick=0.8, elinewidth=0.7,
                    linestyle="-" if "Triton" in kname or "FA2" in kname else "--",
                    zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}\npruned" for p in bs_label])
    ax.set_title(title, fontweight="bold")
    if ax is ax1:
        ax.set_ylabel("Latency (ms, log scale, lower is better)")
    ax.yaxis.set_minor_locator(mticker.NullLocator())

ax1.legend(fontsize=7, loc="lower left", framealpha=0.9)
fig.suptitle("Isolated Attention Kernel Latency — DeiT-B, RTX 4000 Ada\n"
             "median +/- 5/95-pct of 100 runs", fontweight="bold", fontsize=9)
fig.tight_layout(rect=[0.0, 0.0, 1.0, 1.0])
fig.savefig("figures/micro_benchmark.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("  Micro-benchmark medians (ms):")
print(f"  {'Config':<22} | {'FA2':>7} | {'Triton':>7} | Triton/FA2")
print("  " + "-" * 50)
for i, cfg in enumerate(cfg_labels):
    fa2_v = kern_agg["FlashAttention-2 (varlen)"][i]["median"]
    tri_v = kern_agg["Triton Ragged (ours)"][i]["median"]
    ratio = fa2_v / tri_v
    print(f"  {cfg:<22} | {fa2_v:>7.4f} | {tri_v:>7.4f} | {ratio:.2f}×")


# ═══════════════════════════════════════════════════════════════════════════════
# Print table values for paper
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TABLE VALUES FOR paper.tex")
print("="*60)

print("\n── tab:dispatch (micro_benchmark BS=32) ──")
print(f"{'BS':<4} {'Prune':>6} {'Tok':>6}  {'SDPA flash':>10}  {'FA2':>7}  {'Triton':>7}")
tok = {0: 197, 50: 99, 80: 40}
sdpa_flash = kern_agg["SDPA (flash, padded)"]
fa2_k      = kern_agg["FlashAttention-2 (varlen)"]
triton_k   = kern_agg["Triton Ragged (ours)"]
bs32_cfg   = [(0,0),(1,50),(2,80)]
bs64_cfg   = [(3,0),(4,50),(5,80)]
for bs, cfgs, label in [(32, bs32_cfg, "32"), (64, bs64_cfg, "64")]:
    for idx, prune in cfgs:
        sf = sdpa_flash[idx]["median"]
        fa = fa2_k[idx]["median"]
        tr = triton_k[idx]["median"]
        print(f"{bs:<4} {prune:>5}% {tok[prune]:>6}  {sf:>10.3f}  {fa:>7.3f}  {tr:>7.3f}")

print("\n── tab:e2e (throughput medians) ──")
print(f"{'BS':>5} | {'Padded':>8} | {'FA2':>8} | {'Triton':>8} | T/FA2 | T/Pad")
print("-" * 55)
for i, bs in enumerate(bs_labels):
    pad = pad_med[i]
    tri = tri_med[i]
    fa2 = fa2_med[i]
    print(f"{bs:>5} | {pad:>8.0f} | {fa2:>8.0f} | {tri:>8.0f} | "
          f"{tri/fa2:.2f}× | {tri/pad:.2f}×")

print("\n── tab:scaling (model scaling peak) ──")
print(f"{'Model':<10} | {'Padded':>8} | {'FA2':>8} | {'Triton':>8} | T/Pad")
print("-" * 50)
for model in model_order:
    mdata = ms["models"][model]
    pad_peak = max(mdata["padded"])
    tri_peak = max(mdata["triton"])
    fa2_peak = max(mdata["fa2"])
    print(f"{model:<10} | {pad_peak:>8.0f} | {fa2_peak:>8.0f} | {tri_peak:>8.0f} | "
          f"{tri_peak/pad_peak:.2f}×")

print("\nAll figures saved to ./figures/")
