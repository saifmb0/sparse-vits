#!/usr/bin/env python3
"""Generate e2e_benchmark.png from merged_data.json (100-run aggregated medians)."""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("merged_data.json") as f:
    d = json.load(f)

labels    = d["e2e_throughput"]["config_labels"]   # batch sizes
tp        = d["e2e_throughput"]["results"]
lat       = d["e2e_latency"]["results"]

pipelines_tp  = {k: [v["median"] for v in vals] for k, vals in tp.items()}
pipelines_lat = {k: [v["median"] for v in vals] for k, vals in lat.items()}
pipelines_p5  = {k: [v["min"]    for v in vals] for k, vals in lat.items()}
pipelines_p95 = {k: [v["max"]    for v in vals] for k, vals in lat.items()}

styles = {
    "PyTorch Padded (SDPA)":      ("s--", "#d62728"),
    "Triton Ragged (ours)":       ("D-",  "#2ca02c"),
    "FlashAttention-2 (varlen)":  ("^-",  "#9467bd"),
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for name, vals in pipelines_tp.items():
    marker, color = styles.get(name, ("o-", "gray"))
    ax1.plot(labels, vals, marker, label=name, color=color, linewidth=2, markersize=8)

ax1.set_xlabel("Batch size", fontsize=12)
ax1.set_ylabel("Throughput (img/s, higher is better)", fontsize=11)
ax1.set_title(
    "E2E Throughput — DeiT-B, 50% pruned\n"
    "RTX 4000 Ada (SM89), median of 100 independent runs",
    fontsize=11, fontweight="bold",
)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

for name, vals in pipelines_lat.items():
    marker, color = styles.get(name, ("o-", "gray"))
    p5v  = pipelines_p5[name]
    p95v = pipelines_p95[name]
    ax2.plot(labels, vals, marker, label=name, color=color, linewidth=2, markersize=8)
    ax2.fill_between(labels, p5v, p95v, alpha=0.12, color=color)

ax2.set_xlabel("Batch size", fontsize=12)
ax2.set_ylabel("Latency (ms, lower is better)", fontsize=11)
ax2.set_title(
    "E2E Latency — DeiT-B, 50% pruned\n"
    "shaded band = min–max over 100 runs",
    fontsize=11, fontweight="bold",
)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs("figures", exist_ok=True)
out = "figures/e2e_benchmark.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")

# Speedup table
ref = "FlashAttention-2 (varlen)"
if ref in pipelines_tp:
    print(f"\nSpeedup over FA2 varlen (throughput, median of 100 runs):")
    ref_vals = pipelines_tp[ref]
    for name, vals in pipelines_tp.items():
        if name == ref:
            continue
        row = [f"{v/r:.3f}x" for r, v in zip(ref_vals, vals)]
        print(f"  {name:35s}  {' | '.join(row)}")
    print(f"  Batch sizes: {labels}")
