"""
Benchmark 4: Model-Size Scaling (DeiT-Tiny → Small → Base)
============================================================
Demonstrates that the Triton ragged-batch advantage **grows** with
model size.  For each model variant we measure throughput (img/s)
across batch sizes for three execution paths:

  1. Unpruned (standard forward)
  2. Threshold-L2 + PyTorch padded
  3. Threshold-L2 + Triton ragged

All three share the same pruning signal and prune ratio, so the only
difference is the execution backend.  HEAD_DIM=64 for all DeiT models,
so our Triton kernels need zero modification.

Output: results/bench4_model_scaling.json + .png
"""

import sys, os, gc, time, json
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BATCH_SIZES, WARMUP_ITERS, BENCH_ITERS, DEVICE, IMG_SIZE, PRUNE_AFTER_LAYER
from models.deit_base import load_deit, split_deit, get_dtype, DeiTFrontEnd, DeiTBackEnd
from benchmarks.bench5_accuracy import get_imagenet_val
from models.pruning import threshold_prune_mask, pytorch_gather_and_pad
from kernels.pack_tokens import triton_pack_tokens
from kernels.ragged_attention import triton_ragged_attention
from baselines.pytorch_pruned import PaddedBlock
from models.triton_ragged_deit import RaggedAttentionBlock
from model_configs import SCALING_MODELS, ModelConfig


FIXED_RATIO = 0.5   # prune 50% of tokens


# ─────────────────────────────────────────────────────────────────────
# Builder helpers — construct pipelines for *any* DeiT model
# ─────────────────────────────────────────────────────────────────────

def build_unpruned(cfg: ModelConfig):
    """Standard DeiT forward — no pruning, no padding."""
    model = load_deit(cfg.timm_name)
    return model


def build_padded(cfg: ModelConfig):
    """Threshold-L2 pruning + PyTorch padded attention."""
    deit = load_deit(cfg.timm_name)
    front, early, late_seq, back = split_deit(deit)
    late_blocks = nn.ModuleList([PaddedBlock(b) for b in late_seq])

    class PaddedPipeline(nn.Module):
        def __init__(self):
            super().__init__()
            self.front = front
            self.early = early
            self.late_blocks = late_blocks
            self.back = back

        @torch.inference_mode()
        def forward(self, images, fixed_ratio=None):
            x = self.front(images)
            x = self.early(x)

            mask = threshold_prune_mask(x, fixed_ratio=fixed_ratio)
            x_pad, attn_mask_bool, _ = pytorch_gather_and_pad(x, mask)

            B, S, D = x_pad.shape
            attn_mask = torch.zeros(B, 1, 1, S, device=x.device, dtype=x.dtype)
            attn_mask.masked_fill_(~attn_mask_bool.unsqueeze(1).unsqueeze(2), float("-inf"))

            for block in self.late_blocks:
                x_pad = block(x_pad, attn_mask=attn_mask)

            return self.back(x_pad)

    return PaddedPipeline().to(DEVICE, dtype=get_dtype()).eval()


def build_ragged(cfg: ModelConfig):
    """Threshold-L2 pruning + Triton ragged attention."""
    deit = load_deit(cfg.timm_name)
    front, early, late_seq, back = split_deit(deit)
    late_blocks_list = list(late_seq)

    class RaggedPipeline(nn.Module):
        def __init__(self):
            super().__init__()
            self.front = front
            self.early = early
            self.ragged_blocks = nn.ModuleList(
                [RaggedAttentionBlock(b) for b in late_blocks_list]
            )
            self.back_norm = back.norm
            self.back_head = back.head

        @torch.inference_mode()
        def forward(self, images, fixed_ratio=None):
            B = images.shape[0]
            x = self.front(images)
            x = self.early(x)

            mask = threshold_prune_mask(x, fixed_ratio=fixed_ratio)
            packed, cu_seqlens = triton_pack_tokens(x, mask)

            for block in self.ragged_blocks:
                packed = block(packed, cu_seqlens)

            cls_indices = cu_seqlens[:-1].long()
            cls_tokens = packed[cls_indices]

            cls_tokens = self.back_norm(cls_tokens)
            return self.back_head(cls_tokens)

    return RaggedPipeline().to(DEVICE, dtype=get_dtype()).eval()


# ─────────────────────────────────────────────────────────────────────
# Measurement
# ─────────────────────────────────────────────────────────────────────

def measure_throughput(model_fn, images, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    batch_size = images.shape[0]

    for _ in range(warmup):
        model_fn(images)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        model_fn(images)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return (batch_size * iters) / elapsed


def build_imagenet_batch(val_data, batch_size):
    """Build a fixed batch from streamed ImageNet tensors."""
    batch = torch.stack([val_data[i][0] for i in range(batch_size)])
    return batch.to(DEVICE)


# ─────────────────────────────────────────────────────────────────────
# Main benchmark loop
# ─────────────────────────────────────────────────────────────────────

def run_benchmark_4():
    results = {
        "batch_sizes": BATCH_SIZES,
        "models": {},
    }

    max_samples = max(BATCH_SIZES)
    val_data = get_imagenet_val(max_samples=max_samples)

    for cfg in SCALING_MODELS:
        print(f"\n{'='*60}")
        print(f"  MODEL: {cfg.short_name}  ({cfg.num_params_m:.1f}M params, "
              f"dim={cfg.embed_dim}, heads={cfg.num_heads})")
        print(f"{'='*60}")

        model_results = {"unpruned": [], "padded": [], "ragged": []}

        # ── 1. Unpruned ──────────────────────────────────────────
        print(f"\n--- {cfg.short_name} · Unpruned ---")
        model = build_unpruned(cfg)
        for bs in BATCH_SIZES:
            try:
                images = build_imagenet_batch(val_data, bs)
                tp = measure_throughput(lambda img: model(img), images)
                print(f"  BS={bs:3d}  → {tp:.1f} img/s")
            except torch.cuda.OutOfMemoryError:
                tp = 0.0
                print(f"  BS={bs:3d}  → OOM")
            model_results["unpruned"].append(round(tp, 1))
            torch.cuda.empty_cache(); gc.collect()
        del model; torch.cuda.empty_cache(); gc.collect()

        # ── 2. Padded ────────────────────────────────────────────
        print(f"\n--- {cfg.short_name} · Threshold-L2 + PyTorch Padded ---")
        model = build_padded(cfg)
        for bs in BATCH_SIZES:
            try:
                images = build_imagenet_batch(val_data, bs)
                tp = measure_throughput(lambda img: model(img, fixed_ratio=FIXED_RATIO), images)
                print(f"  BS={bs:3d}  → {tp:.1f} img/s")
            except torch.cuda.OutOfMemoryError:
                tp = 0.0
                print(f"  BS={bs:3d}  → OOM")
            model_results["padded"].append(round(tp, 1))
            torch.cuda.empty_cache(); gc.collect()
        del model; torch.cuda.empty_cache(); gc.collect()

        # ── 3. Triton Ragged ─────────────────────────────────────
        print(f"\n--- {cfg.short_name} · Threshold-L2 + Triton Ragged ---")
        model = build_ragged(cfg)
        for bs in BATCH_SIZES:
            try:
                images = build_imagenet_batch(val_data, bs)
                tp = measure_throughput(lambda img: model(img, fixed_ratio=FIXED_RATIO), images)
                print(f"  BS={bs:3d}  → {tp:.1f} img/s")
            except torch.cuda.OutOfMemoryError:
                tp = 0.0
                print(f"  BS={bs:3d}  → OOM")
            model_results["ragged"].append(round(tp, 1))
            torch.cuda.empty_cache(); gc.collect()
        del model; torch.cuda.empty_cache(); gc.collect()

        results["models"][cfg.short_name] = model_results

    # ── Save ─────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    with open("results/bench4_model_scaling.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✓ Results saved to results/bench4_model_scaling.json")

    # ── Plot ─────────────────────────────────────────────────────────
    plot_benchmark_4(results)


# ─────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────

def plot_benchmark_4(results=None):
    if results is None:
        with open("results/bench4_model_scaling.json") as f:
            results = json.load(f)

    bs = results["batch_sizes"]
    model_names = list(results["models"].keys())
    n_models = len(model_names)

    # ── Figure 1: Per-model throughput subplots ──────────────────────
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5),
                             sharey=False, squeeze=False)
    axes = axes[0]

    colors = {"unpruned": "#1f77b4", "padded": "#d62728", "ragged": "#2ca02c"}
    markers = {"unpruned": "o", "padded": "s", "ragged": "D"}
    labels = {"unpruned": "Unpruned", "padded": "Pruned · PyTorch (pad)",
              "ragged": "Pruned · Triton (ours)"}

    for i, name in enumerate(model_names):
        ax = axes[i]
        data = results["models"][name]
        for variant in ["unpruned", "padded", "ragged"]:
            vals = data[variant]
            if vals and any(v > 0 for v in vals):
                ax.plot(bs, vals, f"{markers[variant]}-",
                        label=labels[variant], color=colors[variant],
                        linewidth=2, markersize=7)
        ax.set_title(name, fontsize=14, fontweight="bold")
        ax.set_xlabel("Batch Size", fontsize=11)
        if i == 0:
            ax.set_ylabel("Throughput (img/s)", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Throughput Scaling by Model Size  (50% prune)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/bench4_model_scaling_throughput.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ Saved results/bench4_model_scaling_throughput.png")

    # ── Figure 2: Speedup summary bar chart (BS=32) ─────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 5))

    # Find BS=32 index (or last available)
    try:
        bs_idx = bs.index(32)
    except ValueError:
        bs_idx = len(bs) - 1

    x_pos = range(n_models)
    bar_width = 0.25

    unpruned_vals = []
    padded_speedups = []
    ragged_speedups = []

    for name in model_names:
        data = results["models"][name]
        base = data["unpruned"][bs_idx] if data["unpruned"][bs_idx] > 0 else 1.0
        unpruned_vals.append(base)
        padded_speedups.append(
            data["padded"][bs_idx] / base if data["padded"][bs_idx] > 0 else 0
        )
        ragged_speedups.append(
            data["ragged"][bs_idx] / base if data["ragged"][bs_idx] > 0 else 0
        )

    bars1 = ax2.bar([x - bar_width/2 for x in x_pos], padded_speedups,
                     bar_width, label="Pruned · PyTorch (pad)", color="#d62728", alpha=0.85)
    bars2 = ax2.bar([x + bar_width/2 for x in x_pos], ragged_speedups,
                     bar_width, label="Pruned · Triton (ours)", color="#2ca02c", alpha=0.85)

    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, label="Unpruned baseline")
    ax2.set_xticks(list(x_pos))
    ax2.set_xticklabels([f"{n}\n({v:.0f} img/s)" for n, v in zip(model_names, unpruned_vals)],
                         fontsize=10)
    ax2.set_ylabel("Speedup over Unpruned", fontsize=12)
    ax2.set_title(f"Ragged vs Padded Speedup at BS={bs[bs_idx]}  (50% prune)",
                  fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, axis="y", alpha=0.3)

    # Annotate bar values
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        if h > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                     f"{h:.2f}×", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig2.savefig("results/bench4_model_scaling_speedup.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("✓ Saved results/bench4_model_scaling_speedup.png")
