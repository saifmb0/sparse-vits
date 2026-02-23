"""
Benchmark 2: Sparsity vs. Real Speedup
========================================
X-axis: Pruning Ratio (0% to 90%)
Y-axis: Speedup relative to unpruned baseline (1.0x, 1.5x, etc.)
Lines:  Theoretical ideal | PyTorch Pruned | Triton Ragged
"""

import sys, os, gc, time, json
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PRUNE_RATIOS, WARMUP_ITERS, BENCH_ITERS, DEVICE, IMG_SIZE
from models.deit_base import load_deit, get_dtype


FIXED_BATCH = 8  # fixed batch size for this benchmark


def measure_latency(model_fn, batch_size, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """Returns average latency in ms per batch."""
    dtype = get_dtype()
    images = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE, device=DEVICE, dtype=dtype)

    for _ in range(warmup):
        model_fn(images)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        model_fn(images)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return (elapsed / iters) * 1000  # ms


def run_benchmark_2():
    results = {
        "prune_ratios": PRUNE_RATIOS,
        "theoretical": [],
        "pytorch_pruned": [],
        "triton_ragged": [],
        "dynamicvit_pytorch": [],
        "dynamicvit_triton": [],
    }

    # ── Baseline latency (unpruned) ──────────────────────────────────
    print("=== Measuring unpruned baseline latency ===")
    model_base = load_deit()
    base_latency = measure_latency(lambda img: model_base(img), FIXED_BATCH)
    print(f"  Unpruned latency: {base_latency:.2f} ms")
    del model_base; torch.cuda.empty_cache(); gc.collect()

    # ── Theoretical speedup ──────────────────────────────────────────
    # Late layers (8 out of 12) dominate compute; speedup ≈ 1/(1 - ratio * 8/12)
    # Simplified: attention FLOPs scale as (1-ratio)^2, MLP as (1-ratio)
    # We use a weighted model: 4 early layers at full + 8 late layers pruned
    for ratio in PRUNE_RATIOS:
        kept = 1.0 - ratio
        # Attention: O(S^2), MLP: O(S). Both only in late layers.
        late_frac = 8.0 / 12.0
        early_frac = 4.0 / 12.0
        # Relative compute = early(full) + late(kept^2 for attn + kept for MLP)
        # Approximate: attn ≈ 40% of layer, MLP ≈ 60% of layer
        late_compute = 0.4 * (kept ** 2) + 0.6 * kept
        total_compute = early_frac + late_frac * late_compute
        theoretical_speedup = 1.0 / total_compute if total_compute > 0 else 10.0
        results["theoretical"].append(round(theoretical_speedup, 3))
    print(f"  Theoretical speedups: {results['theoretical']}")

    # ── PyTorch Pruned ───────────────────────────────────────────────
    print("\n=== PyTorch Pruned (padded) ===")
    from baselines.pytorch_pruned import build_pytorch_pruned_model
    model_pt = build_pytorch_pruned_model()
    for ratio in PRUNE_RATIOS:
        try:
            lat = measure_latency(lambda img: model_pt(img, fixed_ratio=ratio), FIXED_BATCH)
            speedup = base_latency / lat
            print(f"  ratio={ratio:.1f}  lat={lat:.2f}ms  speedup={speedup:.2f}x")
        except torch.cuda.OutOfMemoryError:
            speedup = 0.0
            print(f"  ratio={ratio:.1f}  OOM")
        results["pytorch_pruned"].append(round(speedup, 3))
        torch.cuda.empty_cache(); gc.collect()
    del model_pt; torch.cuda.empty_cache(); gc.collect()

    # ── Triton Ragged ────────────────────────────────────────────────
    print("\n=== Triton Ragged (ours) ===")
    from models.triton_ragged_deit import build_triton_ragged_model
    model_tr = build_triton_ragged_model()
    for ratio in PRUNE_RATIOS:
        try:
            lat = measure_latency(lambda img: model_tr(img, fixed_ratio=ratio), FIXED_BATCH)
            speedup = base_latency / lat
            print(f"  ratio={ratio:.1f}  lat={lat:.2f}ms  speedup={speedup:.2f}x")
        except torch.cuda.OutOfMemoryError:
            speedup = 0.0
            print(f"  ratio={ratio:.1f}  OOM")
        results["triton_ragged"].append(round(speedup, 3))
        torch.cuda.empty_cache(); gc.collect()
    del model_tr; torch.cuda.empty_cache(); gc.collect()

    # ── DynamicViT + PyTorch Padded ──────────────────────────────────
    print("\n=== DynamicViT + PyTorch Padded ===")
    from baselines.dynamicvit_pytorch import build_dynamicvit_pytorch_model
    model_dp = build_dynamicvit_pytorch_model()
    for ratio in PRUNE_RATIOS:
        try:
            lat = measure_latency(lambda img: model_dp(img, fixed_ratio=ratio), FIXED_BATCH)
            speedup = base_latency / lat
            print(f"  ratio={ratio:.1f}  lat={lat:.2f}ms  speedup={speedup:.2f}x")
        except torch.cuda.OutOfMemoryError:
            speedup = 0.0
            print(f"  ratio={ratio:.1f}  OOM")
        results["dynamicvit_pytorch"].append(round(speedup, 3))
        torch.cuda.empty_cache(); gc.collect()
    del model_dp; torch.cuda.empty_cache(); gc.collect()

    # ── DynamicViT + Triton Ragged ───────────────────────────────────
    print("\n=== DynamicViT + Triton Ragged ===")
    from models.dynamicvit_ragged import build_dynamicvit_triton_model
    model_dt = build_dynamicvit_triton_model()
    for ratio in PRUNE_RATIOS:
        try:
            lat = measure_latency(lambda img: model_dt(img, fixed_ratio=ratio), FIXED_BATCH)
            speedup = base_latency / lat
            print(f"  ratio={ratio:.1f}  lat={lat:.2f}ms  speedup={speedup:.2f}x")
        except torch.cuda.OutOfMemoryError:
            speedup = 0.0
            print(f"  ratio={ratio:.1f}  OOM")
        results["dynamicvit_triton"].append(round(speedup, 3))
        torch.cuda.empty_cache(); gc.collect()
    del model_dt; torch.cuda.empty_cache(); gc.collect()

    # ── Save ─────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    with open("results/bench2_sparsity.json", "w") as f:
        json.dump(results, f, indent=2)

    plot_benchmark_2(results)
    return results


def plot_benchmark_2(results=None):
    if results is None:
        with open("results/bench2_sparsity.json") as f:
            results = json.load(f)

    ratios = [r * 100 for r in results["prune_ratios"]]  # as percent

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ratios, results["theoretical"],    "k--", label="Theoretical (ideal)", linewidth=2, alpha=0.7)
    ax.plot(ratios, results["pytorch_pruned"], "s-",  label="A-ViT PyTorch (padded)", linewidth=2)
    ax.plot(ratios, results["triton_ragged"],  "D-",  label="A-ViT Triton Ragged (ours)", linewidth=2, color="red")
    if "dynamicvit_pytorch" in results and results["dynamicvit_pytorch"]:
        ax.plot(ratios, results["dynamicvit_pytorch"], "^--", label="DynamicViT PyTorch (padded)", linewidth=2, color="purple")
    if "dynamicvit_triton" in results and results["dynamicvit_triton"]:
        ax.plot(ratios, results["dynamicvit_triton"],  "v-",  label="DynamicViT Triton Ragged (ours)", linewidth=2, color="orangered")

    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)

    ax.set_xlabel("Pruning Ratio (%)", fontsize=12)
    ax.set_ylabel("Speedup (×)", fontsize=12)
    ax.set_title("Benchmark 2: Sparsity vs. Real Speedup", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/bench2_sparsity.png", dpi=150)
    plt.savefig("results/bench2_sparsity.pdf")
    print("Saved: results/bench2_sparsity.png")


if __name__ == "__main__":
    run_benchmark_2()
