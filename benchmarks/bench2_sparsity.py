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
from benchmarks.bench5_accuracy import get_imagenet_val


FIXED_BATCH = 128  # fixed batch size for this benchmark


def measure_latency(model_fn, images, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """Returns average latency in ms per batch."""
    batch_size = images.shape[0]

    for _ in range(warmup):
        model_fn(images)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        model_fn(images)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return (elapsed / iters) * 1000  # ms


def build_imagenet_batch(val_data, batch_size):
    """Build a fixed batch from streamed ImageNet tensors."""
    batch = torch.stack([val_data[i][0] for i in range(batch_size)])
    return batch.to(DEVICE)


def run_benchmark_2():
    results = {
        "prune_ratios": PRUNE_RATIOS,
        "theoretical": [],
        "pytorch_pruned": [],
        "triton_ragged": [],
        "dynamicvit_pytorch": [],
        "dynamicvit_triton": [],
        "evit_pytorch": [],
        "evit_triton": [],
        "ats_pytorch": [],
        "ats_triton": [],
        "tome_pytorch": [],
    }

    val_data = get_imagenet_val(max_samples=FIXED_BATCH)
    images = build_imagenet_batch(val_data, FIXED_BATCH)

    # ── Baseline latency (unpruned) ──────────────────────────────────
    print("=== Measuring unpruned baseline latency ===")
    model_base = load_deit()
    base_latency = measure_latency(lambda img: model_base(img), images)
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
            lat = measure_latency(lambda img: model_pt(img, fixed_ratio=ratio), images)
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
            lat = measure_latency(lambda img: model_tr(img, fixed_ratio=ratio), images)
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
            lat = measure_latency(lambda img: model_dp(img, fixed_ratio=ratio), images)
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
            lat = measure_latency(lambda img: model_dt(img, fixed_ratio=ratio), images)
            speedup = base_latency / lat
            print(f"  ratio={ratio:.1f}  lat={lat:.2f}ms  speedup={speedup:.2f}x")
        except torch.cuda.OutOfMemoryError:
            speedup = 0.0
            print(f"  ratio={ratio:.1f}  OOM")
        results["dynamicvit_triton"].append(round(speedup, 3))
        torch.cuda.empty_cache(); gc.collect()
    del model_dt; torch.cuda.empty_cache(); gc.collect()

    # ── EViT + PyTorch Padded ─────────────────────────────────────
    print("\n=== EViT + PyTorch Padded ===")
    from baselines.evit_pytorch import build_evit_pytorch_model
    model_ep = build_evit_pytorch_model()
    for ratio in PRUNE_RATIOS:
        try:
            lat = measure_latency(lambda img: model_ep(img, fixed_ratio=ratio), images)
            speedup = base_latency / lat
            print(f"  ratio={ratio:.1f}  lat={lat:.2f}ms  speedup={speedup:.2f}x")
        except torch.cuda.OutOfMemoryError:
            speedup = 0.0
            print(f"  ratio={ratio:.1f}  OOM")
        results["evit_pytorch"].append(round(speedup, 3))
        torch.cuda.empty_cache(); gc.collect()
    del model_ep; torch.cuda.empty_cache(); gc.collect()

    # ── EViT + Triton Ragged ─────────────────────────────────────
    print("\n=== EViT + Triton Ragged ===")
    from models.evit_ragged import build_evit_triton_model
    model_et = build_evit_triton_model()
    for ratio in PRUNE_RATIOS:
        try:
            lat = measure_latency(lambda img: model_et(img, fixed_ratio=ratio), images)
            speedup = base_latency / lat
            print(f"  ratio={ratio:.1f}  lat={lat:.2f}ms  speedup={speedup:.2f}x")
        except torch.cuda.OutOfMemoryError:
            speedup = 0.0
            print(f"  ratio={ratio:.1f}  OOM")
        results["evit_triton"].append(round(speedup, 3))
        torch.cuda.empty_cache(); gc.collect()
    del model_et; torch.cuda.empty_cache(); gc.collect()
    # ── ATS + PyTorch Padded ─────────────────────────────────────
    print("\n=== ATS + PyTorch Padded ===")
    from baselines.ats_pytorch import build_ats_pytorch_model
    model_ap = build_ats_pytorch_model()
    for ratio in PRUNE_RATIOS:
        try:
            lat = measure_latency(lambda img: model_ap(img, fixed_ratio=ratio), images)
            speedup = base_latency / lat
            print(f"  ratio={ratio:.1f}  lat={lat:.2f}ms  speedup={speedup:.2f}x")
        except torch.cuda.OutOfMemoryError:
            speedup = 0.0
            print(f"  ratio={ratio:.1f}  OOM")
        results["ats_pytorch"].append(round(speedup, 3))
        torch.cuda.empty_cache(); gc.collect()
    del model_ap; torch.cuda.empty_cache(); gc.collect()

    # ── ATS + Triton Ragged ─────────────────────────────────────
    print("\n=== ATS + Triton Ragged ===")
    from models.ats_ragged import build_ats_triton_model
    model_at = build_ats_triton_model()
    for ratio in PRUNE_RATIOS:
        try:
            lat = measure_latency(lambda img: model_at(img, fixed_ratio=ratio), images)
            speedup = base_latency / lat
            print(f"  ratio={ratio:.1f}  lat={lat:.2f}ms  speedup={speedup:.2f}x")
        except torch.cuda.OutOfMemoryError:
            speedup = 0.0
            print(f"  ratio={ratio:.1f}  OOM")
        results["ats_triton"].append(round(speedup, 3))
        torch.cuda.empty_cache(); gc.collect()
    del model_at; torch.cuda.empty_cache(); gc.collect()

    # ── ToMe (PyTorch only) ──────────────────────────────────────────
    print("\n=== ToMe · PyTorch ===")
    from baselines.tome_pytorch import build_tome_model
    model_tm = build_tome_model()
    for ratio in PRUNE_RATIOS:
        try:
            lat = measure_latency(lambda img: model_tm(img, fixed_ratio=ratio), images)
            speedup = base_latency / lat
            print(f"  ratio={ratio:.1f}  lat={lat:.2f}ms  speedup={speedup:.2f}x")
        except torch.cuda.OutOfMemoryError:
            speedup = 0.0
            print(f"  ratio={ratio:.1f}  OOM")
        results["tome_pytorch"].append(round(speedup, 3))
        torch.cuda.empty_cache(); gc.collect()
    del model_tm; torch.cuda.empty_cache(); gc.collect()

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
    ax.plot(ratios, results["pytorch_pruned"], "s-",  label="Threshold-L2 · PyTorch (pad)", linewidth=2, color="#1f77b4")
    ax.plot(ratios, results["triton_ragged"],  "D-",  label="Threshold-L2 · Triton (ours)", linewidth=2, color="red")
    if "dynamicvit_pytorch" in results and results["dynamicvit_pytorch"]:
        ax.plot(ratios, results["dynamicvit_pytorch"], "^--", label="DynamicViT · PyTorch (pad)", linewidth=2, color="purple")
    if "dynamicvit_triton" in results and results["dynamicvit_triton"]:
        ax.plot(ratios, results["dynamicvit_triton"],  "v-",  label="DynamicViT · Triton (ours)", linewidth=2, color="orangered")
    if "evit_pytorch" in results and results["evit_pytorch"]:
        ax.plot(ratios, results["evit_pytorch"],  "p--", label="EViT · PyTorch (pad)", linewidth=2, color="teal")
    if "evit_triton" in results and results["evit_triton"]:
        ax.plot(ratios, results["evit_triton"],   "h-",  label="EViT · Triton (ours)", linewidth=2, color="darkgreen")
    if "ats_pytorch" in results and results["ats_pytorch"]:
        ax.plot(ratios, results["ats_pytorch"],   "*--", label="ATS · PyTorch (pad)", linewidth=2, color="goldenrod", markersize=10)
    if "ats_triton" in results and results["ats_triton"]:
        ax.plot(ratios, results["ats_triton"],    "X-",  label="ATS · Triton (ours)", linewidth=2, color="darkorange", markersize=9)
    if "tome_pytorch" in results and results["tome_pytorch"]:
        ax.plot(ratios, results["tome_pytorch"],  "d-",  label="ToMe · PyTorch (merge)", linewidth=2, color="brown")

    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)

    ax.set_xlabel("Pruning Ratio (%)", fontsize=12)
    ax.set_ylabel("Speedup (×)", fontsize=12)
    ax.set_title("Benchmark 2: Sparsity vs. Real Speedup", fontsize=14)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/bench2_sparsity.png", dpi=150)
    plt.savefig("results/bench2_sparsity.pdf")
    print("Saved: results/bench2_sparsity.png")


if __name__ == "__main__":
    run_benchmark_2()
