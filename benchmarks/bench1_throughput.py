"""
Benchmark 1: Batch-Size Scaling Curve (Throughput)
===================================================
X-axis: Batch Size (1, 4, 8, 16, 32)
Y-axis: Throughput (Images / sec)
Lines:  Standard DeiT | PyTorch Pruned | Triton Ragged
"""

import sys, os, gc, time, json
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BATCH_SIZES, WARMUP_ITERS, BENCH_ITERS, DEVICE, IMG_SIZE
from models.deit_base import load_deit, get_dtype
from benchmarks.bench5_accuracy import get_imagenet_val


def measure_throughput(model_fn, images, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """
    model_fn: callable(images) → logits
    Returns images/sec.
    """
    batch_size = images.shape[0]

    # warmup
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


def run_benchmark_1():
    results = {
        "batch_sizes": BATCH_SIZES,
        "unpruned": [], "pytorch_pruned": [], "triton_ragged": [],
        "dynamicvit_pytorch": [], "dynamicvit_triton": [],
        "evit_pytorch": [], "evit_triton": [],
        "ats_pytorch": [], "ats_triton": [],
        "tome_pytorch": [],
    }

    max_samples = max(BATCH_SIZES)
    val_data = get_imagenet_val(max_samples=max_samples)

    # ── 1. Standard DeiT ─────────────────────────────────────────────
    print("=== Standard DeiT (unpruned) ===")
    model = load_deit()
    for bs in BATCH_SIZES:
        images = build_imagenet_batch(val_data, bs)
        try:
            tp = measure_throughput(lambda img: model(img), images)
            print(f"  BS={bs:3d}  → {tp:.1f} img/s")
        except torch.cuda.OutOfMemoryError:
            tp = 0.0
            print(f"  BS={bs:3d}  → OOM")
        results["unpruned"].append(tp)
        torch.cuda.empty_cache(); gc.collect()
    del model; torch.cuda.empty_cache(); gc.collect()

    # ── 2. PyTorch Pruned ────────────────────────────────────────────
    print("\n=== PyTorch Pruned DeiT (padded) ===")
    from baselines.pytorch_pruned import build_pytorch_pruned_model
    model = build_pytorch_pruned_model()
    for bs in BATCH_SIZES:
        images = build_imagenet_batch(val_data, bs)
        try:
            tp = measure_throughput(lambda img: model(img, fixed_ratio=0.5), images)
            print(f"  BS={bs:3d}  → {tp:.1f} img/s")
        except torch.cuda.OutOfMemoryError:
            tp = 0.0
            print(f"  BS={bs:3d}  → OOM")
        results["pytorch_pruned"].append(tp)
        torch.cuda.empty_cache(); gc.collect()
    del model; torch.cuda.empty_cache(); gc.collect()

    # ── 3. Triton Ragged ─────────────────────────────────────────────
    print("\n=== Triton Ragged DeiT ===")
    from models.triton_ragged_deit import build_triton_ragged_model
    model = build_triton_ragged_model()
    for bs in BATCH_SIZES:
        images = build_imagenet_batch(val_data, bs)
        try:
            tp = measure_throughput(lambda img: model(img, fixed_ratio=0.5), images)
            print(f"  BS={bs:3d}  → {tp:.1f} img/s")
        except torch.cuda.OutOfMemoryError:
            tp = 0.0
            print(f"  BS={bs:3d}  → OOM")
        results["triton_ragged"].append(tp)
        torch.cuda.empty_cache(); gc.collect()
    del model; torch.cuda.empty_cache(); gc.collect()

    # ── 4. DynamicViT + PyTorch Padded ────────────────────────────────
    print("\n=== DynamicViT + PyTorch Padded ===")
    from baselines.dynamicvit_pytorch import build_dynamicvit_pytorch_model
    model = build_dynamicvit_pytorch_model()
    for bs in BATCH_SIZES:
        images = build_imagenet_batch(val_data, bs)
        try:
            tp = measure_throughput(lambda img: model(img, fixed_ratio=0.5), images)
            print(f"  BS={bs:3d}  → {tp:.1f} img/s")
        except torch.cuda.OutOfMemoryError:
            tp = 0.0
            print(f"  BS={bs:3d}  → OOM")
        results["dynamicvit_pytorch"].append(tp)
        torch.cuda.empty_cache(); gc.collect()
    del model; torch.cuda.empty_cache(); gc.collect()

    # ── 5. DynamicViT + Triton Ragged ────────────────────────────────
    print("\n=== DynamicViT + Triton Ragged ===")
    from models.dynamicvit_ragged import build_dynamicvit_triton_model
    model = build_dynamicvit_triton_model()
    for bs in BATCH_SIZES:
        images = build_imagenet_batch(val_data, bs)
        try:
            tp = measure_throughput(lambda img: model(img, fixed_ratio=0.5), images)
            print(f"  BS={bs:3d}  → {tp:.1f} img/s")
        except torch.cuda.OutOfMemoryError:
            tp = 0.0
            print(f"  BS={bs:3d}  → OOM")
        results["dynamicvit_triton"].append(tp)
        torch.cuda.empty_cache(); gc.collect()
    del model; torch.cuda.empty_cache(); gc.collect()

    # ── 6. EViT + PyTorch Padded ──────────────────────────────────────
    print("\n=== EViT + PyTorch Padded ===")
    from baselines.evit_pytorch import build_evit_pytorch_model
    model = build_evit_pytorch_model()
    for bs in BATCH_SIZES:
        images = build_imagenet_batch(val_data, bs)
        try:
            tp = measure_throughput(lambda img: model(img, fixed_ratio=0.5), images)
            print(f"  BS={bs:3d}  → {tp:.1f} img/s")
        except torch.cuda.OutOfMemoryError:
            tp = 0.0
            print(f"  BS={bs:3d}  → OOM")
        results["evit_pytorch"].append(tp)
        torch.cuda.empty_cache(); gc.collect()
    del model; torch.cuda.empty_cache(); gc.collect()

    # ── 7. EViT + Triton Ragged ──────────────────────────────────────
    print("\n=== EViT + Triton Ragged ===")
    from models.evit_ragged import build_evit_triton_model
    model = build_evit_triton_model()
    for bs in BATCH_SIZES:
        images = build_imagenet_batch(val_data, bs)
        try:
            tp = measure_throughput(lambda img: model(img, fixed_ratio=0.5), images)
            print(f"  BS={bs:3d}  → {tp:.1f} img/s")
        except torch.cuda.OutOfMemoryError:
            tp = 0.0
            print(f"  BS={bs:3d}  → OOM")
        results["evit_triton"].append(tp)
        torch.cuda.empty_cache(); gc.collect()
    del model; torch.cuda.empty_cache(); gc.collect()

    # ── 8. ATS + PyTorch Padded ──────────────────────────────────────
    print("\n=== ATS + PyTorch Padded ===")
    from baselines.ats_pytorch import build_ats_pytorch_model
    model = build_ats_pytorch_model()
    for bs in BATCH_SIZES:
        images = build_imagenet_batch(val_data, bs)
        try:
            tp = measure_throughput(lambda img: model(img, fixed_ratio=0.5), images)
            print(f"  BS={bs:3d}  → {tp:.1f} img/s")
        except torch.cuda.OutOfMemoryError:
            tp = 0.0
            print(f"  BS={bs:3d}  → OOM")
        results["ats_pytorch"].append(tp)
        torch.cuda.empty_cache(); gc.collect()
    del model; torch.cuda.empty_cache(); gc.collect()

    # ── 9. ATS + Triton Ragged ──────────────────────────────────────
    print("\n=== ATS + Triton Ragged ===")
    from models.ats_ragged import build_ats_triton_model
    model = build_ats_triton_model()
    for bs in BATCH_SIZES:
        images = build_imagenet_batch(val_data, bs)
        try:
            tp = measure_throughput(lambda img: model(img, fixed_ratio=0.5), images)
            print(f"  BS={bs:3d}  → {tp:.1f} img/s")
        except torch.cuda.OutOfMemoryError:
            tp = 0.0
            print(f"  BS={bs:3d}  → OOM")
        results["ats_triton"].append(tp)
        torch.cuda.empty_cache(); gc.collect()
    del model; torch.cuda.empty_cache(); gc.collect()

    # ── 10. ToMe (PyTorch only) ──────────────────────────────────────
    print("\n=== ToMe · PyTorch ===")
    from baselines.tome_pytorch import build_tome_model
    model = build_tome_model()
    for bs in BATCH_SIZES:
        images = build_imagenet_batch(val_data, bs)
        try:
            tp = measure_throughput(lambda img: model(img, fixed_ratio=0.5), images)
            print(f"  BS={bs:3d}  → {tp:.1f} img/s")
        except torch.cuda.OutOfMemoryError:
            tp = 0.0
            print(f"  BS={bs:3d}  → OOM")
        results["tome_pytorch"].append(tp)
        torch.cuda.empty_cache(); gc.collect()
    del model; torch.cuda.empty_cache(); gc.collect()

    # ── Save results ─────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    with open("results/bench1_throughput.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Plot ─────────────────────────────────────────────────────────
    plot_benchmark_1(results)
    return results


def plot_benchmark_1(results=None):
    if results is None:
        with open("results/bench1_throughput.json") as f:
            results = json.load(f)

    bs = results["batch_sizes"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(bs, results["unpruned"],        "o-",  label="Unpruned DeiT-S", linewidth=2, color="black")
    ax.plot(bs, results["pytorch_pruned"],  "s--", label="Threshold-L2 · PyTorch (pad)", linewidth=2, color="#1f77b4")
    ax.plot(bs, results["triton_ragged"],   "D-",  label="Threshold-L2 · Triton (ours)", linewidth=2, color="red")
    if "dynamicvit_pytorch" in results and results["dynamicvit_pytorch"]:
        ax.plot(bs, results["dynamicvit_pytorch"], "^--", label="DynamicViT · PyTorch (pad)", linewidth=2, color="purple")
    if "dynamicvit_triton" in results and results["dynamicvit_triton"]:
        ax.plot(bs, results["dynamicvit_triton"],  "v-",  label="DynamicViT · Triton (ours)", linewidth=2, color="orangered")
    if "evit_pytorch" in results and results["evit_pytorch"]:
        ax.plot(bs, results["evit_pytorch"],  "p--", label="EViT · PyTorch (pad)", linewidth=2, color="teal")
    if "evit_triton" in results and results["evit_triton"]:
        ax.plot(bs, results["evit_triton"],   "h-",  label="EViT · Triton (ours)", linewidth=2, color="darkgreen")
    if "ats_pytorch" in results and results["ats_pytorch"]:
        ax.plot(bs, results["ats_pytorch"],   "*--", label="ATS · PyTorch (pad)", linewidth=2, color="goldenrod", markersize=10)
    if "ats_triton" in results and results["ats_triton"]:
        ax.plot(bs, results["ats_triton"],    "X-",  label="ATS · Triton (ours)", linewidth=2, color="darkorange", markersize=9)
    if "tome_pytorch" in results and results["tome_pytorch"]:
        ax.plot(bs, results["tome_pytorch"],  "d--", label="ToMe · PyTorch (merge)", linewidth=2, color="brown")

    ax.set_xlabel("Batch Size", fontsize=12)
    ax.set_ylabel("Throughput (images / sec)", fontsize=12)
    ax.set_title("Benchmark 1: Batch-Size Scaling", fontsize=14)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(bs)

    plt.tight_layout()
    plt.savefig("results/bench1_throughput.png", dpi=150)
    plt.savefig("results/bench1_throughput.pdf")
    print("Saved: results/bench1_throughput.png")


if __name__ == "__main__":
    run_benchmark_1()
