"""
Benchmark 3: Peak VRAM Allocation
===================================
Measures torch.cuda.max_memory_allocated() during forward pass
across batch sizes for all three methods.
"""

import sys, os, gc, json
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BATCH_SIZES, DEVICE, IMG_SIZE
from models.deit_base import load_deit, get_dtype


def measure_vram(model_fn, batch_size):
    """Returns peak VRAM in MB."""
    dtype = get_dtype()
    images = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE, device=DEVICE, dtype=dtype)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # warmup once
    model_fn(images)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    model_fn(images)
    torch.cuda.synchronize()

    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    return peak


def run_benchmark_3():
    results = {
        "batch_sizes": BATCH_SIZES,
        "unpruned": [], "pytorch_pruned": [], "triton_ragged": [],
        "dynamicvit_pytorch": [], "dynamicvit_triton": [],
        "evit_pytorch": [], "evit_triton": [],
        "ats_pytorch": [], "ats_triton": [],
    }

    # ── 1. Standard DeiT ─────────────────────────────────────────────
    print("=== Standard DeiT VRAM ===")
    model = load_deit()
    for bs in BATCH_SIZES:
        try:
            vram = measure_vram(lambda img: model(img), bs)
            print(f"  BS={bs:3d}  → {vram:.1f} MB")
        except torch.cuda.OutOfMemoryError:
            vram = -1
            print(f"  BS={bs:3d}  → OOM")
        results["unpruned"].append(round(vram, 1))
        torch.cuda.empty_cache(); gc.collect()
    del model; torch.cuda.empty_cache(); gc.collect()

    # ── 2. PyTorch Pruned ────────────────────────────────────────────
    print("\n=== PyTorch Pruned VRAM ===")
    from baselines.pytorch_pruned import build_pytorch_pruned_model
    model = build_pytorch_pruned_model()
    for bs in BATCH_SIZES:
        try:
            vram = measure_vram(lambda img: model(img, fixed_ratio=0.5), bs)
            print(f"  BS={bs:3d}  → {vram:.1f} MB")
        except torch.cuda.OutOfMemoryError:
            vram = -1
            print(f"  BS={bs:3d}  → OOM")
        results["pytorch_pruned"].append(round(vram, 1))
        torch.cuda.empty_cache(); gc.collect()
    del model; torch.cuda.empty_cache(); gc.collect()

    # ── 3. Triton Ragged ─────────────────────────────────────────────
    print("\n=== Triton Ragged VRAM ===")
    from models.triton_ragged_deit import build_triton_ragged_model
    model = build_triton_ragged_model()
    for bs in BATCH_SIZES:
        try:
            vram = measure_vram(lambda img: model(img, fixed_ratio=0.5), bs)
            print(f"  BS={bs:3d}  → {vram:.1f} MB")
        except torch.cuda.OutOfMemoryError:
            vram = -1
            print(f"  BS={bs:3d}  → OOM")
        results["triton_ragged"].append(round(vram, 1))
        torch.cuda.empty_cache(); gc.collect()
    del model; torch.cuda.empty_cache(); gc.collect()

    # ── 4. DynamicViT + PyTorch Padded ────────────────────────────────
    print("\n=== DynamicViT + PyTorch Padded VRAM ===")
    from baselines.dynamicvit_pytorch import build_dynamicvit_pytorch_model
    model = build_dynamicvit_pytorch_model()
    for bs in BATCH_SIZES:
        try:
            vram = measure_vram(lambda img: model(img, fixed_ratio=0.5), bs)
            print(f"  BS={bs:3d}  → {vram:.1f} MB")
        except torch.cuda.OutOfMemoryError:
            vram = -1
            print(f"  BS={bs:3d}  → OOM")
        results["dynamicvit_pytorch"].append(round(vram, 1))
        torch.cuda.empty_cache(); gc.collect()
    del model; torch.cuda.empty_cache(); gc.collect()

    # ── 5. DynamicViT + Triton Ragged ────────────────────────────────
    print("\n=== DynamicViT + Triton Ragged VRAM ===")
    from models.dynamicvit_ragged import build_dynamicvit_triton_model
    model = build_dynamicvit_triton_model()
    for bs in BATCH_SIZES:
        try:
            vram = measure_vram(lambda img: model(img, fixed_ratio=0.5), bs)
            print(f"  BS={bs:3d}  → {vram:.1f} MB")
        except torch.cuda.OutOfMemoryError:
            vram = -1
            print(f"  BS={bs:3d}  → OOM")
        results["dynamicvit_triton"].append(round(vram, 1))
        torch.cuda.empty_cache(); gc.collect()
    del model; torch.cuda.empty_cache(); gc.collect()

    # ── 6. EViT + PyTorch Padded ──────────────────────────────────────
    print("\n=== EViT + PyTorch Padded VRAM ===")
    from baselines.evit_pytorch import build_evit_pytorch_model
    model = build_evit_pytorch_model()
    for bs in BATCH_SIZES:
        try:
            vram = measure_vram(lambda img: model(img, fixed_ratio=0.5), bs)
            print(f"  BS={bs:3d}  → {vram:.1f} MB")
        except torch.cuda.OutOfMemoryError:
            vram = -1
            print(f"  BS={bs:3d}  → OOM")
        results["evit_pytorch"].append(round(vram, 1))
        torch.cuda.empty_cache(); gc.collect()
    del model; torch.cuda.empty_cache(); gc.collect()

    # ── 7. EViT + Triton Ragged ──────────────────────────────────────
    print("\n=== EViT + Triton Ragged VRAM ===")
    from models.evit_ragged import build_evit_triton_model
    model = build_evit_triton_model()
    for bs in BATCH_SIZES:
        try:
            vram = measure_vram(lambda img: model(img, fixed_ratio=0.5), bs)
            print(f"  BS={bs:3d}  → {vram:.1f} MB")
        except torch.cuda.OutOfMemoryError:
            vram = -1
            print(f"  BS={bs:3d}  → OOM")
        results["evit_triton"].append(round(vram, 1))
        torch.cuda.empty_cache(); gc.collect()
    del model; torch.cuda.empty_cache(); gc.collect()
    # ── 8. ATS + PyTorch Padded ──────────────────────────────────────
    print("\n=== ATS + PyTorch Padded VRAM ===")
    from baselines.ats_pytorch import build_ats_pytorch_model
    model = build_ats_pytorch_model()
    for bs in BATCH_SIZES:
        try:
            vram = measure_vram(lambda img: model(img, fixed_ratio=0.5), bs)
            print(f"  BS={bs:3d}  → {vram:.1f} MB")
        except torch.cuda.OutOfMemoryError:
            vram = -1
            print(f"  BS={bs:3d}  → OOM")
        results["ats_pytorch"].append(round(vram, 1))
        torch.cuda.empty_cache(); gc.collect()
    del model; torch.cuda.empty_cache(); gc.collect()

    # ── 9. ATS + Triton Ragged ──────────────────────────────────────
    print("\n=== ATS + Triton Ragged VRAM ===")
    from models.ats_ragged import build_ats_triton_model
    model = build_ats_triton_model()
    for bs in BATCH_SIZES:
        try:
            vram = measure_vram(lambda img: model(img, fixed_ratio=0.5), bs)
            print(f"  BS={bs:3d}  → {vram:.1f} MB")
        except torch.cuda.OutOfMemoryError:
            vram = -1
            print(f"  BS={bs:3d}  → OOM")
        results["ats_triton"].append(round(vram, 1))
        torch.cuda.empty_cache(); gc.collect()
    del model; torch.cuda.empty_cache(); gc.collect()
    # ── Save ─────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    with open("results/bench3_vram.json", "w") as f:
        json.dump(results, f, indent=2)

    plot_benchmark_3(results)
    return results


def plot_benchmark_3(results=None):
    if results is None:
        with open("results/bench3_vram.json") as f:
            results = json.load(f)

    bs = results["batch_sizes"]
    # Filter out OOM entries (-1)
    fig, ax = plt.subplots(figsize=(10, 6))

    for key, label, marker, color in [
        ("unpruned", "Unpruned DeiT-S", "o", "black"),
        ("pytorch_pruned", "Threshold-L2 · PyTorch (pad)", "s", "#1f77b4"),
        ("triton_ragged", "Threshold-L2 · Triton (ours)", "D", "red"),
        ("dynamicvit_pytorch", "DynamicViT · PyTorch (pad)", "^", "purple"),
        ("dynamicvit_triton", "DynamicViT · Triton (ours)", "v", "orangered"),
        ("evit_pytorch", "EViT · PyTorch (pad)", "p", "teal"),
        ("evit_triton", "EViT · Triton (ours)", "h", "darkgreen"),
        ("ats_pytorch", "ATS · PyTorch (pad)", "*", "goldenrod"),
        ("ats_triton", "ATS · Triton (ours)", "X", "darkorange"),
    ]:
        if key not in results or not results[key]:
            continue
        vals = results[key]
        valid_bs = [b for b, v in zip(bs, vals) if v > 0]
        valid_v  = [v for v in vals if v > 0]
        kwargs = {"color": color} if color else {}
        kwms = {"markersize": 10} if marker in ("*", "X") else {}
        ax.plot(valid_bs, valid_v, f"{marker}-", label=label, linewidth=2, **kwargs, **kwms)

    ax.set_xlabel("Batch Size", fontsize=12)
    ax.set_ylabel("Peak VRAM (MB)", fontsize=12)
    ax.set_title("Benchmark 3: Peak VRAM Allocation", fontsize=14)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(bs)

    plt.tight_layout()
    plt.savefig("results/bench3_vram.png", dpi=150)
    plt.savefig("results/bench3_vram.pdf")
    print("Saved: results/bench3_vram.png")


if __name__ == "__main__":
    run_benchmark_3()
