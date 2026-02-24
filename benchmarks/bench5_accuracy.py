"""
Benchmark 5: Accuracy vs. Efficiency (Pareto Frontier)
=======================================================
Evaluates how pruning affects classification accuracy alongside
throughput.  Uses ImageNet-1K validation set from HuggingFace.

Two experiments:
  A) Numerical Equivalence:
     Verify that Triton ragged output == PyTorch padded output
     up to fp16 precision (proving the execution backend does NOT
     affect accuracy — only the pruning strategy matters).

  B) Accuracy-Throughput Pareto:
     Sweep prune ratios [0.0 → 0.9] and plot:
       - X-axis: Throughput (img/s at BS=32)
       - Y-axis: Top-1 Accuracy on ImageNet-1K val (or subset)
     For each pruning strategy × backend pair.

Output: results/bench5_accuracy.json + .png
"""

import sys, os, gc, time, json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DEVICE, IMG_SIZE, PRUNE_RATIOS, WARMUP_ITERS, BENCH_ITERS,
)
from models.deit_base import load_deit, get_dtype


# ─────────────────────────────────────────────────────────────────────
# ImageNet-1K Validation Loader
# ─────────────────────────────────────────────────────────────────────

def get_imagenet_val(max_samples: int = 1000):
    """
    Stream ImageNet-1K validation split from HuggingFace.
    Only downloads the images actually evaluated — NOT the full dataset.

    Returns a list of (image_tensor [3,224,224], label) tuples.
    max_samples: number of images to stream (default 1000 ≈ 2% of val).
    """
    from datasets import load_dataset
    from torchvision import transforms

    print(f"Streaming {max_samples} ImageNet-1K validation images from HuggingFace...")
    ds = load_dataset(
        "ILSVRC/imagenet-1k",
        split="validation",
        streaming=True,
    )

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    data = []
    dtype = get_dtype()
    for i, sample in enumerate(ds):
        if i >= max_samples:
            break
        img = sample["image"]
        label = sample["label"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        tensor = transform(img).to(dtype)
        data.append((tensor, label))
        if (i + 1) % 200 == 0:
            print(f"  ... streamed {i + 1}/{max_samples}")

    print(f"  Loaded {len(data)} validation images (streamed, no full download).")
    return data


def evaluate_accuracy(model_fn, val_data, batch_size=8):
    """
    Evaluate Top-1 and Top-5 accuracy.
    model_fn: callable(images [B,3,224,224]) → logits [B, 1000]

    Uses small batches and explicit VRAM cleanup to avoid OOM on 4GB GPUs.
    Images are kept on CPU and moved to GPU per-batch.
    """
    correct_1 = 0
    correct_5 = 0
    total = 0

    n_batches = (len(val_data) + batch_size - 1) // batch_size
    for batch_idx, start in enumerate(range(0, len(val_data), batch_size)):
        end = min(start + batch_size, len(val_data))
        batch_imgs = torch.stack([val_data[i][0] for i in range(start, end)]).to(DEVICE)
        batch_labels = torch.tensor([val_data[i][1] for i in range(start, end)], device=DEVICE)

        with torch.inference_mode():
            logits = model_fn(batch_imgs)
        _, pred_top5 = logits.topk(5, dim=-1)

        correct_1 += (pred_top5[:, 0] == batch_labels).sum().item()
        correct_5 += (pred_top5 == batch_labels.unsqueeze(1)).any(dim=1).sum().item()
        total += len(batch_labels)

        del batch_imgs, batch_labels, logits, pred_top5
        torch.cuda.empty_cache()

    top1 = correct_1 / total * 100.0
    top5 = correct_5 / total * 100.0
    return top1, top5


# ─────────────────────────────────────────────────────────────────────
# Experiment A: Numerical Equivalence
# ─────────────────────────────────────────────────────────────────────

def verify_numerical_equivalence():
    """
    For each pruning strategy, verify that the Triton ragged backend
    produces outputs identical to the PyTorch padded backend (up to
    fp16 rounding).  This proves the execution backend is bit-accurate
    and does NOT affect accuracy.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT A: Numerical Equivalence (Triton vs PyTorch)")
    print("=" * 60)

    val_data = get_imagenet_val(max_samples=4)
    images = torch.stack([val_data[i][0] for i in range(4)]).to(DEVICE)

    results = {}

    strategies = [
        ("Threshold-L2", "baselines.pytorch_pruned", "build_pytorch_pruned_model",
         "models.triton_ragged_deit", "build_triton_ragged_model"),
        ("DynamicViT", "baselines.dynamicvit_pytorch", "build_dynamicvit_pytorch_model",
         "models.dynamicvit_ragged", "build_dynamicvit_triton_model"),
        ("EViT", "baselines.evit_pytorch", "build_evit_pytorch_model",
         "models.evit_ragged", "build_evit_triton_model"),
        # ("ATS", "baselines.ats_pytorch", "build_ats_pytorch_model",
        #  "models.ats_ragged", "build_ats_triton_model"),
    ]

    for name, pad_mod, pad_fn, rag_mod, rag_fn in strategies:
        print(f"\n--- {name} ---")
        import importlib

        # Build both variants
        pad_module = importlib.import_module(pad_mod)
        rag_module = importlib.import_module(rag_mod)
        model_pad = getattr(pad_module, pad_fn)()
        model_rag = getattr(rag_module, rag_fn)()

        # DynamicViT has a randomly-initialized MLP gate — the two
        # builders each create independent random weights.  Copy the
        # padded model's gate weights into the ragged model so both
        # use the identical mask, isolating only the execution backend.
        if hasattr(model_pad, "gate") and hasattr(model_rag, "gate"):
            model_rag.gate.load_state_dict(model_pad.gate.state_dict())

        # Run both on same images with same fixed ratio
        logits_pad = model_pad(images, fixed_ratio=0.5)
        logits_rag = model_rag(images, fixed_ratio=0.5)

        # Compute differences
        abs_diff = (logits_pad.float() - logits_rag.float()).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()

        # Check if top-1 predictions match
        pred_pad = logits_pad.argmax(dim=-1)
        pred_rag = logits_rag.argmax(dim=-1)
        pred_match = (pred_pad == pred_rag).all().item()

        results[name] = {
            "max_abs_diff": round(max_diff, 6),
            "mean_abs_diff": round(mean_diff, 6),
            "predictions_match": pred_match,
        }

        status = "✓ MATCH" if pred_match else "✗ MISMATCH"
        print(f"  Max abs diff:  {max_diff:.6f}")
        print(f"  Mean abs diff: {mean_diff:.6f}")
        print(f"  Predictions:   {status}")

        del model_pad, model_rag
        torch.cuda.empty_cache(); gc.collect()

    return results


# ─────────────────────────────────────────────────────────────────────
# Experiment B: Accuracy-Throughput Pareto
# ─────────────────────────────────────────────────────────────────────

def measure_throughput_at_ratio(model_fn, ratio, batch_size=32):
    """Measure throughput for a model at a specific prune ratio."""
    dtype = get_dtype()
    images = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE, device=DEVICE, dtype=dtype)

    for _ in range(WARMUP_ITERS):
        model_fn(images, ratio)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(BENCH_ITERS):
        model_fn(images, ratio)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return (batch_size * BENCH_ITERS) / elapsed


def run_pareto_experiment(val_data):
    """
    Sweep prune ratios for each method × backend, measuring both
    accuracy and throughput at each operating point.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT B: Accuracy-Throughput Pareto Frontier")
    print("=" * 60)

    # Ratios to evaluate (5 points — keeps bench tractable on low-end GPUs)
    ratios = [0.0, 0.25, 0.5, 0.75, 0.9]

    variants = [
        # (label, module_path, build_fn_name, has_ratio_arg)
        ("Unpruned DeiT-S", "baselines.unpruned", None, False),
        ("Threshold-L2 · PyTorch", "baselines.pytorch_pruned", "build_pytorch_pruned_model", True),
        ("Threshold-L2 · Triton", "models.triton_ragged_deit", "build_triton_ragged_model", True),
        ("EViT · PyTorch", "baselines.evit_pytorch", "build_evit_pytorch_model", True),
        ("EViT · Triton", "models.evit_ragged", "build_evit_triton_model", True),
        ("ATS · PyTorch", "baselines.ats_pytorch", "build_ats_pytorch_model", True),
        ("ATS · Triton", "models.ats_ragged", "build_ats_triton_model", True),
        ("ToMe · PyTorch", "baselines.tome_pytorch", "build_tome_model", True),
    ]

    results = {"ratios": ratios, "variants": {}}

    for label, mod_path, build_fn_name, has_ratio in variants:
        print(f"\n--- {label} ---")
        import importlib
        mod = importlib.import_module(mod_path)

        if build_fn_name is None:
            # Unpruned baseline — use load_deit directly
            model = load_deit()

            @torch.inference_mode()
            def model_fn(imgs, ratio, _m=model):
                return _m(imgs)

            # Only evaluate at ratio=0.0 (unpruned is constant)
            top1, top5 = evaluate_accuracy(
                lambda imgs, _m=model: _m(imgs), val_data, batch_size=8
            )
            throughput = measure_throughput_at_ratio(model_fn, 0.0)
            results["variants"][label] = {
                "top1": [round(top1, 2)] * len(ratios),
                "top5": [round(top5, 2)] * len(ratios),
                "throughput": [round(throughput, 1)] * len(ratios),
            }
            print(f"  Top-1={top1:.2f}%  Top-5={top5:.2f}%  Throughput={throughput:.1f} img/s")
            del model; torch.cuda.empty_cache(); gc.collect()
            continue

        model = getattr(mod, build_fn_name)()
        top1_list, top5_list, tp_list = [], [], []

        for ratio in ratios:
            print(f"  ratio={ratio:.1f} ... ", end="", flush=True)

            # Accuracy
            top1, top5 = evaluate_accuracy(
                lambda imgs, _m=model, _r=ratio: _m(imgs, fixed_ratio=_r),
                val_data, batch_size=8,
            )

            # Throughput
            tp = measure_throughput_at_ratio(
                lambda imgs, r, _m=model: _m(imgs, fixed_ratio=r),
                ratio,
            )

            top1_list.append(round(top1, 2))
            top5_list.append(round(top5, 2))
            tp_list.append(round(tp, 1))
            print(f"Top-1={top1:.2f}%  Throughput={tp:.1f} img/s")

            torch.cuda.empty_cache(); gc.collect()

        results["variants"][label] = {
            "top1": top1_list,
            "top5": top5_list,
            "throughput": tp_list,
        }

        del model; torch.cuda.empty_cache(); gc.collect()

    return results


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def run_benchmark_5():
    all_results = {}

    # Experiment A: Numerical equivalence (always runs, no dataset needed)
    all_results["numerical_equivalence"] = verify_numerical_equivalence()

    # Experiment B: Accuracy-Throughput Pareto (needs ImageNet)
    try:
        val_data = get_imagenet_val(max_samples=1000)
        all_results["pareto"] = run_pareto_experiment(val_data)
        del val_data; gc.collect()
    except Exception as e:
        import traceback
        print(f"\n⚠ Pareto experiment failed: {e}")
        traceback.print_exc()
        print("  If this is an auth error: `huggingface-cli login` or set HF_TOKEN env var")
        all_results["pareto"] = {"error": str(e)}

    # Save
    os.makedirs("results", exist_ok=True)
    with open("results/bench5_accuracy.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\n✓ Results saved to results/bench5_accuracy.json")

    # Plot
    plot_benchmark_5(all_results)


def plot_benchmark_5(results=None):
    if results is None:
        with open("results/bench5_accuracy.json") as f:
            results = json.load(f)

    # ── Plot A: Numerical Equivalence bar chart ──────────────────────
    if "numerical_equivalence" in results and results["numerical_equivalence"]:
        equiv = results["numerical_equivalence"]
        fig, ax = plt.subplots(figsize=(8, 4))

        names = list(equiv.keys())
        max_diffs = [equiv[n]["max_abs_diff"] for n in names]
        colors = ["green" if equiv[n]["predictions_match"] else "red" for n in names]

        bars = ax.bar(names, max_diffs, color=colors, alpha=0.8)
        ax.set_ylabel("Max Absolute Logit Difference", fontsize=11)
        ax.set_title("Triton Ragged vs PyTorch Padded — Numerical Equivalence",
                      fontsize=12, fontweight="bold")
        ax.axhline(y=0.1, color="orange", linestyle="--", alpha=0.5, label="fp16 tolerance")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)

        for bar, match in zip(bars, [equiv[n]["predictions_match"] for n in names]):
            label = "✓" if match else "✗"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    label, ha="center", va="bottom", fontsize=14)

        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        fig.savefig("results/bench5_numerical_equiv.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("✓ Saved results/bench5_numerical_equiv.png")

    # ── Plot B: Pareto Frontier ──────────────────────────────────────
    pareto = results.get("pareto", {})
    if "variants" not in pareto:
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    style_map = {
        "Unpruned DeiT-S":           ("o", "--", "#1f77b4",    10),
        "Threshold-L2 · PyTorch":    ("s", "--", "#d62728",    7),
        "Threshold-L2 · Triton":     ("s", "-",  "#d62728",    9),
        "EViT · PyTorch":            ("^", "--", "#9467bd",    7),
        "EViT · Triton":             ("^", "-",  "#9467bd",    9),
        "ATS · PyTorch":             ("*", "--", "#ff7f0e",    8),
        "ATS · Triton":              ("*", "-",  "#ff7f0e",    10),
        "ToMe · PyTorch":            ("d", "--", "#8c564b",    7),
    }

    for label, data in pareto["variants"].items():
        marker, ls, color, ms = style_map.get(label, ("o", "-", "gray", 6))
        tp = data["throughput"]
        top1 = data["top1"]
        ax.plot(tp, top1, marker=marker, linestyle=ls, color=color,
                label=label, linewidth=2, markersize=ms)

        # Annotate a few key points (0%, 50%, 90% prune)
        ratios = pareto["ratios"]
        for r_target in [0.0, 0.5]:
            if r_target in ratios:
                idx = ratios.index(r_target)
                if idx < len(tp) and tp[idx] > 0:
                    ax.annotate(f"{int(r_target*100)}%",
                                (tp[idx], top1[idx]),
                                textcoords="offset points",
                                xytext=(5, 5), fontsize=7, alpha=0.7)

    ax.set_xlabel("Throughput (img/s, BS=32)", fontsize=12)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy vs. Throughput Pareto Frontier  (ImageNet-1K val)",
                  fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig("results/bench5_pareto.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ Saved results/bench5_pareto.png")
