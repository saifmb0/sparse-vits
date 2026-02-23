# Ragged-Batch ViT Inference Engine

A systems-level research project that demonstrates how **Triton GPU kernels** eliminate the padding waste inherent in variable-length (ragged) token sequences after dynamic token pruning in Vision Transformers.

## What This Project Does

When you prune tokens from a Vision Transformer, each image in a batch ends up with a **different number of tokens**. Standard PyTorch handles this by padding every image back to the same length — wasting compute on zeros. This project builds two custom Triton kernels that process ragged batches natively, achieving real speedups that track the theoretical ideal.

We benchmark our Triton ragged-attention engine against **four different pruning strategies** × **two execution backends** (PyTorch padded vs. Triton ragged), for a total of **9 model configurations**.

## Architecture

```
Input Images [B, 3, 224, 224]
        │
        ▼
┌─────────────────────────┐
│  DeiT-Small Front-End   │  Patch embed + CLS + Positional
│  (pretrained, frozen)   │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  Early Blocks (0–3)     │  4 layers at full 197-token sequence
│  (pretrained, frozen)   │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  PRUNING DECISION       │  ← One of 4 strategies (see below)
│  → keep_mask [B, 197]   │
└─────────────────────────┘
        │
        ├──── PyTorch path: gather + zero-pad + attention mask
        │
        ├──── Triton path:  Fused Token Packer → flat [Total_Kept, D]
        │
        ▼
┌─────────────────────────┐
│  Late Blocks (4–11)     │  8 layers on pruned tokens
│  (pretrained, frozen)   │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  CLS → LayerNorm → Head │  → logits [B, 1000]
└─────────────────────────┘
```

## Pruning Strategies Compared

| Strategy | Type | Extra Params | Mask Overhead | How It Decides |
|---|---|---|---|---|
| **Threshold-L2** | Heuristic | 0 | Minimal | L2-norm of token embeddings → top-k |
| **DynamicViT** | Learned (active) | ~150K | MLP forward pass | Trained MLP gate predicts keep probability |
| **EViT** | Attention-based (passive) | 0 | Near-zero | CLS→patch attention scores from last early block |
| **ATS** | Statistical (CDF) | 0 | Moderate (sort+cumsum) | CLS-attn × value-norm → PDF → CDF inverse sampling |

Each strategy is tested with two execution backends:
- **PyTorch (pad)** — gather kept tokens, zero-pad back to max length, run with attention mask
- **Triton (ours)** — pack into a flat ragged tensor, run with cu_seqlens-guided kernels

Plus the **Unpruned DeiT-S** baseline (no pruning, standard forward pass).

## Custom Triton Kernels

### 1. Fused Token Packer (`kernels/pack_tokens.py`)
Packs kept tokens from `[B, S, D]` into a flat `[Total_Kept, D]` tensor with cumulative sequence length offsets (`cu_seqlens`). Index computation via PyTorch `nonzero()`, vectorized D-wide copy via Triton.

### 2. Ragged Self-Attention (`kernels/ragged_attention.py`)
FlashAttention-2–style fused self-attention that processes variable-length sequences using `cu_seqlens` boundaries. Online softmax, tiled SRAM blocks (BLOCK_M=32, BLOCK_N=32), bidirectional. Zero padding overhead — FLOPs scale with actual token count, not padded length.

## Benchmarks

### Benchmark 1: Throughput Scaling
Measures images/sec across batch sizes (1, 4, 8, 16, 32) at 50% pruning. Shows how Triton variants maintain throughput advantage as batch size grows.

### Benchmark 2: Sparsity vs. Speedup
Sweeps pruning ratio from 0% to 90% at fixed batch size. Compares real speedup against the theoretical ideal. Triton ragged variants track the ideal curve; PyTorch padded variants stay flat (padding waste negates the pruning benefit).

### Benchmark 3: Peak VRAM
Measures `torch.cuda.max_memory_allocated()` across batch sizes. Triton ragged uses significantly less memory since it never materializes padded tensors.

## Project Structure

```
triton_pruner/
├── config.py                       # Global constants (model, pruning, benchmarking)
├── smoke_test.py                   # Quick correctness checks (10 tests)
├── run_all.py                      # Master runner (--bench N, --plot)
│
├── models/
│   ├── deit_base.py                # DeiT-Small loader + front/early/late/back splitter
│   ├── pruning.py                  # Threshold-L2 mask + PyTorch gather-and-pad
│   ├── triton_ragged_deit.py       # Threshold-L2 + Triton ragged pipeline
│   ├── dynamicvit_gate.py          # DynamicViT MLP prediction module
│   ├── dynamicvit_ragged.py        # DynamicViT + Triton ragged pipeline
│   ├── evit_gate.py                # EViT CLS-attention mask + attention-capture block
│   ├── evit_ragged.py              # EViT + Triton ragged pipeline
│   ├── ats_gate.py                 # ATS CDF-based inverse transform sampling
│   └── ats_ragged.py               # ATS + Triton ragged pipeline
│
├── baselines/
│   ├── unpruned.py                 # Standard DeiT-S forward pass
│   ├── pytorch_pruned.py           # Threshold-L2 + PyTorch padded baseline
│   ├── dynamicvit_pytorch.py       # DynamicViT + PyTorch padded baseline
│   ├── evit_pytorch.py             # EViT + PyTorch padded baseline
│   └── ats_pytorch.py              # ATS + PyTorch padded baseline
│
├── kernels/
│   ├── pack_tokens.py              # Triton Kernel 1: Fused Token Packer
│   └── ragged_attention.py         # Triton Kernel 2: Ragged Self-Attention
│
├── benchmarks/
│   ├── bench1_throughput.py        # Batch-size scaling (throughput)
│   ├── bench2_sparsity.py          # Sparsity vs. speedup
│   └── bench3_vram.py              # Peak VRAM measurement
│
└── results/                        # Auto-generated JSON + PNG/PDF plots
```

## Requirements

- Python 3.10+
- PyTorch 2.x with CUDA support
- Triton 3.x
- timm (for pretrained DeiT-Small weights)
- matplotlib (for benchmark plots)

## Quick Start

```bash
# 1. Smoke test — verify everything loads and produces valid output
python smoke_test.py

# 2. Run all benchmarks
python run_all.py

# 3. Run a specific benchmark
python run_all.py --bench 1   # throughput scaling
python run_all.py --bench 2   # sparsity vs. speedup
python run_all.py --bench 3   # peak VRAM

# 4. Re-generate plots from saved results
python run_all.py --plot
```

## Hardware

Developed and benchmarked on:
- **GPU:** NVIDIA GTX 1650 (4 GB VRAM)
- **Precision:** FP16 throughout
- **Model:** DeiT-Small (22M params, 384-dim, 6 heads, 12 layers)

## Key Insight

The central finding: **the execution backend matters more than the pruning strategy.** All four pruning methods achieve similar theoretical FLOPs reduction, but only the Triton ragged backend converts those reduced FLOPs into real wall-clock speedups. The PyTorch padded backend wastes the savings on zero-padding overhead, regardless of how clever the pruning strategy is.

| Pruning Signal | Overhead | Ragged Quality | Best Use Case |
|---|---|---|---|
| Threshold-L2 | Negligible | Moderate variance | Simple baseline |
| DynamicViT | Highest (MLP gate) | Moderate variance | When you can afford retraining |
| EViT | Near-zero | Moderate variance | Drop-in replacement, no retraining |
| ATS | Moderate (sort+cumsum) | Highest variance | Stress-testing ragged engines |
