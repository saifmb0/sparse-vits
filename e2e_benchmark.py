#!/usr/bin/env python3
"""
e2e_benchmark.py
================
End-to-end throughput comparison: three full DeiT-B inference pipelines.

Pipelines under test
--------------------
  1. PyTorch Padded (SDPA)      — pad kept tokens to max_len, additive mask
  2. Triton Ragged (ours)       — flat packed tokens, cu_seqlens Triton kernel
  3. FlashAttention-2 (varlen)  — flat packed tokens, FA2 flash_attn_varlen_func
                                  (skipped with --no-fa2 for GPUs without varlen API)

All three use the same front-end (patch embed + early blocks) and the same
threshold-L2 token pruning at fixed_ratio=0.5.  Only the late-block
attention implementation differs.

Inputs: real ImageNet-1K validation images streamed from HuggingFace
        (set HF_TOKEN env var or pass --hf-token).

Timing: CUDA events per iteration, median over BENCH_ITERS reported.
"""

import argparse
import gc
import json
import os
import statistics
import sys

import torch
# cuDNN 9.x fails to initialize on this system; disable it so the
# patch-embed conv2d falls back to the native CUDA path.  Has no
# effect on the transformer blocks (which are all matmul / attention).
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import timm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME       = "deit_base_patch16_224"
EMBED_DIM        = 768
NUM_HEADS        = 12
HEAD_DIM         = EMBED_DIM // NUM_HEADS    # 64
PRUNE_AFTER      = 4
DEVICE           = "cuda"
DTYPE            = torch.float16
FIXED_RATIO      = 0.5                       # 50% tokens dropped

BATCH_SIZES         = [1, 4, 8, 16, 32, 64, 128]
WARMUP_ITERS        = 20
BENCH_ITERS         = 100
IMG_SIZE            = 224
DEFAULT_IN_SAMPLES = 1000  # ImageNet val images to stream


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernels (inline, SM89-tuned)
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _pack_copy_kernel(tokens_ptr, packed_ptr, src_idx_ptr,
                      D: tl.constexpr, BLOCK_D: tl.constexpr):
    pid     = tl.program_id(0)
    src_row = tl.load(src_idx_ptr + pid).to(tl.int64)
    cols    = tl.arange(0, BLOCK_D)
    mask    = cols < D
    vals    = tl.load(tokens_ptr + src_row * D + cols, mask=mask, other=0.0)
    tl.store(packed_ptr + pid * D + cols, vals, mask=mask)


def triton_pack_tokens(tokens, mask):
    B, S, D = tokens.shape
    mask_i32   = mask.to(torch.int32)
    counts     = mask_i32.sum(dim=1)
    cu_seqlens = torch.zeros(B + 1, dtype=torch.int32, device=DEVICE)
    torch.cumsum(counts, dim=0, out=cu_seqlens[1:])
    total      = int(cu_seqlens[-1].item())
    src_idx    = mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
    packed     = torch.empty(total, D, dtype=tokens.dtype, device=DEVICE)
    if total > 0:
        _pack_copy_kernel[(total,)](
            tokens.reshape(-1, D), packed, src_idx,
            D=D, BLOCK_D=triton.next_power_of_2(D),
        )
    return packed, cu_seqlens


@triton.jit
def _ragged_attn_fwd(Q_ptr, K_ptr, V_ptr, O_ptr, cu_seqlens_ptr,
                     num_heads: tl.constexpr, head_dim: tl.constexpr,
                     stride_tok: tl.constexpr, stride_head: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                     BLOCK_D: tl.constexpr):
    pid      = tl.program_id(0)
    head_idx = pid % num_heads
    img_idx  = pid // num_heads
    seq_start = tl.load(cu_seqlens_ptr + img_idx).to(tl.int32)
    seq_end   = tl.load(cu_seqlens_ptr + img_idx + 1).to(tl.int32)
    seq_len   = seq_end - seq_start
    if seq_len <= 0:
        return
    sm_scale = 1.0 / tl.sqrt(tl.cast(BLOCK_D, tl.float32))
    d_offs   = tl.arange(0, BLOCK_D)
    for m_start in range(0, seq_len, BLOCK_M):
        m_offs    = m_start + tl.arange(0, BLOCK_M)
        m_mask    = m_offs < seq_len
        q_tok_ids = seq_start + tl.minimum(m_offs, seq_len - 1)
        q_ptrs = (Q_ptr + q_tok_ids[:, None] * stride_tok
                  + head_idx * stride_head + d_offs[None, :])
        q   = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
        m_i = tl.full([BLOCK_M], value=-1e9, dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        for n_start in range(0, seq_len, BLOCK_N):
            n_offs     = n_start + tl.arange(0, BLOCK_N)
            n_mask     = n_offs < seq_len
            kv_tok_ids = seq_start + tl.minimum(n_offs, seq_len - 1)
            k_ptrs = (K_ptr + kv_tok_ids[:, None] * stride_tok
                      + head_idx * stride_head + d_offs[None, :])
            k     = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)
            s     = tl.dot(q, tl.trans(k)) * sm_scale
            s     = tl.where(m_mask[:, None] & n_mask[None, :], s, -1e9)
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp(m_i - m_new)
            p     = tl.exp(s - m_new[:, None])
            l_new = alpha * l_i + tl.sum(p, axis=1)
            v_ptrs = (V_ptr + kv_tok_ids[:, None] * stride_tok
                      + head_idx * stride_head + d_offs[None, :])
            v   = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)
            m_i = m_new
            l_i = l_new
        acc   = acc / l_i[:, None]
        o_ptrs = (O_ptr + q_tok_ids[:, None] * stride_tok
                  + head_idx * stride_head + d_offs[None, :])
        tl.store(o_ptrs, acc.to(tl.float16), mask=m_mask[:, None])


def triton_ragged_attention(q, k, v, cu_seqlens):
    Total, H, D = q.shape
    B    = cu_seqlens.shape[0] - 1
    out  = torch.empty_like(q)
    _ragged_attn_fwd[(B * H,)](
        q, k, v, out, cu_seqlens,
        num_heads=H, head_dim=D,
        stride_tok=q.stride(0), stride_head=q.stride(1),
        BLOCK_M=64, BLOCK_N=64,
        BLOCK_D=triton.next_power_of_2(D),
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Model / pruning helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_deit():
    model = timm.create_model(MODEL_NAME, pretrained=True)
    return model.to(DEVICE, dtype=DTYPE).eval()


class _FrontEnd(nn.Module):
    def __init__(self, deit):
        super().__init__()
        self.patch_embed   = deit.patch_embed
        self.cls_token     = deit.cls_token
        self.pos_embed     = deit.pos_embed
        self.pos_drop      = getattr(deit, "pos_drop", nn.Identity())
        self.no_embed_class = getattr(deit, "no_embed_class", False)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        if self.no_embed_class:
            x = x + self.pos_embed
            x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        else:
            x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
            x = x + self.pos_embed
        return self.pos_drop(x)


def _threshold_prune(x, ratio):
    """Returns bool keep-mask [B, S]; CLS always kept."""
    B, S, _ = x.shape
    scores   = x.norm(dim=-1)
    n_keep   = max(1, int(S * (1.0 - ratio)))
    topk_idx = scores.topk(n_keep, dim=1).indices
    mask     = torch.zeros(B, S, dtype=torch.bool, device=x.device)
    mask.scatter_(1, topk_idx, True)
    mask[:, 0] = True   # CLS always kept
    return mask


def _gather_pad(x, mask):
    """Gather kept tokens into [B, S, D] padded form + bool mask."""
    B, S, D = x.shape
    padded   = torch.zeros_like(x)
    attn_m   = torch.zeros(B, S, dtype=torch.bool, device=x.device)
    for i in range(B):
        kept = x[i][mask[i]]
        K    = kept.shape[0]
        padded[i, :K] = kept
        attn_m[i, :K] = True
    return padded, attn_m


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline 1 — PyTorch Padded (SDPA with additive mask)
# ─────────────────────────────────────────────────────────────────────────────

class _PaddedAttn(nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.qkv      = attn.qkv
        self.proj     = attn.proj
        self.proj_drop = attn.proj_drop
        self.num_heads = attn.num_heads
        self.head_dim  = attn.head_dim

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scale = self.head_dim ** -0.5
        attn  = (q @ k.transpose(-2, -1)) * scale
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = attn.softmax(dim=-1)
        out  = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))


class _PaddedBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.norm1      = block.norm1
        self.attn       = _PaddedAttn(block.attn)
        self.norm2      = block.norm2
        self.mlp        = block.mlp
        self.ls1        = getattr(block, "ls1", nn.Identity())
        self.ls2        = getattr(block, "ls2", nn.Identity())
        self.drop_path1 = getattr(block, "drop_path1", nn.Identity())
        self.drop_path2 = getattr(block, "drop_path2", nn.Identity())

    def forward(self, x, attn_mask=None):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class PaddedPipeline(nn.Module):
    def __init__(self, deit):
        super().__init__()
        self.front       = _FrontEnd(deit)
        self.early       = nn.Sequential(*list(deit.blocks[:PRUNE_AFTER]))
        self.late_blocks = nn.ModuleList(
            [_PaddedBlock(b) for b in list(deit.blocks[PRUNE_AFTER:])]
        )
        self.norm = deit.norm
        self.head = deit.head

    @torch.inference_mode()
    def forward(self, images):
        x    = self.early(self.front(images))
        mask = _threshold_prune(x, FIXED_RATIO)
        x_pad, attn_m = _gather_pad(x, mask)
        B, S, D = x_pad.shape
        amask = x_pad.new_zeros(B, 1, 1, S)
        amask.masked_fill_(~attn_m.unsqueeze(1).unsqueeze(2), float("-inf"))
        for blk in self.late_blocks:
            x_pad = blk(x_pad, amask)
        cls = self.norm(x_pad[:, 0])
        return self.head(cls)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline 2 — Triton Ragged
# ─────────────────────────────────────────────────────────────────────────────

class _RaggedBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.norm1      = block.norm1
        self.qkv_proj   = block.attn.qkv
        self.out_proj   = block.attn.proj
        self.proj_drop  = block.attn.proj_drop
        self.num_heads  = block.attn.num_heads
        self.head_dim   = block.attn.head_dim
        self.norm2      = block.norm2
        self.mlp        = block.mlp
        self.ls1        = getattr(block, "ls1", nn.Identity())
        self.ls2        = getattr(block, "ls2", nn.Identity())
        self.drop_path1 = getattr(block, "drop_path1", nn.Identity())
        self.drop_path2 = getattr(block, "drop_path2", nn.Identity())

    def forward(self, x, cu_seqlens):
        Total, D = x.shape
        H, d     = self.num_heads, self.head_dim
        res      = x
        qkv = self.qkv_proj(self.norm1(x)).reshape(Total, 3, H, d)
        q, k, v = qkv[:, 0].contiguous(), qkv[:, 1].contiguous(), qkv[:, 2].contiguous()
        attn_out = triton_ragged_attention(q, k, v, cu_seqlens).reshape(Total, D)
        attn_out = self.proj_drop(self.out_proj(attn_out))
        x = res + self.drop_path1(self.ls1(attn_out))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class TritonRaggedPipeline(nn.Module):
    def __init__(self, deit):
        super().__init__()
        self.front        = _FrontEnd(deit)
        self.early        = nn.Sequential(*list(deit.blocks[:PRUNE_AFTER]))
        self.ragged_blocks = nn.ModuleList(
            [_RaggedBlock(b) for b in list(deit.blocks[PRUNE_AFTER:])]
        )
        self.norm = deit.norm
        self.head = deit.head

    @torch.inference_mode()
    def forward(self, images):
        x    = self.early(self.front(images))
        mask = _threshold_prune(x, FIXED_RATIO)
        packed, cu = triton_pack_tokens(x, mask)
        for blk in self.ragged_blocks:
            packed = blk(packed, cu)
        cls = self.norm(packed[cu[:-1].long()])
        return self.head(cls)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline 3 — FlashAttention-2 varlen
# ─────────────────────────────────────────────────────────────────────────────

class _FA2Block(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.norm1      = block.norm1
        self.qkv_proj   = block.attn.qkv
        self.out_proj   = block.attn.proj
        self.proj_drop  = block.attn.proj_drop
        self.num_heads  = block.attn.num_heads
        self.head_dim   = block.attn.head_dim
        self.norm2      = block.norm2
        self.mlp        = block.mlp
        self.ls1        = getattr(block, "ls1", nn.Identity())
        self.ls2        = getattr(block, "ls2", nn.Identity())
        self.drop_path1 = getattr(block, "drop_path1", nn.Identity())
        self.drop_path2 = getattr(block, "drop_path2", nn.Identity())

    def forward(self, x, cu_seqlens, max_seqlen):
        from flash_attn import flash_attn_varlen_func
        Total, D = x.shape
        H, d     = self.num_heads, self.head_dim
        res      = x
        qkv = self.qkv_proj(self.norm1(x)).reshape(Total, 3, H, d)
        q, k, v = qkv[:, 0].contiguous(), qkv[:, 1].contiguous(), qkv[:, 2].contiguous()
        attn_out = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            causal=False,
        ).reshape(Total, D)
        attn_out = self.proj_drop(self.out_proj(attn_out))
        x = res + self.drop_path1(self.ls1(attn_out))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class FA2VarlenPipeline(nn.Module):
    def __init__(self, deit):
        super().__init__()
        self.front     = _FrontEnd(deit)
        self.early     = nn.Sequential(*list(deit.blocks[:PRUNE_AFTER]))
        self.fa2_blocks = nn.ModuleList(
            [_FA2Block(b) for b in list(deit.blocks[PRUNE_AFTER:])]
        )
        self.norm = deit.norm
        self.head = deit.head

    @torch.inference_mode()
    def forward(self, images):
        x    = self.early(self.front(images))
        mask = _threshold_prune(x, FIXED_RATIO)
        packed, cu = triton_pack_tokens(x, mask)
        lengths   = (cu[1:] - cu[:-1])
        max_seqlen = int(lengths.max().item())
        for blk in self.fa2_blocks:
            packed = blk(packed, cu, max_seqlen)
        cls = self.norm(packed[cu[:-1].long()])
        return self.head(cls)


# ─────────────────────────────────────────────────────────────────────────────
# ImageNet-1K data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_imagenet_val(max_samples: int = DEFAULT_IN_SAMPLES, hf_token: str = None):
    """
    Stream ImageNet-1K validation images from HuggingFace.
    Returns a list of float16 tensors [3, 224, 224] on CPU.
    Images are kept on CPU and moved to GPU per-batch to avoid OOM.
    """
    from datasets import load_dataset
    from torchvision import transforms

    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN not set. Export it or pass --hf-token. "
            "Get one at https://huggingface.co/settings/tokens"
        )

    print(f"Streaming {max_samples} ImageNet-1K val images from HuggingFace...",
          flush=True)
    ds = load_dataset(
        "ILSVRC/imagenet-1k",
        split="validation",
        streaming=True,
        token=token,
    )
    # Shuffle so we sample uniformly across all 1000 classes rather than
    # taking the first N images in class-sorted order.
    # buffer_size=5000 gives good mixing (val set = 50 imgs/class × 1000 classes).
    # seed=42 makes the sample reproducible across machines.
    ds = ds.shuffle(seed=42, buffer_size=5000)

    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    tensors = []
    for i, sample in enumerate(ds):
        if i >= max_samples:
            break
        img = sample["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        tensors.append(tfm(img).to(DTYPE))   # float16, CPU
        if (i + 1) % 200 == 0:
            print(f"  ... {i + 1}/{max_samples}", flush=True)

    print(f"  Loaded {len(tensors)} images.", flush=True)
    return tensors


def _make_batch(imagenet_tensors: list, bs: int) -> torch.Tensor:
    """
    Sample a batch of `bs` images from the pre-loaded ImageNet tensors.
    Wraps around if bs > len(tensors). Returned tensor is on GPU.
    """
    n = len(imagenet_tensors)
    indices = [i % n for i in range(bs)]
    batch = torch.stack([imagenet_tensors[i] for i in indices])
    return batch.to(DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# Timing
# ─────────────────────────────────────────────────────────────────────────────


def cuda_event_bench(model, images, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    for _ in range(warmup):
        model(images)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        model(images)
        ends[i].record()
    torch.cuda.synchronize()

    times_ms = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    med_ms   = statistics.median(times_ms)
    lo_ms    = times_ms[int(0.05 * iters)]
    hi_ms    = times_ms[int(0.95 * iters) - 1]
    throughput = (images.shape[0] / (med_ms / 1000.0))
    return throughput, med_ms, lo_ms, hi_ms


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_e2e(results_dir="results/RTX4000Ada",
            imagenet_tensors=None,
            no_fa2=False):
    gpu_name = torch.cuda.get_device_name(0)
    print("\n" + "=" * 70)
    print(f"{gpu_name} — End-to-End Pipeline Throughput (ImageNet-1K, CUDA-event timing)")
    print("=" * 70)
    print(f"Model  : {MODEL_NAME}")
    print(f"Pruning: {int(FIXED_RATIO*100)}% tokens dropped (threshold-L2)")
    print(f"Images : {len(imagenet_tensors)} real ImageNet-1K val images")
    print(f"Timing : {BENCH_ITERS} iterations, median throughput reported")
    print(f"Warmup : {WARMUP_ITERS} iterations\n")

    print("Loading DeiT-B weights...", flush=True)
    deit = _load_deit()

    pipelines = {}
    print("Building pipelines...", flush=True)

    padded = PaddedPipeline(deit).to(DEVICE, dtype=DTYPE).eval()
    pipelines["PyTorch Padded (SDPA)"] = padded

    triton_p = TritonRaggedPipeline(deit).to(DEVICE, dtype=DTYPE).eval()
    pipelines["Triton Ragged (ours)"] = triton_p

    if not no_fa2:
        try:
            from flash_attn import flash_attn_varlen_func  # noqa: F401
            fa2_p = FA2VarlenPipeline(deit).to(DEVICE, dtype=DTYPE).eval()
            pipelines["FlashAttention-2 (varlen)"] = fa2_p
            print("FlashAttention-2 available — including in comparison.\n")
        except ImportError:
            print("flash_attn not installed — skipping FA2 pipeline.\n")
    else:
        print("--no-fa2 set — skipping FlashAttention-2 varlen pipeline.\n")

    del deit
    torch.cuda.empty_cache()

    results = {
        "gpu":              gpu_name,
        "model":           MODEL_NAME,
        "fixed_ratio":     FIXED_RATIO,
        "timing":          "cuda_events_median_throughput",
        "input":           "imagenet1k_val",
        "imagenet_samples": len(imagenet_tensors),
        "batch_sizes":     BATCH_SIZES,
        "pipelines":       {k: [] for k in pipelines},
        "latency_ms":      {k: [] for k in pipelines},
        "p5_ms":           {k: [] for k in pipelines},
        "p95_ms":          {k: [] for k in pipelines},
    }

    for bs in BATCH_SIZES:
        print(f"[BS={bs:3d}]", flush=True)
        images = _make_batch(imagenet_tensors, bs)
        for name, model in pipelines.items():
            try:
                tp, med, lo, hi = cuda_event_bench(model, images)
                print(f"  {name:35s}  {tp:8.1f} img/s  ({med:.2f} ms [{lo:.2f}\u2013{hi:.2f}])")
                results["pipelines"][name].append(round(tp, 1))
                results["latency_ms"][name].append(round(med, 3))
                results["p5_ms"][name].append(round(lo,  3))
                results["p95_ms"][name].append(round(hi,  3))
            except Exception as exc:
                print(f"  {name:35s}  FAILED: {exc}")
                for key in ("pipelines", "latency_ms", "p5_ms", "p95_ms"):
                    results[key][name].append(-1.0)
        del images
        torch.cuda.empty_cache()
        gc.collect()

    return results


def plot_e2e(results, results_dir):
    bs_list   = results["batch_sizes"]
    pipelines = results["pipelines"]

    styles = {
        "PyTorch Padded (SDPA)":      ("s--", "#d62728"),
        "Triton Ragged (ours)":       ("D-",  "#2ca02c"),
        "FlashAttention-2 (varlen)":  ("^-",  "#9467bd"),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Throughput ──────────────────────────────────────────────────
    for name, vals in pipelines.items():
        marker, color = styles.get(name, ("o-", "gray"))
        valid = [(b, v) for b, v in zip(bs_list, vals) if v > 0]
        if valid:
            bx, vy = zip(*valid)
            ax1.plot(bx, vy, marker, label=name, color=color,
                     linewidth=2, markersize=8)

    ax1.set_xlabel("Batch size", fontsize=12)
    ax1.set_ylabel("Throughput (img/s, higher is better)", fontsize=11)
    ax1.set_title(
        f"E2E Throughput — DeiT-B, {int(FIXED_RATIO*100)}% pruned\n"
        f"RTX 4000 Ada (SM89), CUDA-event timing",
        fontsize=11, fontweight="bold",
    )
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── Latency (ms) ────────────────────────────────────────────────
    lat = results["latency_ms"]
    for name, vals in lat.items():
        marker, color = styles.get(name, ("o-", "gray"))
        valid = [(b, v) for b, v in zip(bs_list, vals) if v > 0]
        if valid:
            bx, vy = zip(*valid)
            p5v  = [results["p5_ms"][name][bs_list.index(b)] for b in bx]
            p95v = [results["p95_ms"][name][bs_list.index(b)] for b in bx]
            ax2.plot(bx, vy, marker, label=name, color=color,
                     linewidth=2, markersize=8)
            ax2.fill_between(bx, p5v, p95v, alpha=0.12, color=color)

    ax2.set_xlabel("Batch size", fontsize=12)
    ax2.set_ylabel("Latency (ms, lower is better)", fontsize=11)
    ax2.set_title(
        f"E2E Latency — DeiT-B, {int(FIXED_RATIO*100)}% pruned\n"
        "shaded band = p5–p95",
        fontsize=11, fontweight="bold",
    )
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(results_dir, exist_ok=True)
    out = os.path.join(results_dir, "e2e_benchmark.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✓  Saved {out}")

    # ── speedup table ───────────────────────────────────────────────
    ref = "FlashAttention-2 (varlen)"
    if ref in pipelines:
        print(f"\nSpeedup over '{ref}' (throughput):")
        ref_vals = pipelines[ref]
        for name, vals in pipelines.items():
            if name == ref:
                continue
            row = []
            for r, v in zip(ref_vals, vals):
                row.append(f"{v/r:.2f}×" if r > 0 and v > 0 else "  N/A")
            print(f"  {name:35s}  {' | '.join(row)}")
        print(f"  Batch sizes: {bs_list}")


def main():
    parser = argparse.ArgumentParser(description="E2E throughput benchmark (ImageNet-1K)")
    parser.add_argument("--results-dir", default="results/RTX4000Ada",
                        help="directory for JSON + PNG output")
    parser.add_argument("--no-fa2", action="store_true",
                        help="Skip FlashAttention-2 varlen pipeline "
                             "(use on GPUs without FA2 varlen support, e.g. T4/GTX1650)")
    parser.add_argument("--imagenet-samples", type=int, default=DEFAULT_IN_SAMPLES,
                        help=f"Number of ImageNet-1K val images to stream (default {DEFAULT_IN_SAMPLES})")
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace token (falls back to HF_TOKEN env var)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA not available")

    # ── Load real ImageNet-1K images once, reuse across all batch sizes ──
    imagenet_tensors = load_imagenet_val(
        max_samples=args.imagenet_samples,
        hf_token=args.hf_token,
    )

    results_dir = args.results_dir
    results = run_e2e(
        results_dir=results_dir,
        imagenet_tensors=imagenet_tensors,
        no_fa2=args.no_fa2,
    )
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "e2e_benchmark.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\u2713  Saved {path}")
    plot_e2e(results, results_dir)


if __name__ == "__main__":
    main()
