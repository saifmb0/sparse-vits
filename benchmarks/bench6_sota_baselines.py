"""
Benchmark 6: SOTA Systems Baselines — Attention Kernel Comparison
==================================================================
Compares our Triton ragged attention kernel against the optimized
attention implementations available in PyTorch and external libraries:

  1. PyTorch SDPA (math)       — naive matmul fallback with padding+mask
  2. PyTorch SDPA (efficient)  — memory-efficient attention with padding+mask
  3. PyTorch SDPA (flash)      — SDPA with FlashAttention backend (PRIMARY BASELINE)
  4. NestedTensor + SDPA       — PyTorch's native ragged attention
  5. FlashAttention-2 (varlen) — Tri Dao's FA2 with native cu_seqlens (SECONDARY BASELINE)
  6. Our Triton Ragged         — custom cu_seqlens-guided kernel

Two experiments:
  A) Kernel-level micro-benchmark (attention only, no MLP)
  B) End-to-end pipeline comparison (full pruned inference)

Requirements:
  - A100 (SM80+) for PyTorch SDPA flash backend and FlashAttention-2
  - pip install flash-attn --no-build-isolation

Output: results/bench6_sota_baselines.json + .png
"""

import sys, os, gc, time, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DEVICE, IMG_SIZE, NUM_HEADS, HEAD_DIM, EMBED_DIM,
    WARMUP_ITERS, BENCH_ITERS,
)
from models.deit_base import get_dtype
from benchmarks.bench5_accuracy import get_imagenet_val


# ─────────────────────────────────────────────────────────────────────
# Attention Kernel Implementations
# ─────────────────────────────────────────────────────────────────────

def attn_pytorch_math_padded(q, k, v, cu_seqlens, max_len):
    """
    Standard PyTorch math-backend SDPA with zero-padding + attention mask.
    Q/K/V are flat [Total, H, D]; we reconstruct padded [B, H, S, D].
    """
    B = cu_seqlens.shape[0] - 1
    H, D = q.shape[1], q.shape[2]

    # Pad to [B, max_len, H, D]
    q_pad = torch.zeros(B, max_len, H, D, device=q.device, dtype=q.dtype)
    k_pad = torch.zeros_like(q_pad)
    v_pad = torch.zeros_like(q_pad)
    mask = torch.zeros(B, max_len, device=q.device, dtype=torch.bool)

    for i in range(B):
        s, e = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        length = e - s
        q_pad[i, :length] = q[s:e]
        k_pad[i, :length] = k[s:e]
        v_pad[i, :length] = v[s:e]
        mask[i, :length] = True

    # Reshape to [B, H, S, D] for SDPA
    q_pad = q_pad.transpose(1, 2)
    k_pad = k_pad.transpose(1, 2)
    v_pad = v_pad.transpose(1, 2)

    # Additive mask: 0 where valid, -inf where padded
    attn_mask = torch.zeros(B, 1, 1, max_len, device=q.device, dtype=q.dtype)
    attn_mask.masked_fill_(~mask.unsqueeze(1).unsqueeze(2), float("-inf"))

    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out = F.scaled_dot_product_attention(q_pad, k_pad, v_pad, attn_mask=attn_mask)

    # Unpad: extract valid tokens back to [Total, H, D]
    out = out.transpose(1, 2)  # [B, S, H, D]
    result = torch.zeros_like(q)
    for i in range(B):
        s, e = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        length = e - s
        result[s:e] = out[i, :length]

    return result


def attn_pytorch_efficient_padded(q, k, v, cu_seqlens, max_len):
    """Same as math but uses the memory-efficient SDPA backend."""
    B = cu_seqlens.shape[0] - 1
    H, D = q.shape[1], q.shape[2]

    q_pad = torch.zeros(B, max_len, H, D, device=q.device, dtype=q.dtype)
    k_pad = torch.zeros_like(q_pad)
    v_pad = torch.zeros_like(q_pad)
    mask = torch.zeros(B, max_len, device=q.device, dtype=torch.bool)

    for i in range(B):
        s, e = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        length = e - s
        q_pad[i, :length] = q[s:e]
        k_pad[i, :length] = k[s:e]
        v_pad[i, :length] = v[s:e]
        mask[i, :length] = True

    q_pad = q_pad.transpose(1, 2)
    k_pad = k_pad.transpose(1, 2)
    v_pad = v_pad.transpose(1, 2)

    attn_mask = torch.zeros(B, 1, 1, max_len, device=q.device, dtype=q.dtype)
    attn_mask.masked_fill_(~mask.unsqueeze(1).unsqueeze(2), float("-inf"))

    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION):
        out = F.scaled_dot_product_attention(q_pad, k_pad, v_pad, attn_mask=attn_mask)

    out = out.transpose(1, 2)
    result = torch.zeros_like(q)
    for i in range(B):
        s, e = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        length = e - s
        result[s:e] = out[i, :length]

    return result


def attn_sdpa_flash_padded(q, k, v, cu_seqlens, max_len):
    """
    PyTorch SDPA with FLASH_ATTENTION backend — primary baseline.
    Requires SM80+ (A100/H100). Uses the same pad+unpad pattern.
    """
    B = cu_seqlens.shape[0] - 1
    H, D = q.shape[1], q.shape[2]

    q_pad = torch.zeros(B, max_len, H, D, device=q.device, dtype=q.dtype)
    k_pad = torch.zeros_like(q_pad)
    v_pad = torch.zeros_like(q_pad)
    mask = torch.zeros(B, max_len, device=q.device, dtype=torch.bool)

    for i in range(B):
        s, e = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        length = e - s
        q_pad[i, :length] = q[s:e]
        k_pad[i, :length] = k[s:e]
        v_pad[i, :length] = v[s:e]
        mask[i, :length] = True

    q_pad = q_pad.transpose(1, 2)
    k_pad = k_pad.transpose(1, 2)
    v_pad = v_pad.transpose(1, 2)

    # FLASH_ATTENTION does not support arbitrary attention masks.
    # We pass no mask; attention scores for valid tokens vs padding will be computed,
    # but since we slice out the exact length afterwards, the padding doesn't affect valid lengths.
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
        out = F.scaled_dot_product_attention(q_pad, k_pad, v_pad)

    out = out.transpose(1, 2)
    result = torch.zeros_like(q)
    for i in range(B):
        s, e = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        length = e - s
        result[s:e] = out[i, :length]

    return result


def attn_flash_attn2_varlen(q, k, v, cu_seqlens, max_len):
    """
    FlashAttention-2 via flash_attn.flash_attn_varlen_func — secondary baseline.
    Operates directly on flat [Total, H, D] + cu_seqlens — native ragged support,
    no padding needed.  Apples-to-apples comparison with our Triton kernel.
    Requires SM80+ and: pip install flash-attn --no-build-isolation
    """
    from flash_attn import flash_attn_varlen_func
    out = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_len,
        max_seqlen_k=max_len,
        causal=False,
    )
    return out


def attn_nested_tensor_sdpa(q, k, v, cu_seqlens, max_len):
    """
    PyTorch NestedTensor + SDPA — the framework-native ragged path.
    Packs into jagged NestedTensors, runs SDPA, unpacks.
    """
    B = cu_seqlens.shape[0] - 1
    H, D = q.shape[1], q.shape[2]

    # Build list of per-image tensors
    q_list = [q[cu_seqlens[i]:cu_seqlens[i + 1]] for i in range(B)]
    k_list = [k[cu_seqlens[i]:cu_seqlens[i + 1]] for i in range(B)]
    v_list = [v[cu_seqlens[i]:cu_seqlens[i + 1]] for i in range(B)]

    # Create jagged nested tensors: each [S_i, H, D]
    q_nt = torch.nested.nested_tensor(q_list, layout=torch.jagged)
    k_nt = torch.nested.nested_tensor(k_list, layout=torch.jagged)
    v_nt = torch.nested.nested_tensor(v_list, layout=torch.jagged)

    # Transpose to [B, H, S_i, D] for SDPA
    q_nt = q_nt.transpose(1, 2)
    k_nt = k_nt.transpose(1, 2)
    v_nt = v_nt.transpose(1, 2)

    out_nt = F.scaled_dot_product_attention(q_nt, k_nt, v_nt)

    # Back to [B, S_i, H, D] and unpack
    out_nt = out_nt.transpose(1, 2)
    result = torch.zeros_like(q)
    for i in range(B):
        s, e = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        result[s:e] = out_nt[i]

    return result


def attn_triton_ragged(q, k, v, cu_seqlens, max_len):
    """Our Triton ragged attention kernel."""
    from kernels.ragged_attention import triton_ragged_attention
    return triton_ragged_attention(q, k, v, cu_seqlens)


# ─────────────────────────────────────────────────────────────────────
# Measurement
# ─────────────────────────────────────────────────────────────────────

def benchmark_kernel(kernel_fn, q, k, v, cu_seqlens, max_len,
                     warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """Returns latency in milliseconds."""
    # Warmup
    for _ in range(warmup):
        kernel_fn(q, k, v, cu_seqlens, max_len)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        kernel_fn(q, k, v, cu_seqlens, max_len)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return (elapsed / iters) * 1000.0  # ms


def build_imagenet_batch(val_data, batch_size):
    """Build a fixed batch from streamed ImageNet tensors."""
    batch = torch.stack([val_data[i][0] for i in range(batch_size)])
    return batch.to(DEVICE)


# ─────────────────────────────────────────────────────────────────────
# Experiment A: Kernel-level micro-benchmark
# ─────────────────────────────────────────────────────────────────────

def run_kernel_microbenchmark():
    """
    Compare attention kernels on ragged sequences at various batch sizes
    and sparsity levels.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT A: Attention Kernel Micro-benchmark")
    print("=" * 60)

    dtype = get_dtype()
    H, D = NUM_HEADS, HEAD_DIM

    from models.deit_base import load_deit, split_deit
    from models.pruning import threshold_prune_mask
    from kernels.pack_tokens import triton_pack_tokens

    kernels = {
        "SDPA (math, padded)": attn_pytorch_math_padded,
        "SDPA (efficient, padded)": attn_pytorch_efficient_padded,
        "SDPA (flash, padded)": attn_sdpa_flash_padded,
        "FlashAttention-2 (varlen)": attn_flash_attn2_varlen,
        "NestedTensor + SDPA": attn_nested_tensor_sdpa,
        "Triton Ragged (ours)": attn_triton_ragged,
    }

    # Test at different batch sizes and sparsity levels
    configs = [
        # (batch_size, prune_ratio, description)
        (4, 0.0, "BS=4, 0% pruned"),
        (4, 0.5, "BS=4, 50% pruned"),
        (4, 0.8, "BS=4, 80% pruned"),
        (16, 0.0, "BS=16, 0% pruned"),
        (16, 0.5, "BS=16, 50% pruned"),
        (16, 0.8, "BS=16, 80% pruned"),
        (32, 0.0, "BS=32, 0% pruned"),
        (32, 0.5, "BS=32, 50% pruned"),
        (32, 0.8, "BS=32, 80% pruned"),
    ]

    max_samples = max(bs for bs, _, _ in configs)
    val_data = get_imagenet_val(max_samples=max_samples)

    deit = load_deit()
    front, early, late_seq, _ = split_deit(deit)
    attn_block = list(late_seq)[0]

    results = {"configs": [], "kernels": {k: [] for k in kernels}}

    for bs, ratio, desc in configs:
        print(f"\n--- {desc} ---")
        results["configs"].append(desc)

        images = build_imagenet_batch(val_data, bs)
        with torch.inference_mode():
            x = front(images)
            x = early(x)
            mask = threshold_prune_mask(x, fixed_ratio=ratio)
            packed, cu_seqlens = triton_pack_tokens(x, mask)

            x_norm = attn_block.norm1(packed)
            qkv = attn_block.attn.qkv(x_norm).reshape(-1, 3, H, D)
            q = qkv[:, 0].contiguous()
            k = qkv[:, 1].contiguous()
            v = qkv[:, 2].contiguous()

        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int32)
        max_len = int(lengths.max().item())

        for kname, kfn in kernels.items():
            try:
                lat = benchmark_kernel(kfn, q, k, v, cu_seqlens, max_len)
                print(f"  {kname:30s}  {lat:.3f} ms")
            except Exception as e:
                lat = -1.0
                print(f"  {kname:30s}  FAILED: {e}")
            results["kernels"][kname].append(round(lat, 4))

        torch.cuda.empty_cache(); gc.collect()

    return results


# ─────────────────────────────────────────────────────────────────────
# Experiment B: End-to-end pipeline comparison
# ─────────────────────────────────────────────────────────────────────

def run_pipeline_comparison():
    """
    Compare full inference pipelines:
      1. Standard PyTorch padded (our existing baseline)
      2. NestedTensor SDPA pipeline (PyTorch-native ragged)
      3. FlashAttention-2 varlen pipeline (secondary baseline)
      4. Our Triton ragged pipeline
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT B: End-to-End Pipeline Comparison")
    print("=" * 60)


    from baselines.pytorch_pruned import build_pytorch_pruned_model
    from models.triton_ragged_deit import build_triton_ragged_model

    results = {"pipelines": {}}
    from config import BATCH_SIZES as batch_sizes

    dtype = get_dtype()
    max_samples = max(batch_sizes)
    val_data = get_imagenet_val(max_samples=max_samples)


    # 1. PyTorch Padded
    print("\n--- Threshold-L2 + PyTorch SDPA Padded ---")
    model = build_pytorch_pruned_model()
    tp_list = []
    for bs in batch_sizes:
        images = build_imagenet_batch(val_data, bs)
        for _ in range(WARMUP_ITERS):
            model(images, fixed_ratio=0.5)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(BENCH_ITERS):
            model(images, fixed_ratio=0.5)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        tp = (bs * BENCH_ITERS) / elapsed
        tp_list.append(round(tp, 1))
        print(f"  BS={bs:3d}  → {tp:.1f} img/s")
        torch.cuda.empty_cache()
    results["pipelines"]["PyTorch Padded (SDPA)"] = tp_list
    del model; torch.cuda.empty_cache(); gc.collect()

    # 2. Our Triton Ragged
    print("\n--- Threshold-L2 + Triton Ragged ---")
    model = build_triton_ragged_model()
    tp_list = []
    for bs in batch_sizes:
        images = build_imagenet_batch(val_data, bs)
        for _ in range(WARMUP_ITERS):
            model(images, fixed_ratio=0.5)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(BENCH_ITERS):
            model(images, fixed_ratio=0.5)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        tp = (bs * BENCH_ITERS) / elapsed
        tp_list.append(round(tp, 1))
        print(f"  BS={bs:3d}  → {tp:.1f} img/s")
        torch.cuda.empty_cache()
    results["pipelines"]["Triton Ragged (ours)"] = tp_list
    del model; torch.cuda.empty_cache(); gc.collect()

    # 3. NestedTensor SDPA Pipeline (build it here)
    print("\n--- Threshold-L2 + NestedTensor SDPA Pipeline ---")
    from models.deit_base import load_deit, split_deit
    from models.pruning import threshold_prune_mask
    from kernels.pack_tokens import triton_pack_tokens

    deit = load_deit()
    front, early, late_seq, back = split_deit(deit)
    late_blocks_list = list(late_seq)

    class NestedTensorBlock(nn.Module):
        """Transformer block using NestedTensor SDPA for attention."""
        def __init__(self, block):
            super().__init__()
            self.norm1 = block.norm1
            self.qkv_proj = block.attn.qkv
            self.out_proj = block.attn.proj
            self.attn_drop = block.attn.attn_drop
            self.proj_drop = block.attn.proj_drop
            self.num_heads = block.attn.num_heads
            self.head_dim = block.attn.head_dim
            self.norm2 = block.norm2
            self.mlp = block.mlp
            self.ls1 = block.ls1 if hasattr(block, "ls1") else nn.Identity()
            self.ls2 = block.ls2 if hasattr(block, "ls2") else nn.Identity()
            self.drop_path1 = block.drop_path1 if hasattr(block, "drop_path1") else nn.Identity()
            self.drop_path2 = block.drop_path2 if hasattr(block, "drop_path2") else nn.Identity()

        def forward(self, x, cu_seqlens):
            Total, D = x.shape
            H, d = self.num_heads, self.head_dim
            B = cu_seqlens.shape[0] - 1

            residual = x
            x_norm = self.norm1(x)
            qkv = self.qkv_proj(x_norm).reshape(Total, 3, H, d)
            q = qkv[:, 0].contiguous()
            k = qkv[:, 1].contiguous()
            v = qkv[:, 2].contiguous()

            # Split into per-image tensors for NestedTensor
            q_list = [q[cu_seqlens[i]:cu_seqlens[i+1]] for i in range(B)]
            k_list = [k[cu_seqlens[i]:cu_seqlens[i+1]] for i in range(B)]
            v_list = [v[cu_seqlens[i]:cu_seqlens[i+1]] for i in range(B)]

            q_nt = torch.nested.nested_tensor(q_list, layout=torch.jagged)
            k_nt = torch.nested.nested_tensor(k_list, layout=torch.jagged)
            v_nt = torch.nested.nested_tensor(v_list, layout=torch.jagged)

            q_nt = q_nt.transpose(1, 2)
            k_nt = k_nt.transpose(1, 2)
            v_nt = v_nt.transpose(1, 2)

            out_nt = F.scaled_dot_product_attention(q_nt, k_nt, v_nt)
            out_nt = out_nt.transpose(1, 2)

            # Reassemble flat tensor
            attn_out = torch.zeros(Total, H, d, device=x.device, dtype=x.dtype)
            for i in range(B):
                s, e = cu_seqlens[i].item(), cu_seqlens[i+1].item()
                attn_out[s:e] = out_nt[i]

            attn_out = attn_out.reshape(Total, D)
            attn_out = self.out_proj(attn_out)
            attn_out = self.proj_drop(attn_out)

            x = residual + self.drop_path1(self.ls1(attn_out))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
            return x

    class NestedTensorPipeline(nn.Module):
        def __init__(self):
            super().__init__()
            self.front = front
            self.early = early
            self.nt_blocks = nn.ModuleList([NestedTensorBlock(b) for b in late_blocks_list])
            self.back_norm = back.norm
            self.back_head = back.head

        @torch.inference_mode()
        def forward(self, images, fixed_ratio=None):
            B = images.shape[0]
            x = self.front(images)
            x = self.early(x)

            mask = threshold_prune_mask(x, fixed_ratio=fixed_ratio)
            packed, cu_seqlens = triton_pack_tokens(x, mask)

            for block in self.nt_blocks:
                packed = block(packed, cu_seqlens)

            cls_indices = cu_seqlens[:-1].long()
            cls_tokens = packed[cls_indices]
            cls_tokens = self.back_norm(cls_tokens)
            return self.back_head(cls_tokens)

    model = NestedTensorPipeline().to(DEVICE, dtype=get_dtype()).eval()
    tp_list = []
    for bs in batch_sizes:
        images = build_imagenet_batch(val_data, bs)
        try:
            for _ in range(WARMUP_ITERS):
                model(images, fixed_ratio=0.5)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(BENCH_ITERS):
                model(images, fixed_ratio=0.5)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            tp = (bs * BENCH_ITERS) / elapsed
        except Exception as e:
            tp = 0.0
            print(f"  BS={bs:3d}  → FAILED: {e}")
        tp_list.append(round(tp, 1))
        if tp > 0:
            print(f"  BS={bs:3d}  → {tp:.1f} img/s")
        torch.cuda.empty_cache()
    results["pipelines"]["NestedTensor SDPA"] = tp_list
    results["batch_sizes"] = batch_sizes
    del model; torch.cuda.empty_cache(); gc.collect()

    # 4. FlashAttention-2 varlen pipeline
    print("\n--- Threshold-L2 + FlashAttention-2 (varlen) Pipeline ---")
    try:
        from flash_attn import flash_attn_varlen_func

        class FA2Block(nn.Module):
            """Transformer block using FA2 varlen for attention."""
            def __init__(self, block):
                super().__init__()
                self.norm1 = block.norm1
                self.qkv_proj = block.attn.qkv
                self.out_proj = block.attn.proj
                self.proj_drop = block.attn.proj_drop
                self.num_heads = block.attn.num_heads
                self.head_dim = block.attn.head_dim
                self.norm2 = block.norm2
                self.mlp = block.mlp
                self.ls1 = block.ls1 if hasattr(block, "ls1") else nn.Identity()
                self.ls2 = block.ls2 if hasattr(block, "ls2") else nn.Identity()
                self.drop_path1 = block.drop_path1 if hasattr(block, "drop_path1") else nn.Identity()
                self.drop_path2 = block.drop_path2 if hasattr(block, "drop_path2") else nn.Identity()

            def forward(self, x, cu_seqlens, max_seqlen):
                Total, D = x.shape
                H, d = self.num_heads, self.head_dim

                residual = x
                x_norm = self.norm1(x)
                qkv = self.qkv_proj(x_norm).reshape(Total, 3, H, d)
                q = qkv[:, 0].contiguous()
                k = qkv[:, 1].contiguous()
                v = qkv[:, 2].contiguous()

                attn_out = flash_attn_varlen_func(
                    q, k, v,
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_k=max_seqlen,
                    causal=False,
                )

                attn_out = attn_out.reshape(Total, D)
                attn_out = self.out_proj(attn_out)
                attn_out = self.proj_drop(attn_out)

                x = residual + self.drop_path1(self.ls1(attn_out))
                x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
                return x

        deit2 = load_deit()
        front2, early2, late_seq2, back2 = split_deit(deit2)
        late_blocks_list2 = list(late_seq2)

        class FA2Pipeline(nn.Module):
            def __init__(self):
                super().__init__()
                self.front = front2
                self.early = early2
                self.fa2_blocks = nn.ModuleList([FA2Block(b) for b in late_blocks_list2])
                self.back_norm = back2.norm
                self.back_head = back2.head

            @torch.inference_mode()
            def forward(self, images, fixed_ratio=None):
                B = images.shape[0]
                x = self.front(images)
                x = self.early(x)

                mask = threshold_prune_mask(x, fixed_ratio=fixed_ratio)
                packed, cu_seqlens = triton_pack_tokens(x, mask)

                lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int32)
                max_seqlen = int(lengths.max().item())

                for block in self.fa2_blocks:
                    packed = block(packed, cu_seqlens, max_seqlen)

                cls_indices = cu_seqlens[:-1].long()
                cls_tokens = packed[cls_indices]
                cls_tokens = self.back_norm(cls_tokens)
                return self.back_head(cls_tokens)

        model = FA2Pipeline().to(DEVICE, dtype=get_dtype()).eval()
        tp_list = []
        for bs in batch_sizes:
            images = build_imagenet_batch(val_data, bs)
            try:
                for _ in range(WARMUP_ITERS):
                    model(images, fixed_ratio=0.5)
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(BENCH_ITERS):
                    model(images, fixed_ratio=0.5)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                tp = (bs * BENCH_ITERS) / elapsed
            except Exception as e:
                tp = 0.0
                print(f"  BS={bs:3d}  → FAILED: {e}")
            tp_list.append(round(tp, 1))
            if tp > 0:
                print(f"  BS={bs:3d}  → {tp:.1f} img/s")
            torch.cuda.empty_cache()
        results["pipelines"]["FlashAttention-2 (varlen)"] = tp_list
        del model; torch.cuda.empty_cache(); gc.collect()
    except ImportError:
        print("  ⚠ flash-attn not installed, skipping FA2 pipeline")
    except Exception as e:
        print(f"  ⚠ FA2 pipeline failed: {e}")

    return results


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def run_benchmark_6():
    all_results = {}

    all_results["kernel_microbenchmark"] = run_kernel_microbenchmark()
    all_results["pipeline_comparison"] = run_pipeline_comparison()

    os.makedirs("results", exist_ok=True)
    with open("results/bench6_sota_baselines.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\n✓ Results saved to results/bench6_sota_baselines.json")

    plot_benchmark_6(all_results)


def plot_benchmark_6(results=None):
    if results is None:
        with open("results/bench6_sota_baselines.json") as f:
            results = json.load(f)

    # ── Plot A: Kernel micro-benchmark grouped bar chart ─────────────
    micro = results.get("kernel_microbenchmark", {})
    if micro and "configs" in micro:
        configs = micro["configs"]
        kernels = micro["kernels"]

        # Filter to configs where all kernels have data
        n_configs = len(configs)
        kernel_names = list(kernels.keys())
        n_kernels = len(kernel_names)

        fig, ax = plt.subplots(figsize=(14, 6))
        bar_width = 0.8 / n_kernels
        x = range(n_configs)

        colors = ["#1f77b4", "#ff7f0e", "#9467bd", "#e377c2", "#2ca02c", "#d62728"]
        for ki, kname in enumerate(kernel_names):
            vals = kernels[kname]
            # Replace -1 with 0 for display
            vals_display = [max(0, v) for v in vals]
            positions = [xi + ki * bar_width for xi in x]
            bars = ax.bar(positions, vals_display, bar_width,
                          label=kname, color=colors[ki % len(colors)], alpha=0.85)

            # Annotate values
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f"{val:.2f}", ha="center", va="bottom", fontsize=6, rotation=45)

        ax.set_xticks([xi + bar_width * (n_kernels - 1) / 2 for xi in x])
        ax.set_xticklabels(configs, fontsize=7, rotation=30, ha="right")
        ax.set_ylabel("Latency (ms)", fontsize=11)
        ax.set_title("Attention Kernel Micro-benchmark (lower is better)",
                      fontsize=13, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        fig.savefig("results/bench6_kernel_microbench.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("✓ Saved results/bench6_kernel_microbench.png")

    # ── Plot B: Pipeline throughput comparison ───────────────────────
    pipe = results.get("pipeline_comparison", {})
    if pipe and "pipelines" in pipe:
        bs = pipe["batch_sizes"]
        pipelines = pipe["pipelines"]

        fig, ax = plt.subplots(figsize=(8, 5))

        styles = {
            "PyTorch Padded (SDPA)":       ("s--", "#d62728"),
            "NestedTensor SDPA":           ("^-",  "#ff7f0e"),
            "FlashAttention-2 (varlen)":   ("p-",  "#9467bd"),
            "Triton Ragged (ours)":        ("D-",  "#2ca02c"),
        }

        for pname, vals in pipelines.items():
            marker_line, color = styles.get(pname, ("o-", "gray"))
            if any(v > 0 for v in vals):
                ax.plot(bs, vals, marker_line, label=pname,
                        color=color, linewidth=2, markersize=8)

        ax.set_xlabel("Batch Size", fontsize=12)
        ax.set_ylabel("Throughput (img/s)", fontsize=12)
        ax.set_title("End-to-End Pipeline: SDPA vs FA2 vs NestedTensor vs Triton Ragged\n"
                      "(Threshold-L2, 50% prune)",
                      fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig("results/bench6_pipeline_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("✓ Saved results/bench6_pipeline_comparison.png")
