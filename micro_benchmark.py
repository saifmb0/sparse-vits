#!/usr/bin/env python3
"""
micro_benchmark.py
==================
RTX 4000 Ada Generation (SM89, Ada Lovelace) attention kernel micro-benchmark.

Eliminates Triton dispatch overhead via CUDA-event timing:
  - Each kernel is timed with pre-allocated CUDA events (GPU stream time only)
  - Triton JIT is forced to compile for every shape *before* any measurement
  - Synthetic ragged tensors — no model or dataset loading required
  - DeiT-B geometry: NUM_HEADS=12, HEAD_DIM=64, seqlens from post-prune range

Compared to bench6's kernel_microbenchmark:
  bench6   → CPU perf_counter + single sync → includes Python padding-loop time
             and per-call Triton dispatch overhead in the measured window.
  here     → CUDA events bracketing only GPU stream work → pure GPU execution
             time, making all kernels directly comparable.

Usage:
    python micro_benchmark.py
    python micro_benchmark.py --results-dir results/RTX4000Ada --iters 300
"""

import argparse
import gc
import json
import os
import statistics
import sys

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── DeiT-B constants ─────────────────────────────────────────────────────────
NUM_HEADS  = 12
HEAD_DIM   = 64
EMBED_DIM  = NUM_HEADS * HEAD_DIM   # 768
DEVICE     = "cuda"
DTYPE      = torch.float16

# ── Benchmark knobs ───────────────────────────────────────────────────────────
# Triton gets JIT_WARMUP + WARMUP iterations; all other kernels get WARMUP.
JIT_WARMUP  = 150   # compilation + JIT cache population passes (Triton only)
WARMUP      = 50    # GPU warm-up passes for all kernels
ITERS       = 200   # timed iterations — median is reported

# ── RTX 4000 Ada (SM89) Triton tile sizes ─────────────────────────────────────
# Ada Lovelace L1: 128 KB/SM.  With BLOCK_M=64, BLOCK_N=64, BLOCK_D=64 fp16:
#   Q  tile : 64 × 64 × 2 B =  8 KB
#   K  tile : 64 × 64 × 2 B =  8 KB
#   V  tile : 64 × 64 × 2 B =  8 KB
#   acc     : 64 × 64 × 4 B = 16 KB  (fp32)
#   total   ≈ 40 KB  — well within the 100 KB configurable limit.
BLOCK_M_ADA = 64
BLOCK_N_ADA = 64

# ── Test configurations ───────────────────────────────────────────────────────
# (batch_size, seq_len, label)
# seq_len approximates DeiT-B tokens remaining after threshold pruning:
#   0%  kept → 197 tokens (all patches + CLS)
#  50%  kept →  99 tokens
#  80%  kept →  40 tokens (aggressive pruning)
CONFIGS = [
    (32, 197, "BS=32, 0% pruned"),
    (32,  99, "BS=32, 50% pruned"),
    (32,  40, "BS=32, 80% pruned"),
    (64, 197, "BS=64, 0% pruned"),
    (64,  99, "BS=64, 50% pruned"),
    (64,  40, "BS=64, 80% pruned"),
]

# (batch_size, actual_seq_len, max_len, label)
# max_len == actual_seq_len for unpruned; max_len > actual_seq_len when
# simulating a pruned high-resolution model (padded kernels waste max_len
# tokens; ragged kernels operate on actual_seq_len only).
LARGE_SEQLEN_CONFIGS = [
    (32,  40, 197, "40 tok\n(80% pruned\n224²)"),
    (32,  99, 197, "99 tok\n(50% pruned\n224²)"),
    (32, 197, 197, "197 tok\n(full\n224²)"),
    (32, 288, 577, "288 tok\n(50% pruned\n384²)"),
    (32, 577, 577, "577 tok\n(full\n384²)"),
    (32, 785, 785, "785 tok\n(full\n448²)"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Triton Ragged Attention Kernel (SM89 / Ada Lovelace variant)
# ─────────────────────────────────────────────────────────────────────────────
# Bidirectional FlashAttention-2 style kernel for variable-length ViT tokens.
# Grid: one program per (image, head) pair.

@triton.jit
def _ragged_attn_fwd_sm89(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    cu_seqlens_ptr,
    num_heads:   tl.constexpr,
    head_dim:    tl.constexpr,
    stride_tok:  tl.constexpr,
    stride_head: tl.constexpr,
    BLOCK_M:     tl.constexpr,
    BLOCK_N:     tl.constexpr,
    BLOCK_D:     tl.constexpr,
):
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

        q_ptrs = (Q_ptr
                  + q_tok_ids[:, None] * stride_tok
                  + head_idx * stride_head
                  + d_offs[None, :])
        q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
        m_i = tl.full([BLOCK_M], value=-1e9, dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

        for n_start in range(0, seq_len, BLOCK_N):
            n_offs     = n_start + tl.arange(0, BLOCK_N)
            n_mask     = n_offs < seq_len
            kv_tok_ids = seq_start + tl.minimum(n_offs, seq_len - 1)

            k_ptrs = (K_ptr
                      + kv_tok_ids[:, None] * stride_tok
                      + head_idx * stride_head
                      + d_offs[None, :])
            k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)

            s = tl.dot(q, tl.trans(k)) * sm_scale
            s = tl.where(m_mask[:, None] & n_mask[None, :], s, -1e9)

            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp(m_i - m_new)
            p     = tl.exp(s - m_new[:, None])
            l_new = alpha * l_i + tl.sum(p, axis=1)

            v_ptrs = (V_ptr
                      + kv_tok_ids[:, None] * stride_tok
                      + head_idx * stride_head
                      + d_offs[None, :])
            v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)
            m_i = m_new
            l_i = l_new

        acc   = acc / l_i[:, None]
        o_ptrs = (O_ptr
                  + q_tok_ids[:, None] * stride_tok
                  + head_idx * stride_head
                  + d_offs[None, :])
        tl.store(o_ptrs, acc.to(tl.float16), mask=m_mask[:, None])


def triton_ragged_attn_ada(q, k, v, cu_seqlens):
    """
    Ragged attention for RTX 4000 Ada (SM89).
    q/k/v : [Total, num_heads, head_dim] fp16
    cu_seqlens : [B+1] int32
    Returns output : [Total, num_heads, head_dim] fp16
    """
    Total, num_heads, head_dim = q.shape
    B    = cu_seqlens.shape[0] - 1
    out  = torch.empty_like(q)
    BLOCK_D = triton.next_power_of_2(head_dim)

    _ragged_attn_fwd_sm89[(B * num_heads,)](
        q, k, v, out,
        cu_seqlens,
        num_heads   = num_heads,
        head_dim    = head_dim,
        stride_tok  = q.stride(0),
        stride_head = q.stride(1),
        BLOCK_M     = BLOCK_M_ADA,
        BLOCK_N     = BLOCK_N_ADA,
        BLOCK_D     = BLOCK_D,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Baseline kernel wrappers
# (Same API as bench6: all take flat q/k/v + cu_seqlens + max_len)
# ─────────────────────────────────────────────────────────────────────────────

def _unpack_padded(out_pad, cu_seqlens):
    """Scatter padded [B, S, H, D] back to flat [Total, H, D]."""
    B = cu_seqlens.shape[0] - 1
    result = torch.empty(
        cu_seqlens[-1].item(), out_pad.shape[2], out_pad.shape[3],
        device=out_pad.device, dtype=out_pad.dtype)
    for i in range(B):
        s, e = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        result[s:e] = out_pad[i, :e - s]
    return result


def _build_padded(q, k, v, cu_seqlens, max_len):
    """Pack flat [Total, H, D] → padded [B, max_len, H, D] tensors."""
    B = cu_seqlens.shape[0] - 1
    H, D = q.shape[1], q.shape[2]
    q_pad = q.new_zeros(B, max_len, H, D)
    k_pad = q.new_zeros(B, max_len, H, D)
    v_pad = q.new_zeros(B, max_len, H, D)
    mask  = torch.zeros(B, max_len, device=q.device, dtype=torch.bool)
    for i in range(B):
        s, e = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        n = e - s
        q_pad[i, :n] = q[s:e]
        k_pad[i, :n] = k[s:e]
        v_pad[i, :n] = v[s:e]
        mask[i, :n]  = True
    return q_pad, k_pad, v_pad, mask


def attn_sdpa_math(q, k, v, cu_seqlens, max_len):
    q_pad, k_pad, v_pad, mask = _build_padded(q, k, v, cu_seqlens, max_len)
    q_pad = q_pad.transpose(1, 2)   # [B, H, S, D]
    k_pad = k_pad.transpose(1, 2)
    v_pad = v_pad.transpose(1, 2)
    attn_mask = q.new_zeros(q_pad.shape[0], 1, 1, max_len)
    attn_mask.masked_fill_(~mask.unsqueeze(1).unsqueeze(2), float("-inf"))
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out = F.scaled_dot_product_attention(q_pad, k_pad, v_pad, attn_mask=attn_mask)
    return _unpack_padded(out.transpose(1, 2), cu_seqlens)


def attn_sdpa_efficient(q, k, v, cu_seqlens, max_len):
    q_pad, k_pad, v_pad, mask = _build_padded(q, k, v, cu_seqlens, max_len)
    q_pad = q_pad.transpose(1, 2)
    k_pad = k_pad.transpose(1, 2)
    v_pad = v_pad.transpose(1, 2)
    attn_mask = q.new_zeros(q_pad.shape[0], 1, 1, max_len)
    attn_mask.masked_fill_(~mask.unsqueeze(1).unsqueeze(2), float("-inf"))
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION):
        out = F.scaled_dot_product_attention(q_pad, k_pad, v_pad, attn_mask=attn_mask)
    return _unpack_padded(out.transpose(1, 2), cu_seqlens)


def attn_sdpa_flash(q, k, v, cu_seqlens, max_len):
    """PyTorch SDPA with FlashAttention backend (SM89 supported).
    Flash backend does not accept an arbitrary attn_mask, so we omit it
    and unpack only valid tokens — padding rows are discarded anyway."""
    q_pad, k_pad, v_pad, _ = _build_padded(q, k, v, cu_seqlens, max_len)
    q_pad = q_pad.transpose(1, 2)
    k_pad = k_pad.transpose(1, 2)
    v_pad = v_pad.transpose(1, 2)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
        out = F.scaled_dot_product_attention(q_pad, k_pad, v_pad)
    return _unpack_padded(out.transpose(1, 2), cu_seqlens)


def attn_nested_tensor(q, k, v, cu_seqlens, max_len):
    B = cu_seqlens.shape[0] - 1
    q_list = [q[cu_seqlens[i]:cu_seqlens[i + 1]] for i in range(B)]
    k_list = [k[cu_seqlens[i]:cu_seqlens[i + 1]] for i in range(B)]
    v_list = [v[cu_seqlens[i]:cu_seqlens[i + 1]] for i in range(B)]
    q_nt = torch.nested.nested_tensor(q_list, layout=torch.jagged).transpose(1, 2)
    k_nt = torch.nested.nested_tensor(k_list, layout=torch.jagged).transpose(1, 2)
    v_nt = torch.nested.nested_tensor(v_list, layout=torch.jagged).transpose(1, 2)
    out_nt = F.scaled_dot_product_attention(q_nt, k_nt, v_nt).transpose(1, 2)
    result = torch.zeros_like(q)
    for i in range(B):
        s, e = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        result[s:e] = out_nt[i]
    return result


def attn_triton_ragged(q, k, v, cu_seqlens, max_len):
    return triton_ragged_attn_ada(q, k, v, cu_seqlens)


try:
    from flash_attn import flash_attn_varlen_func as _fa2_fn

    def attn_fa2_varlen(q, k, v, cu_seqlens, max_len):
        return _fa2_fn(
            q, k, v,
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_len, max_seqlen_k=max_len,
            causal=False,
        )
    _FA2_AVAILABLE = True
except ImportError:
    _FA2_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Timing utilities
# ─────────────────────────────────────────────────────────────────────────────

def _build_synthetic(batch_size, seq_len, max_len=None):
    """Build synthetic ragged batch with uniform seq_len for reproducibility.

    max_len: padded length for padded kernels (defaults to seq_len).
    When max_len > seq_len (e.g. simulating 50%-pruned 384×384 where
    actual=288 but the padded pipeline allocates a 577-token buffer),
    ragged kernels still process only seq_len tokens while padded
    kernels pay the full max_len allocation cost.
    """
    if max_len is None:
        max_len = seq_len
    total = batch_size * seq_len
    q = torch.randn(total, NUM_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    cu_seqlens = torch.arange(0, (batch_size + 1) * seq_len, seq_len,
                               device=DEVICE, dtype=torch.int32)
    return q, k, v, cu_seqlens, max_len


def cuda_event_bench(fn, *args, warmup=WARMUP, iters=ITERS):
    """
    Measures GPU-stream execution time of fn(*args) using pre-allocated
    CUDA events.  Returns (median_ms, p5_ms, p95_ms).

    CUDA events record timestamps directly in the GPU stream, so CPU-side
    Python overhead that precedes CUDA work (e.g. padding loops that touch
    only CPU tensors) does NOT inflate the measured time.  This gives a
    fair, dispatch-overhead-free comparison across all kernel types.
    """
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        starts[i].record()
        fn(*args)
        ends[i].record()

    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))

    lo  = int(0.05 * iters)
    hi  = int(0.95 * iters)
    return statistics.median(times), times[lo], times[hi - 1]


def triton_jit_warmup(q, k, v, cu_seqlens, label=""):
    """
    Force Triton to compile and cache the kernel for this exact tensor shape
    before any timed measurement.  Without this, the first few timing windows
    include JIT overhead, biasing the measured latency upward.
    """
    print(f"  [Triton JIT warmup{' (' + label + ')' if label else ''}...]",
          end=" ", flush=True)
    for _ in range(JIT_WARMUP):
        triton_ragged_attn_ada(q, k, v, cu_seqlens)
    torch.cuda.synchronize()
    print("done", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel registry
# ─────────────────────────────────────────────────────────────────────────────

def build_kernel_registry():
    reg = {
        "SDPA (efficient, padded)": attn_sdpa_efficient,
        "SDPA (flash, padded)":     attn_sdpa_flash,
        "Triton Ragged (ours)":     attn_triton_ragged,
    }
    if _FA2_AVAILABLE:
        reg["FlashAttention-2 (varlen)"] = attn_fa2_varlen
    return reg


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark
# ─────────────────────────────────────────────────────────────────────────────

def run_microbenchmark(iters=ITERS, results_dir="results"):
    print("\n" + "=" * 70)
    print("RTX 4000 Ada — Attention Kernel Micro-benchmark (CUDA-event timing)")
    print("=" * 70)
    print(f"GPU  : {torch.cuda.get_device_name(0)}  (SM{torch.cuda.get_device_capability(0)[0]}"
          f"{torch.cuda.get_device_capability(0)[1]})")
    print(f"VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Timing: {iters} iterations, median reported (p5–p95 range)")
    print(f"Note : CUDA events measure GPU stream time only — Triton dispatch")
    print(f"       overhead is excluded.  Triton JIT warmup: {JIT_WARMUP} passes.\n")

    kernels = build_kernel_registry()

    results = {
        "gpu":         torch.cuda.get_device_name(0),
        "sm":          f"SM{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}",
        "timing":      "cuda_events_median_ms",
        "warmup":      WARMUP,
        "jit_warmup":  JIT_WARMUP,
        "iters":       iters,
        "configs":     [],
        "kernels":     {k: [] for k in kernels},
        "p5":          {k: [] for k in kernels},
        "p95":         {k: [] for k in kernels},
    }

    # Pre-warm Triton JIT for all shapes that will be tested
    print("Phase 1: Triton JIT compilation for all test shapes")
    print("-" * 50)
    seen_shapes = set()
    for bs, seq_len, _ in CONFIGS:
        shape_key = (bs, seq_len)
        if shape_key in seen_shapes:
            continue
        seen_shapes.add(shape_key)
        q, k, v, cu_seqlens, max_len = _build_synthetic(bs, seq_len)
        triton_jit_warmup(q, k, v, cu_seqlens, label=f"BS={bs}, L={seq_len}")
        del q, k, v, cu_seqlens
        torch.cuda.empty_cache()

    print("\nPhase 2: Timed measurement")
    print("-" * 50)
    for bs, seq_len, label in CONFIGS:
        print(f"\n[{label}]  total_tokens={bs * seq_len}")
        results["configs"].append(label)
        q, k, v, cu_seqlens, max_len = _build_synthetic(bs, seq_len)

        for kname, kfn in kernels.items():
            try:
                med, p5, p95 = cuda_event_bench(
                    kfn, q, k, v, cu_seqlens, max_len,
                    warmup=WARMUP, iters=iters,
                )
                print(f"  {kname:32s}  {med:7.4f} ms  [{p5:.4f}–{p95:.4f}]")
                results["kernels"][kname].append(round(med,  5))
                results["p5"][kname].append(round(p5,  5))
                results["p95"][kname].append(round(p95, 5))
            except Exception as exc:
                print(f"  {kname:32s}  FAILED: {exc}")
                results["kernels"][kname].append(-1.0)
                results["p5"][kname].append(-1.0)
                results["p95"][kname].append(-1.0)

        del q, k, v, cu_seqlens
        torch.cuda.empty_cache()
        gc.collect()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

_COLORS = {
    "SDPA (math, padded)":      "#1f77b4",
    "SDPA (efficient, padded)": "#ff7f0e",
    "SDPA (flash, padded)":     "#9467bd",
    "FlashAttention-2 (varlen)":"#e377c2",
    "NestedTensor + SDPA":      "#bcbd22",
    "Triton Ragged (ours)":     "#2ca02c",
}


def plot_results(results, results_dir):
    configs     = results["configs"]
    kernels_all = results["kernels"]
    n_configs   = len(configs)
    n_kernels   = len(kernels_all)

    fig, ax = plt.subplots(figsize=(max(14, n_configs * 2.2), 6))
    bar_w   = 0.8 / n_kernels
    x       = list(range(n_configs))

    for ki, (kname, vals) in enumerate(kernels_all.items()):
        color    = _COLORS.get(kname, f"C{ki}")
        p5_vals  = results["p5"][kname]
        p95_vals = results["p95"][kname]
        positions = [xi + ki * bar_w for xi in x]

        display  = [max(0.0, v) for v in vals]
        errs_lo  = [max(0.0, v - p) for v, p in zip(display, p5_vals)]
        errs_hi  = [max(0.0, p - v) for v, p in zip(display, p95_vals)]

        bars = ax.bar(positions, display, bar_w,
                      label=kname, color=color, alpha=0.85,
                      yerr=[errs_lo, errs_hi], capsize=3, error_kw={"elinewidth": 1})

        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(errs_hi) * 0.05 + 0.005,
                        f"{val:.3f}", ha="center", va="bottom",
                        fontsize=6, rotation=45)

    ax.set_xticks([xi + bar_w * (n_kernels - 1) / 2 for xi in x])
    ax.set_xticklabels(configs, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("GPU latency (ms, lower is better)", fontsize=11)
    ax.set_title(
        "Attention Kernel Micro-benchmark — RTX 4000 Ada (SM89)\n"
        "CUDA-event timing (dispatch overhead excluded), DeiT-B geometry",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "micro_benchmark.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✓  Saved {out_path}")

    # ── Speedup table (relative to FlashAttention-2 varlen) ─────────────
    ref_key = "FlashAttention-2 (varlen)"
    if ref_key in kernels_all and any(v > 0 for v in kernels_all[ref_key]):
        print(f"\nSpeedup relative to '{ref_key}':")
        ref_vals = kernels_all[ref_key]
        for kname, vals in kernels_all.items():
            if kname == ref_key:
                continue
            speedups = []
            for ref, v in zip(ref_vals, vals):
                if ref > 0 and v > 0:
                    speedups.append(f"{ref / v:.2f}×")
                else:
                    speedups.append("  N/A")
            print(f"  {kname:32s}  {' | '.join(speedups)}")


# ─────────────────────────────────────────────────────────────────────────────
# Large-seqlen sweep (high-resolution / extended range)
# ─────────────────────────────────────────────────────────────────────────────
# Compares only the three fast kernels across token counts from 40 → 785.
# Configs include both unpruned sequences (max_len == seq_len) and pruned
# high-resolution simulations (seq_len < max_len) to expose the extra
# padding cost that padded pipelines pay at 384×384 input.

_LS_KERNELS = ["SDPA (flash, padded)", "FlashAttention-2 (varlen)", "Triton Ragged (ours)"]


def run_large_seqlen(iters=ITERS):
    print("\n" + "=" * 70)
    print("Large-seqlen sweep — fast kernels only (BS=32, CUDA-event timing)")
    print("=" * 70)

    all_kernels = build_kernel_registry()
    kernels = {k: v for k, v in all_kernels.items() if k in _LS_KERNELS}

    results = {
        "gpu":     torch.cuda.get_device_name(0),
        "iters":   iters,
        "configs": [],
        "actual_seq_lens": [],
        "max_lens": [],
        "kernels": {k: [] for k in kernels},
        "p5":      {k: [] for k in kernels},
        "p95":     {k: [] for k in kernels},
    }

    print("Phase 1: Triton JIT compilation for all shapes")
    print("-" * 50)
    seen = set()
    for bs, seq_len, max_len, _ in LARGE_SEQLEN_CONFIGS:
        key = (bs, seq_len)
        if key in seen:
            continue
        seen.add(key)
        q, k, v, cu, _ = _build_synthetic(bs, seq_len)
        triton_jit_warmup(q, k, v, cu, label=f"BS={bs}, L={seq_len}")
        del q, k, v, cu
        torch.cuda.empty_cache()

    print("\nPhase 2: Timed measurement")
    print("-" * 50)
    for bs, seq_len, max_len, label in LARGE_SEQLEN_CONFIGS:
        print(f"\n[{label.replace(chr(10), ' ')}]  actual_tokens={bs * seq_len}, padded_tokens={bs * max_len}")
        results["configs"].append(label)
        results["actual_seq_lens"].append(seq_len)
        results["max_lens"].append(max_len)

        q, k, v, cu_seqlens, _ = _build_synthetic(bs, seq_len, max_len=max_len)

        for kname, kfn in kernels.items():
            try:
                med, p5, p95 = cuda_event_bench(
                    kfn, q, k, v, cu_seqlens, max_len,
                    warmup=WARMUP, iters=iters,
                )
                print(f"  {kname:36s}  {med:7.4f} ms  [{p5:.4f}–{p95:.4f}]")
                results["kernels"][kname].append(round(med,  5))
                results["p5"][kname].append(round(p5,  5))
                results["p95"][kname].append(round(p95, 5))
            except Exception as exc:
                print(f"  {kname:36s}  FAILED: {exc}")
                results["kernels"][kname].append(-1.0)
                results["p5"][kname].append(-1.0)
                results["p95"][kname].append(-1.0)

        del q, k, v, cu_seqlens
        torch.cuda.empty_cache()
        gc.collect()

    return results


def plot_large_seqlen(results, results_dir):
    configs  = results["configs"]
    kernels  = results["kernels"]
    p5_all   = results["p5"]
    p95_all  = results["p95"]
    n        = len(configs)

    _LS_COLORS = {
        "SDPA (flash, padded)":       "#9467bd",
        "FlashAttention-2 (varlen)":  "#e377c2",
        "Triton Ragged (ours)":       "#2ca02c",
    }
    _LS_MARKERS = {
        "SDPA (flash, padded)":       "s",
        "FlashAttention-2 (varlen)":  "^",
        "Triton Ragged (ours)":       "o",
    }

    fig, (ax_lat, ax_spd) = plt.subplots(1, 2, figsize=(16, 6))

    x = list(range(n))
    for kname, vals in kernels.items():
        color  = _LS_COLORS.get(kname, "C0")
        marker = _LS_MARKERS.get(kname, "o")
        p5v    = p5_all[kname]
        p95v   = p95_all[kname]
        valid  = [(i, v, p5v[i], p95v[i]) for i, v in enumerate(vals) if v > 0]
        if not valid:
            continue
        xi, yi, lo, hi = zip(*valid)
        errs_lo = [v - l for v, l in zip(yi, lo)]
        errs_hi = [h - v for v, h in zip(yi, hi)]
        ax_lat.errorbar(xi, yi, yerr=[errs_lo, errs_hi],
                        label=kname, color=color, marker=marker,
                        linewidth=1.8, markersize=7, capsize=4)

    ax_lat.set_xticks(x)
    ax_lat.set_xticklabels(configs, fontsize=8)
    ax_lat.set_ylabel("GPU latency (ms, lower is better)", fontsize=11)
    ax_lat.set_title("Latency vs Sequence Length (BS=32)", fontsize=12, fontweight="bold")
    ax_lat.legend(fontsize=9)
    ax_lat.grid(True, alpha=0.3)

    # Speedup of Triton over FA2 varlen (primary comparator)
    ref_key    = "FlashAttention-2 (varlen)"
    triton_key = "Triton Ragged (ours)"
    if ref_key in kernels and triton_key in kernels:
        ref_vals = kernels[ref_key]
        tri_vals = kernels[triton_key]
        speedups = []
        valid_x  = []
        for i, (r, t) in enumerate(zip(ref_vals, tri_vals)):
            if r > 0 and t > 0:
                speedups.append(r / t)
                valid_x.append(i)
        if speedups:
            bar_colors = ["#2ca02c" if s >= 1.0 else "#d62728" for s in speedups]
            bars = ax_spd.bar(valid_x, speedups, color=bar_colors, alpha=0.8, width=0.6)
            ax_spd.axhline(1.0, color="gray", linestyle="--", linewidth=1.2,
                           label="break-even")
            for bar, spd in zip(bars, speedups):
                ax_spd.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01,
                            f"{spd:.2f}×", ha="center", va="bottom", fontsize=9)
            ax_spd.set_xticks(valid_x)
            ax_spd.set_xticklabels([configs[i] for i in valid_x], fontsize=8)
            ax_spd.set_ylabel("Triton / FA2 varlen  (>1 = Triton faster)", fontsize=11)
            ax_spd.set_title("Triton vs FA2 varlen — Kernel Speedup (BS=32)",
                             fontsize=12, fontweight="bold")
            ax_spd.legend(fontsize=9)
            ax_spd.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "large_seqlen.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✓  Saved {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results/RTX4000Ada",
                        help="directory for JSON + PNG output")
    parser.add_argument("--iters", type=int, default=ITERS,
                        help="number of timed iterations per kernel")
    parser.add_argument("--mode", choices=["standard", "large-seqlen"], default="standard",
                        help="standard: original 6-config benchmark; "
                             "large-seqlen: extended range 40→785 tokens, fast kernels only")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA not available")

    os.makedirs(args.results_dir, exist_ok=True)

    if args.mode == "large-seqlen":
        results = run_large_seqlen(iters=args.iters)
        json_path = os.path.join(args.results_dir, "large_seqlen.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✓  Saved {json_path}")
        plot_large_seqlen(results, args.results_dir)
    else:
        results = run_microbenchmark(iters=args.iters, results_dir=args.results_dir)
        json_path = os.path.join(args.results_dir, "micro_benchmark.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✓  Saved {json_path}")
        plot_results(results, args.results_dir)


if __name__ == "__main__":
    main()
