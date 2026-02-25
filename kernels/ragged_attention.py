"""
Triton Kernel 2 — Ragged (Variable-Length) Self-Attention
==========================================================
Computes multi-head self-attention on a **flat packed** tensor guided
by `cu_seqlens`, with zero padding and zero wasted FLOPs.

Input layout (after packing):
    qkv : [Total_Tokens, 3, num_heads, head_dim]   (fp16)
    cu_seqlens : [B+1]  (int32)

For each image i the tokens live at rows cu_seqlens[i] .. cu_seqlens[i+1]-1.

This is a simplified, **bidirectional** (no causal mask) FlashAttention-2
variant stripped of KV-cache and causal logic — ideal for ViTs.

Strategy
--------
* Grid dim-0  → one program per (image, head) pair.
* Each program loads its own Q/K/V slice, does the full NxN attention
  in tiled SRAM blocks, and writes the output.
* For the small sequence lengths typical after pruning (20-197 tokens)
  we can often fit everything in SRAM in a single pass.
"""

import math
import torch
import triton
import triton.language as tl


# ── Triton kernel ────────────────────────────────────────────────────────

@triton.jit
def _ragged_attention_fwd(
    # Pointers
    Q_ptr, K_ptr, V_ptr, O_ptr,
    cu_seqlens_ptr,
    # Dimensions
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    stride_tok: tl.constexpr,     # stride between consecutive tokens in Q/K/V/O
    stride_head: tl.constexpr,    # stride between heads within one token
    # Tile sizes
    BLOCK_M: tl.constexpr,       # tile rows (query block)
    BLOCK_N: tl.constexpr,       # tile cols (key block)
    BLOCK_D: tl.constexpr,       # = head_dim (must be power of 2)
):
    # ── identify which image and head ────────────────────────────────
    pid = tl.program_id(0)
    head_idx = pid % num_heads
    img_idx  = pid // num_heads

    # ── sequence bounds for this image ───────────────────────────────
    seq_start = tl.load(cu_seqlens_ptr + img_idx).to(tl.int32)
    seq_end   = tl.load(cu_seqlens_ptr + img_idx + 1).to(tl.int32)
    seq_len   = seq_end - seq_start

    if seq_len <= 0:
        return

    # Scaling factor
    sm_scale = 1.0 / tl.sqrt(tl.cast(BLOCK_D, tl.float32))

    # ── iterate over query tiles ─────────────────────────────────────
    for m_start in range(0, seq_len, BLOCK_M):
        m_offs = m_start + tl.arange(0, BLOCK_M)           # [BLOCK_M]
        m_mask = m_offs < seq_len                           # [BLOCK_M]

        # Global token indices — CLAMPED to avoid OOB address faults
        q_tok_ids = seq_start + tl.minimum(m_offs, seq_len - 1)

        # Load Q tile:  [BLOCK_M, BLOCK_D]
        d_offs = tl.arange(0, BLOCK_D)                      # [BLOCK_D]
        q_ptrs = (Q_ptr
                  + q_tok_ids[:, None] * stride_tok
                  + head_idx * stride_head
                  + d_offs[None, :])
        q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)   # [BLOCK_M, BLOCK_D]

        # Accumulators
        acc   = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32) # O accumulator
        m_i   = tl.full([BLOCK_M], value=-1e9, dtype=tl.float32)  # running max
        l_i   = tl.zeros([BLOCK_M], dtype=tl.float32)         # running sum(exp)

        # ── iterate over key/value tiles ─────────────────────────────
        for n_start in range(0, seq_len, BLOCK_N):
            n_offs = n_start + tl.arange(0, BLOCK_N)        # [BLOCK_N]
            n_mask = n_offs < seq_len                        # [BLOCK_N]

            # CLAMPED to avoid OOB address faults
            kv_tok_ids = seq_start + tl.minimum(n_offs, seq_len - 1)

            # Load K tile: [BLOCK_N, BLOCK_D]
            k_ptrs = (K_ptr
                      + kv_tok_ids[:, None] * stride_tok
                      + head_idx * stride_head
                      + d_offs[None, :])
            k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)

            # S = Q @ K^T : [BLOCK_M, BLOCK_N]
            s = tl.dot(q, tl.trans(k)) * sm_scale

            # mask out invalid positions
            s = tl.where(m_mask[:, None] & n_mask[None, :], s, -1e9)

            # ── online softmax (FlashAttention style) ────────────────
            m_new = tl.maximum(m_i, tl.max(s, axis=1))       # [BLOCK_M]
            alpha = tl.exp(m_i - m_new)                       # rescale old
            p = tl.exp(s - m_new[:, None])                    # [BLOCK_M, BLOCK_N]

            l_new = alpha * l_i + tl.sum(p, axis=1)           # [BLOCK_M]

            # Load V tile: [BLOCK_N, BLOCK_D]
            v_ptrs = (V_ptr
                      + kv_tok_ids[:, None] * stride_tok
                      + head_idx * stride_head
                      + d_offs[None, :])
            v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

            # Update accumulator
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)

            m_i = m_new
            l_i = l_new

        # ── normalise and store ──────────────────────────────────────
        acc = acc / l_i[:, None]
        # Use clamped tok_ids for output store too
        o_ptrs = (O_ptr
                  + q_tok_ids[:, None] * stride_tok
                  + head_idx * stride_head
                  + d_offs[None, :])
        tl.store(o_ptrs, acc.to(tl.float16), mask=m_mask[:, None])


# ── Python wrapper ───────────────────────────────────────────────────────

def triton_ragged_attention(
    q: torch.Tensor,          # [Total, num_heads, head_dim]
    k: torch.Tensor,          # [Total, num_heads, head_dim]
    v: torch.Tensor,          # [Total, num_heads, head_dim]
    cu_seqlens: torch.Tensor, # [B+1] int32
    block_m: int = 64,        # query tile size  (must be power of 2, >= 16)
    block_n: int = 16,        # key/value tile size (must be power of 2, >= 16)
) -> torch.Tensor:
    """
    Computes bidirectional self-attention for a ragged batch.
    Returns output: [Total, num_heads, head_dim]  fp16

    Args:
        q, k, v    : packed token tensors [Total, num_heads, head_dim]
        cu_seqlens : cumulative sequence lengths [B+1] int32
        block_m    : SRAM tile size for query dimension (constexpr at JIT compile)
        block_n    : SRAM tile size for key/value dimension (constexpr at JIT compile)
    """
    assert block_m & (block_m - 1) == 0 and block_m >= 16, \
        f"block_m must be a power of 2 >= 16, got {block_m}"
    assert block_n & (block_n - 1) == 0 and block_n >= 16, \
        f"block_n must be a power of 2 >= 16, got {block_n}"

    Total, num_heads, head_dim = q.shape
    B = cu_seqlens.shape[0] - 1

    o = torch.empty_like(q)

    # Strides (in elements)
    stride_tok  = q.stride(0)     # num_heads * head_dim
    stride_head = q.stride(1)     # head_dim

    BLOCK_D = triton.next_power_of_2(head_dim)

    grid = (B * num_heads,)

    _ragged_attention_fwd[grid](
        q, k, v, o,
        cu_seqlens,
        num_heads=num_heads,
        head_dim=head_dim,
        stride_tok=stride_tok,
        stride_head=stride_head,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=BLOCK_D,
    )

    return o
