"""
Triton Kernel 1 — Fused Token Packer
=====================================
Takes:
    tokens  : [B, S, D]    (fp16)
    mask    : [B, S]        (bool / int8)

Produces:
    packed  : [Total_Kept, D]   contiguous 1-D tensor
    cu_seqlens : [B+1]          cumulative sequence lengths (int32)

Strategy
--------
1. PyTorch pre-pass on GPU: compute per-image keep counts, cu_seqlens,
   and per-token destination indices via cumsum.  This is a handful of
   fast GPU ops (no CPU sync except one scalar read for total_kept).
2. Triton kernel: each program copies one kept token (D elements) from
   its source position to its pre-computed destination in `packed`.
   The copy is fully vectorised with BLOCK_D-wide loads/stores.
"""

import torch
import triton
import triton.language as tl


# ── Triton kernel ────────────────────────────────────────────────────────

@triton.jit
def _pack_copy_kernel(
    # Pointers
    tokens_ptr,      # [B*S, D] fp16  (viewed flat)
    packed_ptr,      # [Total_Kept, D] fp16  (output)
    src_idx_ptr,     # [Total_Kept] int64 — flat source row index
    # Dimensions
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Grid: (Total_Kept,)
    Each program copies one token row from src to dst.
    """
    pid = tl.program_id(0)  # which kept-token this program handles

    src_row = tl.load(src_idx_ptr + pid).to(tl.int64)

    src_base = src_row * D
    dst_base = pid * D

    for d_start in range(0, D, BLOCK_D):
        cols = d_start + tl.arange(0, BLOCK_D)
        col_mask = cols < D
        vals = tl.load(tokens_ptr + src_base + cols, mask=col_mask, other=0.0)
        tl.store(packed_ptr + dst_base + cols, vals, mask=col_mask)


# ── Python wrapper ───────────────────────────────────────────────────────

def triton_pack_tokens(
    tokens: torch.Tensor,   # [B, S, D] fp16
    mask: torch.Tensor,     # [B, S]    bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        packed     : [Total_Kept, D]  fp16 contiguous
        cu_seqlens : [B+1]            int32
    """
    B, S, D = tokens.shape
    device = tokens.device

    # ── GPU-side index computation (no serial scan in kernel) ────────
    mask_i32 = mask.to(torch.int32)                           # [B, S]
    counts = mask_i32.sum(dim=1)                              # [B]
    cu_seqlens = torch.zeros(B + 1, dtype=torch.int32, device=device)
    torch.cumsum(counts, dim=0, out=cu_seqlens[1:])
    total_kept = int(cu_seqlens[-1].item())

    # Flat source indices of all kept tokens
    # mask_flat: [B*S], nonzero gives the flat positions
    mask_flat = mask.reshape(-1)                              # [B*S]
    src_indices = mask_flat.nonzero(as_tuple=False).squeeze(1) # [Total_Kept]

    # ── allocate packed output ───────────────────────────────────────
    packed = torch.empty(total_kept, D, dtype=tokens.dtype, device=device)

    if total_kept == 0:
        return packed, cu_seqlens

    # ── launch Triton copy kernel ────────────────────────────────────
    BLOCK_D = triton.next_power_of_2(D)
    grid = (total_kept,)
    _pack_copy_kernel[grid](
        tokens.reshape(-1, D),   # flat view [B*S, D]
        packed,
        src_indices,
        D=D,
        BLOCK_D=BLOCK_D,
    )

    return packed, cu_seqlens
