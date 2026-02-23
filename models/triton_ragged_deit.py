"""
Triton Ragged-Batch DeiT-Small — our method.

After layer PRUNE_AFTER_LAYER we:
  1. Compute a keep-mask via threshold pruning
  2. Pack kept tokens into a flat [Total_Kept, D] tensor via Triton packer
  3. Run all late transformer layers using Triton ragged attention
  4. Extract CLS token for classification

Zero padding.  Zero wasted FLOPs.  Linear scaling with token count.
"""

import torch
import torch.nn as nn

from models.deit_base import load_deit, split_deit, get_dtype
from models.pruning import threshold_prune_mask
from kernels.pack_tokens import triton_pack_tokens
from kernels.ragged_attention import triton_ragged_attention
from config import DEVICE, EMBED_DIM, NUM_HEADS, HEAD_DIM


class RaggedAttentionBlock(nn.Module):
    """
    A single transformer block that operates on packed [Total, D] tokens
    using cu_seqlens-guided Triton ragged attention.
    
    Re-uses the weights from a timm Block.
    """

    def __init__(self, block):
        super().__init__()
        self.norm1 = block.norm1
        self.qkv_proj = block.attn.qkv       # Linear(D, 3*D)
        self.out_proj = block.attn.proj       # Linear(D, D)
        self.attn_drop = block.attn.attn_drop
        self.proj_drop = block.attn.proj_drop

        self.norm2 = block.norm2
        self.mlp = block.mlp

        self.ls1 = block.ls1 if hasattr(block, "ls1") else nn.Identity()
        self.ls2 = block.ls2 if hasattr(block, "ls2") else nn.Identity()
        self.drop_path1 = block.drop_path1 if hasattr(block, "drop_path1") else nn.Identity()
        self.drop_path2 = block.drop_path2 if hasattr(block, "drop_path2") else nn.Identity()

        self.num_heads = block.attn.num_heads
        self.head_dim = block.attn.head_dim

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        """
        x: [Total, D]   packed tokens
        cu_seqlens: [B+1] int32
        """
        Total, D = x.shape

        # ── self-attention with ragged kernel ────────────────────────
        residual = x
        x_norm = self.norm1(x)

        qkv = self.qkv_proj(x_norm)                       # [Total, 3*D]
        qkv = qkv.reshape(Total, 3, self.num_heads, self.head_dim)
        q = qkv[:, 0].contiguous()   # [Total, H, d]
        k = qkv[:, 1].contiguous()   # [Total, H, d]
        v = qkv[:, 2].contiguous()   # [Total, H, d]

        attn_out = triton_ragged_attention(q, k, v, cu_seqlens)  # [Total, H, d]

        attn_out = attn_out.reshape(Total, D)              # merge heads
        attn_out = self.out_proj(attn_out)
        attn_out = self.proj_drop(attn_out)

        x = residual + self.drop_path1(self.ls1(attn_out))

        # ── MLP ──────────────────────────────────────────────────────
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        return x


class TritonRaggedDeiT(nn.Module):
    """
    Full pipeline:
      front → early blocks → Triton pack → ragged late blocks → classify
    """

    def __init__(self, front, early, late_blocks_list, back):
        super().__init__()
        self.front = front
        self.early = early
        self.ragged_blocks = nn.ModuleList(
            [RaggedAttentionBlock(b) for b in late_blocks_list]
        )
        self.back_norm = back.norm
        self.back_head = back.head

    @torch.inference_mode()
    def forward(self, images: torch.Tensor, fixed_ratio: float | None = None):
        B = images.shape[0]

        x = self.front(images)       # [B, 197, 384]
        x = self.early(x)            # [B, 197, 384]

        # ── prune + pack ─────────────────────────────────────────────
        mask = threshold_prune_mask(x, fixed_ratio=fixed_ratio)  # [B, 197]
        packed, cu_seqlens = triton_pack_tokens(x, mask)         # [T, 384], [B+1]

        # ── ragged transformer layers ────────────────────────────────
        for block in self.ragged_blocks:
            packed = block(packed, cu_seqlens)

        # ── extract CLS tokens (always at the start of each image) ──
        cls_indices = cu_seqlens[:-1].long()                     # [B]
        cls_tokens = packed[cls_indices]                          # [B, 384]

        cls_tokens = self.back_norm(cls_tokens)
        logits = self.back_head(cls_tokens)
        return logits


def build_triton_ragged_model():
    """Returns a TritonRaggedDeiT ready for inference."""
    deit = load_deit()
    front, early, late_seq, back = split_deit(deit)
    late_blocks_list = list(late_seq)

    model = TritonRaggedDeiT(front, early, late_blocks_list, back)
    model = model.to(DEVICE, dtype=get_dtype()).eval()
    return model
