"""
EViT + Triton Ragged pipeline.

Pipeline:
  front → early[0:-1] → EViTLastBlock (captures attn weights)
        → evit_cls_attention_mask → Triton pack → ragged late blocks → classify

Zero extra parameters, zero padding waste.  The pruning signal comes
from CLS attention weights — already computed, effectively free.
Combined with our Triton packer + ragged attention, this should be
the best-case scenario: no gate overhead AND no padding waste.
"""

import torch
import torch.nn as nn

from models.deit_base import load_deit, split_deit, get_dtype
from models.evit_gate import EViTLastBlock, evit_cls_attention_mask
from models.triton_ragged_deit import RaggedAttentionBlock
from kernels.pack_tokens import triton_pack_tokens
from config import DEVICE, PRUNE_AFTER_LAYER


class EViTRaggedDeiT(nn.Module):
    """
    front → early prefix → last early (capture attn) → EViT mask
          → Triton pack → ragged late blocks → classify
    """

    def __init__(self, front, early_prefix, last_early, late_blocks_list, back):
        super().__init__()
        self.front = front
        self.early_prefix = early_prefix
        self.last_early = last_early
        self.ragged_blocks = nn.ModuleList(
            [RaggedAttentionBlock(b) for b in late_blocks_list]
        )
        self.back_norm = back.norm
        self.back_head = back.head

    @torch.inference_mode()
    def forward(self, images: torch.Tensor, fixed_ratio: float | None = None):
        B = images.shape[0]

        x = self.front(images)           # [B, 197, 384]
        x = self.early_prefix(x)         # [B, 197, 384]

        # ── Last early block + attention weight capture ──────────
        x, attn_weights = self.last_early(x)  # [B,197,384], [B,H,S,S]

        # ── EViT mask from CLS attention ─────────────────────────
        mask = evit_cls_attention_mask(
            attn_weights, fixed_ratio=fixed_ratio,
        )                                      # [B, 197] bool

        # ── Triton pack ──────────────────────────────────────────
        packed, cu_seqlens = triton_pack_tokens(x, mask)  # [T, 384], [B+1]

        # ── ragged transformer layers ────────────────────────────
        for block in self.ragged_blocks:
            packed = block(packed, cu_seqlens)

        # ── extract CLS tokens ───────────────────────────────────
        cls_indices = cu_seqlens[:-1].long()              # [B]
        cls_tokens = packed[cls_indices]                   # [B, 384]

        cls_tokens = self.back_norm(cls_tokens)
        logits = self.back_head(cls_tokens)
        return logits


def build_evit_triton_model():
    """Returns an EViTRaggedDeiT ready for inference."""
    deit = load_deit()
    front, early_seq, late_seq, back = split_deit(deit)

    early_blocks_list = list(early_seq)
    early_prefix = nn.Sequential(*early_blocks_list[:-1])  # blocks[0:3]
    last_early = EViTLastBlock(early_blocks_list[-1])       # block[3]
    late_blocks_list = list(late_seq)

    model = EViTRaggedDeiT(front, early_prefix, last_early, late_blocks_list, back)
    model = model.to(DEVICE, dtype=get_dtype()).eval()
    return model
