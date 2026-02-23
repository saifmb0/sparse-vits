"""
ATS + Triton Ragged pipeline.

Pipeline:
  front → early[0:-1] → EViTLastBlock (captures attn weights)
        → ATS scoring (cls_attn × val_norm → PDF → CDF)
        → Triton pack → ragged late blocks → classify

This is the hardest stress test for our Triton engine: ATS's CDF-based
sampling produces the most aggressively *variable* token counts of any
method (sharp focal attention → very few tokens, diffuse → many).
The ragged engine must handle this variance gracefully.
"""

import torch
import torch.nn as nn

from models.deit_base import load_deit, split_deit, get_dtype
from models.evit_gate import EViTLastBlock
from models.ats_gate import ats_importance_scores, ats_cdf_mask
from models.triton_ragged_deit import RaggedAttentionBlock
from kernels.pack_tokens import triton_pack_tokens
from config import DEVICE


class ATSRaggedDeiT(nn.Module):
    """
    front → early prefix → last early (capture attn) → ATS CDF mask
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
        x, attn_weights = self.last_early(x)

        # ── ATS scoring + CDF mask ──────────────────────────────
        scores = ats_importance_scores(attn_weights, x)
        mask = ats_cdf_mask(scores, fixed_ratio=fixed_ratio)

        # ── Triton pack ──────────────────────────────────────────
        packed, cu_seqlens = triton_pack_tokens(x, mask)

        # ── ragged transformer layers ────────────────────────────
        for block in self.ragged_blocks:
            packed = block(packed, cu_seqlens)

        # ── extract CLS tokens ───────────────────────────────────
        cls_indices = cu_seqlens[:-1].long()
        cls_tokens = packed[cls_indices]

        cls_tokens = self.back_norm(cls_tokens)
        logits = self.back_head(cls_tokens)
        return logits


def build_ats_triton_model():
    """Returns an ATSRaggedDeiT ready for inference."""
    deit = load_deit()
    front, early_seq, late_seq, back = split_deit(deit)

    early_blocks_list = list(early_seq)
    early_prefix = nn.Sequential(*early_blocks_list[:-1])
    last_early = EViTLastBlock(early_blocks_list[-1])
    late_blocks_list = list(late_seq)

    model = ATSRaggedDeiT(front, early_prefix, last_early, late_blocks_list, back)
    model = model.to(DEVICE, dtype=get_dtype()).eval()
    return model
