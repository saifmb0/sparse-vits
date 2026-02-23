"""
ATS + PyTorch Padded baseline.

Pipeline:
  front → early[0:-1] → EViTLastBlock (captures attn weights)
        → ATS scoring (cls_attn × val_norm → PDF → CDF)
        → gather + pad → padded late blocks → classify

Parameter-free — no learned gate, but heavier mask-generation overhead
(sort + cumsum + scatter) than EViT's simple threshold.
"""

import torch
import torch.nn as nn

from models.deit_base import load_deit, split_deit, get_dtype
from models.evit_gate import EViTLastBlock          # reuse attn-capture block
from models.ats_gate import ats_importance_scores, ats_cdf_mask
from models.pruning import pytorch_gather_and_pad
from baselines.pytorch_pruned import PaddedBlock
from config import DEVICE


def build_ats_pytorch_model():
    """
    Returns a model with ATS CDF-based pruning + PyTorch padded execution.
    Callable as model(images, fixed_ratio=0.5).
    """
    deit = load_deit()
    front, early_seq, late_seq, back = split_deit(deit)

    early_blocks_list = list(early_seq)
    early_prefix = nn.Sequential(*early_blocks_list[:-1])
    last_early = EViTLastBlock(early_blocks_list[-1])

    late_blocks = nn.ModuleList([PaddedBlock(b) for b in late_seq])

    class ATSPaddedDeiT(nn.Module):
        def __init__(self):
            super().__init__()
            self.front = front
            self.early_prefix = early_prefix
            self.last_early = last_early
            self.late_blocks = late_blocks
            self.back = back

        @torch.inference_mode()
        def forward(self, images, fixed_ratio=None):
            x = self.front(images)              # [B, 197, 384]
            x = self.early_prefix(x)            # [B, 197, 384]

            # ── Last early block + attention weight capture ──────
            x, attn_weights = self.last_early(x)

            # ── ATS scoring + CDF mask ───────────────────────────
            scores = ats_importance_scores(attn_weights, x)
            mask = ats_cdf_mask(scores, fixed_ratio=fixed_ratio)

            # ── gather + pad (PyTorch way) ───────────────────────
            x_pad, attn_mask_bool, num_kept = pytorch_gather_and_pad(x, mask)

            B, S, D = x_pad.shape
            attn_mask = torch.zeros(B, 1, 1, S, device=x.device, dtype=x.dtype)
            attn_mask.masked_fill_(
                ~attn_mask_bool.unsqueeze(1).unsqueeze(2), float("-inf"),
            )

            # ── late layers with mask ────────────────────────────
            for block in self.late_blocks:
                x_pad = block(x_pad, attn_mask=attn_mask)

            return self.back(x_pad)

    model = ATSPaddedDeiT().to(DEVICE, dtype=get_dtype()).eval()
    return model
