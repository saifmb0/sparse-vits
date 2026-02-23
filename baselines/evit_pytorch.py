"""
EViT + PyTorch Padded baseline.

Pipeline:
  front → early[0:-1] → EViTLastBlock (captures attn weights)
        → evit_cls_attention_mask → gather + pad → padded late blocks → classify

Zero extra parameters — pruning decisions come from the attention
weights that the network already computes.  The only variable cost
is the gather/pad overhead from the PyTorch baseline.
"""

import torch
import torch.nn as nn

from models.deit_base import load_deit, split_deit, get_dtype
from models.evit_gate import EViTLastBlock, evit_cls_attention_mask
from models.pruning import pytorch_gather_and_pad
from baselines.pytorch_pruned import PaddedBlock
from config import DEVICE, PRUNE_AFTER_LAYER


def build_evit_pytorch_model():
    """
    Returns a model with EViT CLS-attention pruning + PyTorch padded execution.
    Callable as model(images, fixed_ratio=0.5).
    """
    deit = load_deit()
    front, early_seq, late_seq, back = split_deit(deit)

    # Split early blocks: first (L-1) run normally, last one captures attn
    early_blocks_list = list(early_seq)
    early_prefix = nn.Sequential(*early_blocks_list[:-1])  # blocks[0:3]
    last_early = EViTLastBlock(early_blocks_list[-1])       # block[3] with capture

    late_blocks = nn.ModuleList([PaddedBlock(b) for b in late_seq])

    class EViTPaddedDeiT(nn.Module):
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
            x, attn_weights = self.last_early(x)  # x: [B,197,384], attn: [B,H,S,S]

            # ── EViT mask from CLS attention ─────────────────────
            mask = evit_cls_attention_mask(
                attn_weights, fixed_ratio=fixed_ratio,
            )                                      # [B, 197] bool

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

    model = EViTPaddedDeiT().to(DEVICE, dtype=get_dtype()).eval()
    return model
