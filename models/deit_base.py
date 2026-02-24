"""
DeiT baseline loader from timm.
Splits the model into:
  - patch_embed + pos_embed + cls_token  (front-end)
  - blocks[0:PRUNE_AFTER_LAYER]          (early layers)
  - blocks[PRUNE_AFTER_LAYER:]           (late layers)
  - norm + head                          (classifier)

Supports DeiT-Tiny, DeiT-Small, and DeiT-Base via the optional
`model_name` parameter.  All existing code that calls load_deit()
without arguments gets DeiT-Small (backward compatible).
"""

import torch
import torch.nn as nn
import timm

from config import (
    MODEL_NAME, EMBED_DIM, NUM_HEADS, SEQ_LEN,
    NUM_LAYERS, PRUNE_AFTER_LAYER, DEVICE, DTYPE,
)


def get_dtype():
    return torch.float16 if DTYPE == "float16" else torch.float32


def load_deit(model_name: str | None = None):
    """Return a pretrained DeiT model in eval mode on DEVICE.

    Args:
        model_name: timm model identifier (e.g. 'deit_base_patch16_224').
                    Defaults to MODEL_NAME from config.py (DeiT-Small).
    """
    if model_name is None:
        model_name = MODEL_NAME
    model = timm.create_model(model_name, pretrained=True)
    model = model.to(DEVICE, dtype=get_dtype()).eval()
    return model


class DeiTFrontEnd(nn.Module):
    """Patch embed → CLS token → position embed."""

    def __init__(self, deit):
        super().__init__()
        self.patch_embed = deit.patch_embed
        self.cls_token = deit.cls_token
        self.pos_embed = deit.pos_embed
        self.pos_drop = deit.pos_drop if hasattr(deit, "pos_drop") else nn.Identity()

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)                          # [B, 196, 384]
        cls = self.cls_token.expand(B, -1, -1)           # [B, 1, 384]
        x = torch.cat([cls, x], dim=1)                   # [B, 197, 384]
        x = x + self.pos_embed
        x = self.pos_drop(x)
        return x


class DeiTBackEnd(nn.Module):
    """LayerNorm → classifier head."""

    def __init__(self, deit):
        super().__init__()
        self.norm = deit.norm
        self.head = deit.head

    def forward(self, x):
        # x: [B, S, D] — take CLS token
        x = self.norm(x)
        return self.head(x[:, 0])


def split_deit(deit, prune_after_layer: int | None = None):
    """
    Returns (front_end, early_blocks, late_blocks, back_end).

    Args:
        prune_after_layer: Layer index to split at.  Defaults to
                           PRUNE_AFTER_LAYER from config.py (4).
    """
    if prune_after_layer is None:
        prune_after_layer = PRUNE_AFTER_LAYER
    front = DeiTFrontEnd(deit)
    early = nn.Sequential(*list(deit.blocks[:prune_after_layer]))
    late  = nn.Sequential(*list(deit.blocks[prune_after_layer:]))
    back  = DeiTBackEnd(deit)
    return front, early, late, back
