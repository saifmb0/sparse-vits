"""
PyTorch Pruned DeiT-Small — the "padded" baseline.

After layer PRUNE_AFTER_LAYER we:
  1. Compute a keep-mask via threshold pruning
  2. Gather kept tokens with a Python loop (torch.masked_select per image)
  3. Pad back to [B, 197, D] with zeros
  4. Build an additive attention mask for the remaining layers
  5. Run all late blocks with the attention mask

This is the standard PyTorch approach and it's wasteful because:
  - gather + scatter is slow and memory-fragmenting
  - padded zeros still flow through all MatMuls
  - the attention mask adds extra overhead
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.deit_base import load_deit, split_deit, get_dtype
from models.pruning import threshold_prune_mask, pytorch_gather_and_pad
from config import DEVICE, SEQ_LEN, EMBED_DIM, NUM_HEADS


class PaddedAttention(nn.Module):
    """
    Standard multi-head attention that accepts an additive attention mask
    to hide padded positions.  Wraps timm's Block.attn.
    """

    def __init__(self, attn_module):
        super().__init__()
        self.qkv = attn_module.qkv
        self.proj = attn_module.proj
        self.attn_drop = attn_module.attn_drop
        self.proj_drop = attn_module.proj_drop
        self.num_heads = attn_module.num_heads
        self.head_dim = attn_module.head_dim
        self.scale = attn_module.scale

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # [3, B, H, N, d]
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]

        if attn_mask is not None:
            attn = attn + attn_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class PaddedBlock(nn.Module):
    """Wraps a timm Block to accept an attn_mask parameter."""

    def __init__(self, block):
        super().__init__()
        self.norm1 = block.norm1
        self.attn = PaddedAttention(block.attn)
        self.norm2 = block.norm2
        self.mlp = block.mlp
        self.ls1 = block.ls1 if hasattr(block, "ls1") else nn.Identity()
        self.ls2 = block.ls2 if hasattr(block, "ls2") else nn.Identity()
        self.drop_path1 = block.drop_path1 if hasattr(block, "drop_path1") else nn.Identity()
        self.drop_path2 = block.drop_path2 if hasattr(block, "drop_path2") else nn.Identity()

    def forward(self, x, attn_mask=None):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


def build_pytorch_pruned_model():
    """
    Returns a callable (images, fixed_ratio=None) → logits
    that does threshold pruning + padding internally.
    """
    deit = load_deit()
    front, early, late_seq, back = split_deit(deit)

    # Wrap late blocks to support attn_mask
    late_blocks = nn.ModuleList([PaddedBlock(b) for b in late_seq])

    class PytorchPrunedDeiT(nn.Module):
        def __init__(self):
            super().__init__()
            self.front = front
            self.early = early
            self.late_blocks = late_blocks
            self.back = back

        @torch.inference_mode()
        def forward(self, images, fixed_ratio=None):
            x = self.front(images)        # [B, 197, 384]
            x = self.early(x)             # [B, 197, 384]

            # ── prune ────────────────────────────────────────────
            mask = threshold_prune_mask(x, fixed_ratio=fixed_ratio)  # [B, 197]
            x_pad, attn_mask_bool, num_kept = pytorch_gather_and_pad(x, mask)
            # x_pad: [B, 197, 384]   attn_mask_bool: [B, 197]

            # Build additive attention mask:  0 where valid, -inf where padded
            # Shape: [B, 1, 1, S] — broadcast over heads and query dim
            B, S, D = x_pad.shape
            attn_mask = torch.zeros(B, 1, 1, S, device=x.device, dtype=x.dtype)
            attn_mask.masked_fill_(~attn_mask_bool.unsqueeze(1).unsqueeze(2), float("-inf"))

            # ── run late layers with mask ────────────────────────
            for block in self.late_blocks:
                x_pad = block(x_pad, attn_mask=attn_mask)

            return self.back(x_pad)

    model = PytorchPrunedDeiT().to(DEVICE, dtype=get_dtype()).eval()
    return model
