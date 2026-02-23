"""
EViT — CLS-Attention Token Pruning (Passive, Zero-Parameter)
=============================================================
Implements the pruning strategy from:
  "Not All Patches are What You Need: Expediting Vision Transformers
   via Token Reorganizations" (Liang et al., ICLR 2022)

Key idea:  Use the CLS token's attention scores from the LAST early
block to rank patch token importance.  Tokens with high CLS attention
are inattentive → keep them.  Tokens with low CLS attention are
attentive → prune them.

Advantages over DynamicViT:
  - Zero extra parameters (no MLP gate)
  - Zero overhead (attention weights are already computed)
  - Uses a signal that the network has *already learnt* to produce

This module provides:
  1. EViTLastBlock — runs one transformer block while capturing attention weights
  2. evit_cls_attention_mask() — generates a keep mask from CLS attention scores
"""

import torch
import torch.nn as nn

from config import KEEP_CLS


class EViTLastBlock(nn.Module):
    """
    Wraps a timm Block to execute it AND return the attention weights
    from its self-attention layer.  This is the last early-stage block
    (block index PRUNE_AFTER_LAYER - 1).

    On forward:
      x_out, attn_weights = block(x)
      attn_weights: [B, H, S, S]  (after softmax, before attn_drop)
    """

    def __init__(self, block):
        super().__init__()
        self.norm1 = block.norm1
        self.qkv = block.attn.qkv
        self.proj = block.attn.proj
        self.attn_drop = block.attn.attn_drop
        self.proj_drop = block.attn.proj_drop
        self.num_heads = block.attn.num_heads
        self.head_dim = block.attn.head_dim
        self.scale = block.attn.scale

        self.norm2 = block.norm2
        self.mlp = block.mlp

        self.ls1 = block.ls1 if hasattr(block, "ls1") else nn.Identity()
        self.ls2 = block.ls2 if hasattr(block, "ls2") else nn.Identity()
        self.drop_path1 = block.drop_path1 if hasattr(block, "drop_path1") else nn.Identity()
        self.drop_path2 = block.drop_path2 if hasattr(block, "drop_path2") else nn.Identity()

    def forward(self, x: torch.Tensor):
        """
        x: [B, S, D]
        Returns: (x_out [B, S, D], attn_weights [B, H, S, S])
        """
        B, S, D = x.shape
        H = self.num_heads
        d = self.head_dim

        # ── Self-attention with weight capture ────────────────────
        residual = x
        x_norm = self.norm1(x)

        qkv = self.qkv(x_norm).reshape(B, S, 3, H, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                          # each [B, H, S, d]

        attn = (q @ k.transpose(-2, -1)) * self.scale    # [B, H, S, S]
        attn = attn.softmax(dim=-1)

        # Capture attention weights BEFORE dropout
        attn_weights = attn.detach()

        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, S, D)
        out = self.proj(out)
        out = self.proj_drop(out)

        x = residual + self.drop_path1(self.ls1(out))

        # ── MLP ──────────────────────────────────────────────────
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        return x, attn_weights


def evit_cls_attention_mask(
    attn_weights: torch.Tensor,
    fixed_ratio: float | None = None,
    prune_low: float = 0.30,
    prune_high: float = 0.70,
) -> torch.Tensor:
    """
    Generate a keep mask from CLS token attention scores.

    attn_weights: [B, H, S, S]  — attention weights from last early block
    Returns: keep_mask [B, S] bool

    CLS importance score per patch = mean over heads of attn[cls→patch]
        = attn_weights[:, :, 0, 1:].mean(dim=1)   → [B, S-1]
    """
    B, H, S, _ = attn_weights.shape

    # CLS token (position 0) attending to all patch tokens (positions 1:)
    cls_attn = attn_weights[:, :, 0, 1:]           # [B, H, S-1]
    scores = cls_attn.mean(dim=1)                    # [B, S-1] — mean over heads

    num_patches = S - 1  # exclude CLS

    if fixed_ratio is not None:
        ratios = torch.full((B,), fixed_ratio, device=scores.device)
    else:
        ratios = torch.empty(B, device=scores.device).uniform_(prune_low, prune_high)

    # Number of patch tokens to KEEP
    num_keep = (num_patches * (1.0 - ratios)).long().clamp(min=1)  # [B]

    # Sort by score descending, keep top-k
    sorted_idx = scores.argsort(dim=-1, descending=True)           # [B, S-1]
    ranks = torch.zeros_like(sorted_idx)
    ranks.scatter_(
        1,
        sorted_idx,
        torch.arange(num_patches, device=scores.device).expand(B, -1),
    )

    patch_mask = ranks < num_keep.unsqueeze(1)                     # [B, S-1]

    # Build full mask [B, S] — CLS (pos 0) is always kept
    keep_mask = torch.cat(
        [torch.ones(B, 1, dtype=torch.bool, device=scores.device), patch_mask],
        dim=1,
    )                                                               # [B, S]

    if KEEP_CLS:
        keep_mask[:, 0] = True  # redundant but explicit

    return keep_mask
