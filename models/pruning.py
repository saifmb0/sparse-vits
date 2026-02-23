"""
Threshold-based dynamic token pruning (A-ViT / DynamicViT style).

Generates a per-image, per-token boolean keep-mask based on token
importance scores.  Importance = L2-norm of the token embedding
(simple proxy — no learnable gate needed for inference benchmarking).

The pruning ratio is drawn randomly per image in [low, high] so that
every image in the batch ends up with a *different* sequence length,
creating the "ragged batch" scenario.
"""

import torch
from config import PRUNE_RATIO_LOW, PRUNE_RATIO_HIGH, KEEP_CLS


def compute_importance(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B, S, D]
    Returns scores: [B, S]  (L2 norm per token).
    """
    return x.norm(dim=-1)  # [B, S]


def threshold_prune_mask(
    x: torch.Tensor,
    prune_low: float = PRUNE_RATIO_LOW,
    prune_high: float = PRUNE_RATIO_HIGH,
    fixed_ratio: float | None = None,
) -> torch.Tensor:
    """
    Returns a boolean keep-mask [B, S] where True = keep.

    If fixed_ratio is given, every image drops exactly that fraction.
    Otherwise each image draws a random ratio in [prune_low, prune_high].
    """
    B, S, D = x.shape
    scores = compute_importance(x)  # [B, S]

    if fixed_ratio is not None:
        ratios = torch.full((B,), fixed_ratio, device=x.device)
    else:
        ratios = torch.empty(B, device=x.device).uniform_(prune_low, prune_high)

    # Number of tokens to KEEP per image
    num_keep = (S * (1.0 - ratios)).long().clamp(min=1)  # [B]

    # Sort by score descending, keep top-k
    sorted_idx = scores.argsort(dim=-1, descending=True)  # [B, S]
    ranks = torch.zeros_like(sorted_idx)
    ranks.scatter_(1, sorted_idx, torch.arange(S, device=x.device).expand(B, -1))

    mask = ranks < num_keep.unsqueeze(1)  # [B, S]

    if KEEP_CLS:
        mask[:, 0] = True  # CLS is always position 0

    return mask  # [B, S]  bool


def pytorch_gather_and_pad(x: torch.Tensor, mask: torch.Tensor):
    """
    PyTorch-native way to extract kept tokens and pad back.

    Returns:
        padded:  [B, S, D]  — kept tokens first, zeros after
        attn_mask: [B, S]   — True where real tokens live
        num_kept:  [B]      — number of kept tokens per image
    """
    B, S, D = x.shape
    device = x.device

    num_kept = mask.sum(dim=1)  # [B]

    # Gather kept tokens per image
    padded = torch.zeros_like(x)  # [B, S, D]
    attn_mask = torch.zeros(B, S, dtype=torch.bool, device=device)

    for i in range(B):
        kept = x[i][mask[i]]            # [K_i, D]
        K = kept.shape[0]
        padded[i, :K] = kept
        attn_mask[i, :K] = True

    return padded, attn_mask, num_kept
