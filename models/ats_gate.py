"""
ATS — Adaptive Token Sampling (Statistical, Parameter-Free)
============================================================
Implements the token selection strategy from:
  "Adaptive Token Sampling For Efficient Vision Transformers"
  (Fayyaz et al., ECCV 2022)

Key idea:  Treat CLS-attention × Value-norm as a probability density
function (PDF).  Build the CDF via cumulative sum and use Inverse
Transform Sampling to select the minimal set of tokens that covers
a target probability mass (e.g., 90%).

Why this is a good stress test for our Triton engine:
  - The cumsum + searchsorted operations are memory-bound and harder
    to parallelize than simple thresholds.
  - The CDF crossing point varies *wildly* per image (sharp attention
    → few tokens, diffuse attention → many tokens), producing the
    most aggressively ragged batches of any method.

This module provides:
  1. ats_importance_scores() — compute ATS scores from attn + values
  2. ats_cdf_mask()          — CDF-based inverse-transform selection
  3. Both are called from the pipeline modules (ats_pytorch / ats_ragged)
"""

import torch
from config import KEEP_CLS


def ats_importance_scores(
    attn_weights: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Compute ATS importance scores (Eq 3 of the paper).

    attn_weights: [B, H, S, S]  — softmax attention from last early block
    x:            [B, S, D]     — token embeddings (used as value proxy)

    Returns: scores [B, S-1] — one per patch token, normalized to sum=1
    """
    B, H, S, _ = attn_weights.shape

    # CLS (row 0) attending to every patch (cols 1:), averaged over heads
    cls_attn = attn_weights[:, :, 0, 1:].mean(dim=1)       # [B, S-1]

    # Value-norm weighting (L2 norm of each patch embedding)
    val_norm = x[:, 1:].norm(p=2, dim=-1)                  # [B, S-1]

    # S = A_cls * ||V||
    scores = cls_attn * val_norm                             # [B, S-1]

    # Normalize to a valid PDF (sum = 1)
    scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-6)

    return scores                                            # [B, S-1]


def ats_cdf_mask(
    scores: torch.Tensor,
    fixed_ratio: float | None = None,
    mass_threshold: float = 0.90,
    prune_low: float = 0.30,
    prune_high: float = 0.70,
) -> torch.Tensor:
    """
    Generate a keep mask using CDF-based inverse transform sampling.

    scores:         [B, S-1]  normalized importance scores (patches only)
    fixed_ratio:    if set, ignore CDF and do top-k with ATS scores
                    (keeps overhead but ensures deterministic count for
                    fair benchmark comparison)
    mass_threshold: fraction of probability mass to retain (0.9 = keep
                    fewest tokens that cover 90% of the distribution)

    Returns: keep_mask [B, S] bool  (S = patches + CLS)
    """
    B, num_patches = scores.shape
    S = num_patches + 1  # +CLS

    if fixed_ratio is not None:
        # ── Deterministic top-k mode (for benchmark fairness) ────
        # Still pay full ATS scoring overhead, but prune exact fraction
        num_keep = max(1, int(num_patches * (1.0 - fixed_ratio)))

        # Sort descending → keep top-k
        sorted_scores, sorted_idx = scores.sort(dim=-1, descending=True)

        # Compute CDF anyway (pay the overhead for honest benchmarking)
        _ = torch.cumsum(sorted_scores, dim=-1)

        topk_vals, _ = scores.topk(num_keep, dim=-1)
        threshold = topk_vals[:, -1:]                          # [B, 1]
        patch_mask = scores >= threshold                        # [B, S-1]

        # Handle ties: if we got more than num_keep, trim the excess
        excess = patch_mask.sum(dim=1) - num_keep
        if (excess > 0).any():
            for i in range(B):
                if excess[i] > 0:
                    idxs = patch_mask[i].nonzero(as_tuple=False).squeeze(-1)
                    # Remove the lowest-scored among those kept
                    keep_scores = scores[i, idxs]
                    _, remove_order = keep_scores.sort()
                    for j in range(excess[i].item()):
                        patch_mask[i, idxs[remove_order[j]]] = False

    else:
        # ── True ATS: CDF inverse-transform sampling ────────────
        # Sort descending to build CDF from most important to least
        sorted_scores, sort_indices = scores.sort(dim=-1, descending=True)

        # CDF = cumulative sum of sorted PDF
        cdf = torch.cumsum(sorted_scores, dim=-1)                # [B, S-1]

        # Keep tokens whose CDF position ≤ threshold
        # (all tokens in the "top mass_threshold of the distribution")
        sorted_mask = cdf <= mass_threshold                       # [B, S-1]

        # Always keep at least the first (most important) token
        sorted_mask[:, 0] = True

        # Scatter back to original positions
        patch_mask = torch.zeros_like(scores, dtype=torch.bool)   # [B, S-1]
        patch_mask.scatter_(1, sort_indices, sorted_mask)

    # Build full mask [B, S] — CLS (position 0) always kept
    keep_mask = torch.cat(
        [torch.ones(B, 1, dtype=torch.bool, device=scores.device), patch_mask],
        dim=1,
    )

    if KEEP_CLS:
        keep_mask[:, 0] = True  # redundant but explicit

    # Ensure at least 1 token per image
    any_kept = keep_mask.any(dim=1)
    if not any_kept.all():
        max_idx = scores.argmax(dim=1) + 1  # +1 for CLS offset
        for i in range(B):
            if not any_kept[i]:
                keep_mask[i, max_idx[i]] = True

    return keep_mask  # [B, S] bool
