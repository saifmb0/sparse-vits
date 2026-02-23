"""
DynamicViT Prediction Module — Active Router (Section 3.2)
==========================================================
Implements the lightweight MLP-based token pruning gate from:
  "DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification"
  (Rao et al., NeurIPS 2021)

Architecture (per paper Eq 1-5):
  1. Local feature:   LayerNorm → Linear(D → D/2) → GELU
  2. Global feature:  mean-pool local → expand
  3. Prediction:      cat(local, global) → Linear(D → D/2) → GELU → Linear(D/2 → 2) → Softmax

"No-Retraining Hack": instead of fixed top-k, we threshold on
keep-probability > tau.  This creates *ragged* keep-counts across
the batch — exactly the scenario our Triton engine is built for.
"""

import torch
import torch.nn as nn

from config import EMBED_DIM, KEEP_CLS


class DynamicViTPredictionModule(nn.Module):
    """
    Lightweight MLP gate that produces a per-token binary keep mask.
    Adds measurable "Mask Generation Overhead" to the pipeline.
    """

    def __init__(self, embed_dim: int = EMBED_DIM, tau: float = 0.5):
        super().__init__()
        half = embed_dim // 2

        # Local feature branch — Eq (1)
        self.local_branch = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, half),
            nn.GELU(),
        )

        # Prediction head — Eq (4)/(5): takes concat(local, global) = D-dim
        self.pred_head = nn.Sequential(
            nn.Linear(embed_dim, half),   # D → D/2  (local+global=D)
            nn.GELU(),
            nn.Linear(half, 2),           # → [drop_prob, keep_prob]
            nn.Softmax(dim=-1),
        )

        self.tau = tau

    def forward(
        self,
        x: torch.Tensor,
        fixed_ratio: float | None = None,
    ) -> torch.Tensor:
        """
        x: [B, S, D]  token embeddings (including CLS at position 0)
        Returns: keep_mask [B, S] bool
        """
        B, S, D = x.shape

        # 1. Local features  [B, S, D/2]
        local_feat = self.local_branch(x)

        # 2. Global features (mean-pool → expand)  [B, S, D/2]   — Eq (2)/(3)
        global_feat = local_feat.mean(dim=1, keepdim=True).expand_as(local_feat)

        # 3. Concatenate → predict  [B, S, 2]                    — Eq (5)
        combined = torch.cat([local_feat, global_feat], dim=-1)  # [B, S, D]
        probs = self.pred_head(combined)                          # [B, S, 2]
        keep_prob = probs[:, :, 1]                                # [B, S]

        # ── Mask generation ──────────────────────────────────────────
        if fixed_ratio is not None:
            # Deterministic: keep top-(1-ratio) fraction per image
            num_keep = max(1, int(S * (1.0 - fixed_ratio)))
            topk_vals, _ = keep_prob.topk(num_keep, dim=-1)
            threshold = topk_vals[:, -1:]                         # [B, 1]
            keep_mask = keep_prob >= threshold
        else:
            # Threshold hack → ragged counts
            keep_mask = keep_prob > self.tau

        if KEEP_CLS:
            keep_mask[:, 0] = True

        # Ensure at least 1 token kept per image
        any_kept = keep_mask.any(dim=1)
        if not any_kept.all():
            # Force-keep the highest-prob token for images with nothing kept
            max_idx = keep_prob.argmax(dim=1)
            for i in range(B):
                if not any_kept[i]:
                    keep_mask[i, max_idx[i]] = True

        return keep_mask  # [B, S] bool
