"""
ToMe — Token Merging baseline (PyTorch only, no Triton kernels)
================================================================
Implements the token reduction strategy from:
  "Token Merging: Your ViT But Faster" (Bolya et al., ICLR 2023)

Key idea:  Instead of *dropping* tokens, ToMe *merges* the most similar
pairs within each transformer block using Bipartite Soft Matching on
the key vectors.  The sequence shrinks progressively through layers.

Why ToMe is a good comparison:
  - Token merging preserves information (averaging, not discarding)
  - No extra parameters — purely based on key similarity
  - Reduces seq length in every block, not just at the pruning boundary
  - The "gold standard" for no-retraining ViT acceleration

Why we don't apply Triton ragged kernels to ToMe:
  - ToMe merges tokens uniformly across the batch (every image merges
    the same number of tokens per block), so the batch stays rectangular —
    there is no ragged tensor to process.

This file provides a single PyTorch baseline: DeiT-Small with ToMe
merging applied to every late block.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.deit_base import load_deit, split_deit, get_dtype
from config import DEVICE, PRUNE_AFTER_LAYER, NUM_LAYERS


# ─────────────────────────────────────────────────────────────────────
# ToMe core: Bipartite Soft Matching
# ─────────────────────────────────────────────────────────────────────

def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Bipartite Soft Matching (Algorithm 1 from Bolya et al.).

    metric: [B, N, C]  — key vectors (or any token features)
    r:      number of token pairs to merge (reduces seq by r)

    Returns:
        merge_dst: [B, N-r] long — indices into original that form the
                   merged sequence (dst tokens + src→dst assignments)
        unm_idx:   [B, N-2r] long — indices of tokens that were NOT merged

    Simplified: we partition tokens into alternating src/dst sets,
    find the top-r most similar (src, dst) pairs, and average them.
    """
    B, N, C = metric.shape

    if r <= 0 or r >= N // 2:
        # Nothing to merge — return identity-compatible values
        dst_idx = torch.arange(0, N, 2, device=metric.device)
        src_idx = torch.arange(1, N, 2, device=metric.device)
        node_idx = torch.zeros(B, src_idx.shape[0], dtype=torch.long, device=metric.device)
        edge_idx = torch.zeros(B, 0, dtype=torch.long, device=metric.device)
        return dst_idx, src_idx, node_idx, edge_idx, 0

    # Normalize for cosine similarity
    metric = F.normalize(metric, dim=-1)

    # Partition into dst (even indices) and src (odd indices)
    # This is the standard ToMe partition
    dst_idx = torch.arange(0, N, 2, device=metric.device)  # [N/2]
    src_idx = torch.arange(1, N, 2, device=metric.device)  # [N/2]

    n_dst = dst_idx.shape[0]
    n_src = src_idx.shape[0]

    # Gather features
    dst_feat = metric[:, dst_idx]  # [B, n_dst, C]
    src_feat = metric[:, src_idx]  # [B, n_src, C]

    # Cosine similarity: [B, n_src, n_dst]
    scores = torch.bmm(src_feat, dst_feat.transpose(1, 2))

    # For each src token, find its best-matching dst token
    # node_max: [B, n_src] — best similarity per src
    # node_idx: [B, n_src] — which dst is the best match
    node_max, node_idx = scores.max(dim=-1)

    # Select top-r src tokens to merge (highest similarity to their dst)
    # edge_val: [B, r], edge_idx: [B, r] — index into src
    r_actual = min(r, n_src)
    _, edge_idx = node_max.topk(r_actual, dim=-1)  # [B, r_actual]

    return dst_idx, src_idx, node_idx, edge_idx, r_actual


def merge_tokens(
    x: torch.Tensor,
    dst_idx: torch.Tensor,
    src_idx: torch.Tensor,
    node_idx: torch.Tensor,
    edge_idx: torch.Tensor,
    r: int,
) -> torch.Tensor:
    """
    Merge r source tokens into their matched destinations by averaging.

    x: [B, N, D] — token embeddings
    Returns: [B, N-r, D] — merged token embeddings
    """
    B, N, D = x.shape
    n_src = src_idx.shape[0]

    if r <= 0:
        # Nothing to merge — return input unchanged
        return x

    # Separate src and dst tokens
    dst_tokens = x[:, dst_idx]  # [B, n_dst, D]
    src_tokens = x[:, src_idx]  # [B, n_src, D]

    # Create a mask of which src tokens are being merged
    # edge_idx: [B, r] — indices into src set that will be merged
    src_merged_mask = torch.zeros(B, n_src, dtype=torch.bool, device=x.device)
    src_merged_mask.scatter_(1, edge_idx, True)

    # For merged src tokens, add them to their dst match (then average)
    # We accumulate into dst using scatter_add
    dst_counts = torch.ones(B, dst_idx.shape[0], 1, device=x.device, dtype=x.dtype)

    for b in range(B):
        merged_src_local = edge_idx[b]              # [r] — src indices being merged
        merged_dst_local = node_idx[b, merged_src_local]  # [r] — their dst matches
        dst_tokens[b].scatter_add_(
            0,
            merged_dst_local.unsqueeze(-1).expand(-1, D),
            src_tokens[b, merged_src_local],
        )
        dst_counts[b].scatter_add_(
            0,
            merged_dst_local.unsqueeze(-1),
            torch.ones(r, 1, device=x.device, dtype=x.dtype),
        )

    # Average the merged tokens
    dst_tokens = dst_tokens / dst_counts

    # Unmerged src tokens (keep as-is)
    unmerged_src = src_tokens[~src_merged_mask.unsqueeze(-1).expand_as(src_tokens)].reshape(B, n_src - r, D)

    # Concatenate: merged dst tokens + unmerged src tokens
    out = torch.cat([dst_tokens, unmerged_src], dim=1)  # [B, n_dst + n_src - r, D] = [B, N-r, D]

    return out


# ─────────────────────────────────────────────────────────────────────
# ToMe Block: wraps a timm Block to merge r tokens before attention
# ─────────────────────────────────────────────────────────────────────

class ToMeBlock(nn.Module):
    """
    Wraps a timm Block.  Before the self-attention, extracts keys
    and merges the r most similar token pairs.
    """

    def __init__(self, block, r: int = 12):
        super().__init__()
        self.norm1 = block.norm1
        self.attn = block.attn       # full timm Attention — no mask needed
        self.norm2 = block.norm2
        self.mlp = block.mlp
        self.ls1 = block.ls1 if hasattr(block, "ls1") else nn.Identity()
        self.ls2 = block.ls2 if hasattr(block, "ls2") else nn.Identity()
        self.drop_path1 = block.drop_path1 if hasattr(block, "drop_path1") else nn.Identity()
        self.drop_path2 = block.drop_path2 if hasattr(block, "drop_path2") else nn.Identity()

        # For extracting keys for matching
        self.qkv_proj = block.attn.qkv
        self.num_heads = block.attn.num_heads
        self.head_dim = block.attn.head_dim

        self.r = r

    def forward(self, x):
        B, N, D = x.shape
        r = min(self.r, (N - 1) // 2)  # can't merge more than half, protect CLS

        if r > 0 and N > 2:
            # Extract keys for bipartite matching (on normalized input)
            x_norm = self.norm1(x)
            qkv = self.qkv_proj(x_norm).reshape(B, N, 3, self.num_heads, self.head_dim)
            keys = qkv[:, :, 1].mean(dim=2)  # Average keys across heads: [B, N, head_dim]

            # Protect CLS: only match among patch tokens (positions 1:)
            cls_token = x[:, :1]      # [B, 1, D]
            patch_tokens = x[:, 1:]   # [B, N-1, D]
            patch_keys = keys[:, 1:]  # [B, N-1, head_dim]

            r_patch = min(r, (N - 1) // 2)
            dst_idx, src_idx, node_idx, edge_idx, r_actual = bipartite_soft_matching(
                patch_keys, r_patch,
            )

            patch_merged = merge_tokens(
                patch_tokens, dst_idx, src_idx, node_idx, edge_idx, r_actual,
            )

            # Reassemble: CLS + merged patches
            x = torch.cat([cls_token, patch_merged], dim=1)  # [B, 1 + (N-1-r), D]

        # Standard transformer block on the (shorter) sequence
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        return x


# ─────────────────────────────────────────────────────────────────────
# Full ToMe DeiT-Small model
# ─────────────────────────────────────────────────────────────────────

def build_tome_model():
    """
    Returns a DeiT-Small with ToMe merging in every late block.

    Callable as model(images, fixed_ratio=0.5).
    fixed_ratio controls the total fraction of tokens removed:
      total_to_remove = int(196 * fixed_ratio)
      r_per_block = total_to_remove // num_late_blocks
    """
    deit = load_deit()
    front, early, late_seq, back = split_deit(deit)
    late_blocks_list = list(late_seq)
    num_late = len(late_blocks_list)

    class ToMeDeiT(nn.Module):
        def __init__(self):
            super().__init__()
            self.front = front
            self.early = early
            self.back = back
            # Store raw blocks — we'll wrap them dynamically with the right r
            self._raw_late_blocks = nn.ModuleList(late_blocks_list)
            self._tome_blocks = None
            self._cached_ratio = None

        def _build_tome_blocks(self, fixed_ratio):
            """(Re)build ToMeBlock wrappers with the right r per block."""
            if fixed_ratio is None:
                fixed_ratio = 0.5
            total_remove = int(196 * fixed_ratio)
            r_per_block = max(0, total_remove // num_late)
            self._tome_blocks = [
                ToMeBlock(b, r=r_per_block) for b in self._raw_late_blocks
            ]
            self._cached_ratio = fixed_ratio

        @torch.inference_mode()
        def forward(self, images, fixed_ratio=None):
            if fixed_ratio is None:
                fixed_ratio = 0.5

            # Rebuild ToMe blocks if ratio changed
            if self._cached_ratio != fixed_ratio:
                self._build_tome_blocks(fixed_ratio)

            x = self.front(images)    # [B, 197, 384]
            x = self.early(x)        # [B, 197, 384]

            # ToMe blocks progressively merge tokens
            for block in self._tome_blocks:
                x = block(x)

            return self.back(x)

    model = ToMeDeiT().to(DEVICE, dtype=get_dtype()).eval()
    model._build_tome_blocks(0.5)  # pre-build for default ratio
    return model
