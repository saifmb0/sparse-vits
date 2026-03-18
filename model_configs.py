"""
Model registry — defines configurations for each ViT variant
used in the model-diversity experiments.

All models use patch_size=16, img_size=224, seq_len=197, head_dim=64.
This means our Triton kernels (BLOCK_M=32, BLOCK_N=32 tuned for
head_dim=64) work for all variants without modification.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    timm_name: str          # timm model identifier
    short_name: str         # human-readable label for plots
    embed_dim: int
    num_heads: int
    head_dim: int           # always 64 for DeiT family
    num_layers: int
    num_patches: int        # 196 for patch16_224
    seq_len: int            # 197 = 196 + CLS
    num_params_m: float     # approximate param count in millions


# ── DeiT-I family (original development models) ─────────────────────

DEIT_TINY = ModelConfig(
    timm_name="deit_tiny_patch16_224",
    short_name="DeiT-Ti",
    embed_dim=192,
    num_heads=3,
    head_dim=64,
    num_layers=12,
    num_patches=196,
    seq_len=197,
    num_params_m=5.7,
)

DEIT_SMALL = ModelConfig(
    timm_name="deit_small_patch16_224",
    short_name="DeiT-S",
    embed_dim=384,
    num_heads=6,
    head_dim=64,
    num_layers=12,
    num_patches=196,
    seq_len=197,
    num_params_m=22.1,
)

DEIT_BASE = ModelConfig(
    timm_name="deit_base_patch16_224",
    short_name="DeiT-B",
    embed_dim=768,
    num_heads=12,
    head_dim=64,
    num_layers=12,
    num_patches=196,
    seq_len=197,
    num_params_m=86.6,
)

DEIT3_LARGE = ModelConfig(
    timm_name="deit3_large_patch16_224",
    short_name="DeiT3-L",
    embed_dim=1024,
    num_heads=16,
    head_dim=64,
    num_layers=24,
    num_patches=196,
    seq_len=197,
    num_params_m=304.4,
)


# ── ViT family (T4-scale models; head_dim=64 across all patch16 variants) ──
#
# All share patch_size=16, img_size=224, seq_len=197, head_dim=64 so
# the Triton ragged kernel needs zero modification.

VIT_SMALL = ModelConfig(
    timm_name="vit_small_patch16_224",
    short_name="ViT-S/16",
    embed_dim=384,
    num_heads=6,
    head_dim=64,
    num_layers=12,
    num_patches=196,
    seq_len=197,
    num_params_m=22.1,
)

VIT_BASE = ModelConfig(
    timm_name="vit_base_patch16_224",
    short_name="ViT-B/16",
    embed_dim=768,
    num_heads=12,
    head_dim=64,
    num_layers=12,
    num_patches=196,
    seq_len=197,
    num_params_m=86.6,
)

VIT_LARGE = ModelConfig(
    timm_name="vit_large_patch16_224",
    short_name="ViT-L/16",
    embed_dim=1024,
    num_heads=16,
    head_dim=64,
    num_layers=24,
    num_patches=196,
    seq_len=197,
    num_params_m=307.4,
)

VIT_HUGE = ModelConfig(
    timm_name="vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k",
    short_name="ViT-H/14",
    embed_dim=1280,
    num_heads=16,
    head_dim=80,
    num_layers=32,
    num_patches=256,   # 16x16 patches at patch_size=14
    seq_len=257,
    num_params_m=632.0,
)


# ── Registry ─────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "deit_tiny":   DEIT_TINY,
    "deit_small":  DEIT_SMALL,
    "deit_base":   DEIT_BASE,
    "deit3_large": DEIT3_LARGE,
    "vit_small":   VIT_SMALL,
    "vit_base":    VIT_BASE,
    "vit_large":   VIT_LARGE,
    "vit_huge":    VIT_HUGE,
}

# Ordered list for scaling experiments
# Using DeiT-Tiny, DeiT-Small, and DeiT-Base instead of standard ViTs.
SCALING_MODELS = [DEIT_TINY, DEIT_SMALL, DEIT_BASE]
