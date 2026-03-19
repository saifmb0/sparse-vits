"""
Global configuration for Ragged-Batch ViT Inference Engine.
"""

# ── Model ────────────────────────────────────────────────────────────────
MODEL_NAME = "deit_large_patch16_224"       # timm model id
EMBED_DIM = 384                              # DeiT-Small hidden dim
NUM_HEADS = 6                                # DeiT-Small attention heads
HEAD_DIM = EMBED_DIM // NUM_HEADS            # 64
NUM_PATCHES = 196                            # 14x14 patches
SEQ_LEN = NUM_PATCHES + 1                   # +1 for CLS token (197)
IMG_SIZE = 224
PATCH_SIZE = 16
NUM_LAYERS = 12                              # total transformer layers

# ── Pruning ──────────────────────────────────────────────────────────────
PRUNE_AFTER_LAYER = 4                        # inject pruning mask after this layer
PRUNE_RATIO_LOW = 0.30                       # min fraction of tokens to DROP
PRUNE_RATIO_HIGH = 0.70                      # max fraction of tokens to DROP
KEEP_CLS = True                              # always keep the CLS token

# ── Benchmarking ─────────────────────────────────────────────────────────
BATCH_SIZES = [1, 4, 8, 16, 32, 64, 128, 256, 512]
PRUNE_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
WARMUP_ITERS = 10
BENCH_ITERS = 50

# ── Device ───────────────────────────────────────────────────────────────
DEVICE = "cuda"
DTYPE = "float16"                            # fp16 for GTX 1650
