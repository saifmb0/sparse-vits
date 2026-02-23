"""
Standard (unpruned) DeiT-Small forward pass — our throughput baseline.
"""

import torch
from models.deit_base import load_deit


@torch.inference_mode()
def run_unpruned(images: torch.Tensor, model=None) -> torch.Tensor:
    """
    Full DeiT-S forward (no pruning).
    images: [B, 3, 224, 224]
    Returns logits: [B, num_classes]
    """
    if model is None:
        model = load_deit()
    return model(images)
