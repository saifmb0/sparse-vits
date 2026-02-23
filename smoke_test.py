#!/usr/bin/env python3
"""
Smoke test — verify that all components load and produce valid output.
Run this before the full benchmarks.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from config import DEVICE, EMBED_DIM, SEQ_LEN, NUM_HEADS, HEAD_DIM


def test_pack_kernel():
    print("Testing Triton packer kernel...")
    from kernels.pack_tokens import triton_pack_tokens

    B, S, D = 4, SEQ_LEN, EMBED_DIM
    tokens = torch.randn(B, S, D, device=DEVICE, dtype=torch.float16)

    # Create a mask where each image keeps a different number
    mask = torch.ones(B, S, dtype=torch.bool, device=DEVICE)
    mask[0, 50:] = False   # keep 50
    mask[1, 100:] = False  # keep 100
    mask[2, 150:] = False  # keep 150
    mask[3, :] = True      # keep all 197

    packed, cu_seqlens = triton_pack_tokens(tokens, mask)

    expected_total = 50 + 100 + 150 + 197
    assert packed.shape == (expected_total, D), f"Expected ({expected_total}, {D}), got {packed.shape}"
    assert cu_seqlens.shape == (B + 1,), f"Expected ({B+1},), got {cu_seqlens.shape}"
    assert cu_seqlens[0] == 0
    assert cu_seqlens[1] == 50
    assert cu_seqlens[2] == 150
    assert cu_seqlens[3] == 300
    assert cu_seqlens[4] == expected_total

    # Check that packed values match original kept tokens
    for i in range(B):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        original_kept = tokens[i][mask[i]]
        assert torch.allclose(packed[start:end], original_kept, atol=1e-3), \
            f"Image {i}: packed values don't match original kept tokens"

    print(f"  ✓ Packed shape: {packed.shape}, cu_seqlens: {cu_seqlens.tolist()}")


def test_ragged_attention():
    print("Testing Triton ragged attention kernel...")
    from kernels.ragged_attention import triton_ragged_attention

    # Simple test: 2 images, different lengths
    B = 2
    lens = [20, 50]
    Total = sum(lens)

    q = torch.randn(Total, NUM_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    k = torch.randn(Total, NUM_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    v = torch.randn(Total, NUM_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    cu_seqlens = torch.tensor([0, lens[0], Total], dtype=torch.int32, device=DEVICE)

    o = triton_ragged_attention(q, k, v, cu_seqlens)

    assert o.shape == (Total, NUM_HEADS, HEAD_DIM), f"Expected ({Total}, {NUM_HEADS}, {HEAD_DIM}), got {o.shape}"
    assert not torch.isnan(o).any(), "Output contains NaN!"
    assert not torch.isinf(o).any(), "Output contains Inf!"

    # Verify against PyTorch reference for first image
    q0 = q[:lens[0]]  # [20, H, d]
    k0 = k[:lens[0]]
    v0 = v[:lens[0]]
    scale = HEAD_DIM ** -0.5
    # [H, 20, 20]
    attn = torch.einsum("thd,shd->hts", q0.float(), k0.float()) * scale
    attn = attn.softmax(dim=-1)
    ref = torch.einsum("hts,shd->thd", attn, v0.float()).half()
    # Allow some tolerance for fp16
    err = (o[:lens[0]].float() - ref.float()).abs().max().item()
    print(f"  ✓ Output shape: {o.shape}, max error vs reference: {err:.4f}")
    assert err < 0.1, f"Error too large: {err}"


def test_full_pipeline():
    print("Testing full Triton Ragged DeiT pipeline...")
    from models.triton_ragged_deit import build_triton_ragged_model
    from models.deit_base import get_dtype

    model = build_triton_ragged_model()
    dtype = get_dtype()
    images = torch.randn(2, 3, 224, 224, device=DEVICE, dtype=dtype)

    logits = model(images, fixed_ratio=0.5)
    assert logits.shape == (2, 1000), f"Expected (2, 1000), got {logits.shape}"
    assert not torch.isnan(logits).any(), "Logits contain NaN!"
    print(f"  ✓ Output shape: {logits.shape}")

    del model
    torch.cuda.empty_cache()


def test_pytorch_pruned():
    print("Testing PyTorch Pruned DeiT pipeline...")
    from baselines.pytorch_pruned import build_pytorch_pruned_model
    from models.deit_base import get_dtype

    model = build_pytorch_pruned_model()
    dtype = get_dtype()
    images = torch.randn(2, 3, 224, 224, device=DEVICE, dtype=dtype)

    logits = model(images, fixed_ratio=0.5)
    assert logits.shape == (2, 1000), f"Expected (2, 1000), got {logits.shape}"
    assert not torch.isnan(logits).any(), "Logits contain NaN!"
    print(f"  ✓ Output shape: {logits.shape}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print("=" * 50)
    print("SMOKE TESTS")
    print("=" * 50)

    test_pack_kernel()
    print()
    test_ragged_attention()
    print()
    test_pytorch_pruned()
    print()
    test_full_pipeline()

    print()
    print("=" * 50)
    print("ALL SMOKE TESTS PASSED ✓")
    print("=" * 50)
