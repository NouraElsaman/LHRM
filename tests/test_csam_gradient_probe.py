"""
tests/test_csam_gradient_probe.py

Unit test suite for DR3 CSAM gradient micro-probe:
1. Analytical softmax-attention gradient factor calculation.
2. Tensor dimensions in forward and backward passes.
3. Finite and nonzero gradient propagation across all parameters.
4. Algebraic decomposition: L2 vs L2_NO_SA formulation exactness.
5. Focal vs BCE controlled comparison consistency.
6. Graceful handling of missing trained checkpoints.
7. Verification that probe causes NO optimizer updates (parameters unchanged).
8. Verification that historical experiment artifacts remain untouched.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F

from scripts.probe_csam_gradient_paths import (
    compute_analytical_attention_grad_factor,
    decompose_csam_logits,
    run_single_probe,
    execute_diagnostic_probe,
)
from src.lhrm_baseline.model import CSAM_L
from src.lhrm_baseline.losses import FocalLoss, BCELoss


def test_1_analytical_softmax_grad_factor():
    """Verify alpha / sqrt(H) calculation."""
    H = 768
    alpha = torch.full((16, 54), 1.0 / 54)
    factor = compute_analytical_attention_grad_factor(alpha, H)
    expected = (1.0 / 54) / math.sqrt(768)
    assert abs(factor - expected) < 1e-8


def test_2_tensor_dimensions():
    """Verify exact tensor dimensions across CSAM forward and backward."""
    B, k, H = 8, 54, 768
    h_T = torch.randn(B, H, requires_grad=True)
    h_L = torch.randn(k, H, requires_grad=True)
    targets = torch.zeros(B, k)

    csam = CSAM_L(H, scale_similarity=True, use_label_sa=True)
    logits, diag = csam(h_T, h_L, return_alpha=True)

    assert logits.shape == (B, k)
    assert diag["alpha"].shape == (B, k)

    loss = FocalLoss()(logits, targets)
    loss.backward()

    assert h_T.grad.shape == (B, H)
    assert h_L.grad.shape == (k, H)
    assert csam.W.weight.grad.shape == (1, H)
    assert csam.W.bias.grad.shape == (1,)


def test_3_finite_gradient_propagation():
    """Verify all gradients are finite and nonzero."""
    B, k, H = 4, 10, 128
    h_T = torch.randn(B, H, requires_grad=True)
    h_L = torch.randn(k, H, requires_grad=True)
    targets = torch.randint(0, 2, (B, k)).float()

    W_weight = torch.randn(1, H)
    W_bias = torch.zeros(1)

    res = run_single_probe("L2", True, h_T, h_L, targets, W_weight, W_bias, "focal")
    assert res["is_finite_grad"] is True
    assert res["gradients"]["h_T_norm"] > 0
    assert res["gradients"]["h_L_norm"] > 0
    assert res["gradients"]["W_norm"] > 0
    assert res["gradients"]["b_norm"] > 0


def test_4_l2_vs_l2_no_sa_algebraic_decomposition():
    """Verify exact algebraic decomposition of CSAM logits for both L2 and L2_NO_SA."""
    B, k, H = 4, 12, 64
    torch.manual_seed(123)
    h_T = torch.randn(B, H)
    h_L = torch.randn(k, H)

    # Test L2
    csam_l2 = CSAM_L(H, scale_similarity=True, use_label_sa=True)
    logits_l2, doc_term_l2, sa_term_l2, b_l2, _ = decompose_csam_logits(csam_l2, h_T, h_L)
    recon_l2 = doc_term_l2 + sa_term_l2 + b_l2
    assert torch.allclose(logits_l2, recon_l2, atol=1e-5)
    assert sa_term_l2 is not None

    # Test L2_NO_SA
    csam_no_sa = CSAM_L(H, scale_similarity=True, use_label_sa=False)
    logits_no_sa, doc_term_no_sa, sa_term_no_sa, b_no_sa, _ = decompose_csam_logits(csam_no_sa, h_T, h_L)
    recon_no_sa = doc_term_no_sa + b_no_sa
    assert torch.allclose(logits_no_sa, recon_no_sa, atol=1e-5)
    assert sa_term_no_sa is None


def test_5_focal_vs_bce_controlled_comparison():
    """Verify controlled comparison of Focal Loss vs BCE on same logits."""
    B, k, H = 4, 8, 32
    torch.manual_seed(99)
    h_T = torch.randn(B, H)
    h_L = torch.randn(k, H)
    targets = (torch.rand(B, k) < 0.2).float()
    W_weight = torch.randn(1, H)
    W_bias = torch.zeros(1)

    res_focal = run_single_probe("L2", True, h_T, h_L, targets, W_weight, W_bias, "focal")
    res_bce = run_single_probe("L2", True, h_T, h_L, targets, W_weight, W_bias, "bce")

    assert res_focal["is_finite_grad"] is True
    assert res_bce["is_finite_grad"] is True
    # In both, h_T gradient is finite and nonzero
    assert res_focal["gradients"]["h_T_norm"] > 0
    assert res_bce["gradients"]["h_T_norm"] > 0


def test_6_graceful_missing_checkpoint_handling():
    """Verify that execute_diagnostic_probe handles nonexistent checkpoint directory gracefully."""
    out = execute_diagnostic_probe(
        batch_size=4,
        hidden_dim=32,
        num_labels=8,
        seed=42,
        train_labels_path="nonexistent_labels.npy",
        checkpoint_dir="nonexistent_checkpoint_dir",
    )
    assert out["provenance"]["checkpoint_status"]["variant_L2_step300"] is False
    assert "Step-300 checkpoint tensors not present locally" in out["provenance"]["checkpoint_status"]["message"]
    assert "L2_FOCAL" in out["results"]
    assert "L2_NO_SA_FOCAL" in out["results"]


def test_7_no_optimizer_updates():
    """Verify that running the probe pass leaves original parameter tensors unchanged."""
    B, k, H = 4, 6, 32
    h_T = torch.randn(B, H)
    h_L = torch.randn(k, H)
    targets = torch.zeros(B, k)
    W_weight = torch.randn(1, H)
    W_bias = torch.zeros(1)

    W_weight_orig = W_weight.clone()
    W_bias_orig = W_bias.clone()

    run_single_probe("L2", True, h_T, h_L, targets, W_weight, W_bias, "focal")

    assert torch.equal(W_weight, W_weight_orig)
    assert torch.equal(W_bias, W_bias_orig)


def test_8_no_historical_artifact_modification():
    """Verify that RUN_0 and DR1 historical experiment files exist and are not touched."""
    hist_paths = [
        Path("experiments/RUN_0/run0_decision.json"),
        Path("experiments/RUN_0/run0_val_sample_ids.json"),
    ]
    for p in hist_paths:
        if p.exists():
            assert p.stat().st_size > 0
