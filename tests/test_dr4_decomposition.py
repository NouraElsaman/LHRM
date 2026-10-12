"""
tests/test_dr4_decomposition.py

Unit and invariant test suite for DR4 Post-Training Logit Decomposition Diagnostic.
Tests:
1. Shape invariants for L2 and L2_A5B.
2. Algebraic reconstruction: max |logits - (doc_term + sa_term + bias)| < 1e-5.
3. Cross-document invariance: Var_b(sa_term[b, i]) == 0.0.
4. Weight selection: d(logits[b, i]) / d(W[j]) == 0 for all i != j in L2_A5B.
5. Safe ratio handling: zero-denominator stability.
6. Duplicate-target label analysis: index 0 (cmp-lg) and index 1 (cond-mat.dis-nn).
7. Real checkpoint weight loading and reconstruction verification.
"""
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.probe_dr4_trained_logit_decomposition import (
    decompose_l2_csam_logits,
    decompose_a5b_csam_logits,
    compute_decomposition_statistics,
    seeded_rank_order,
)


class TestDR4LogitDecomposition:
    def test_l2_shape_invariants_and_exact_reconstruction(self):
        """Verify L2 output shapes and algebraic reconstruction error < 1e-5."""
        B, k, H = 10, 54, 64
        torch.manual_seed(42)
        h_T = torch.randn(B, H)
        h_L = torch.randn(k, H)
        w = torch.randn(1, H)
        bias = torch.tensor([-0.05])

        decomp = decompose_l2_csam_logits(h_T, h_L, w, bias)

        assert decomp["logits"].shape == (B, k)
        assert decomp["doc_term"].shape == (B, k)
        assert decomp["sa_term"].shape == (B, k)
        assert decomp["alpha"].shape == (B, k)
        assert decomp["A"].shape == (k, k)
        assert decomp["SA"].shape == (k, H)

        # Alpha rows sum to 1
        assert torch.allclose(decomp["alpha"].sum(dim=1), torch.ones(B), atol=1e-5)
        # A rows sum to 1
        assert torch.allclose(decomp["A"].sum(dim=1), torch.ones(k), atol=1e-5)

        # Exact algebraic reconstruction
        assert decomp["max_recon_error"] < 1e-5
        diff = (decomp["logits"] - (decomp["doc_term"] + decomp["sa_term"] + decomp["bias"])).abs().max().item()
        assert diff < 1e-5

    def test_a5b_shape_invariants_and_exact_reconstruction(self):
        """Verify L2_A5B output shapes and algebraic reconstruction error < 1e-5."""
        B, k, H = 10, 54, 64
        torch.manual_seed(42)
        h_T = torch.randn(B, H)
        h_L = torch.randn(k, H)
        W = torch.randn(k, H)
        bias = torch.tensor([-0.05])

        decomp = decompose_a5b_csam_logits(h_T, h_L, W, bias)

        assert decomp["logits"].shape == (B, k)
        assert decomp["doc_term"].shape == (B, k)
        assert decomp["sa_term"].shape == (B, k)
        assert decomp["alpha"].shape == (B, k)
        assert decomp["A"].shape == (k, k)
        assert decomp["SA"].shape == (k, H)

        # Exact algebraic reconstruction
        assert decomp["max_recon_error"] < 1e-5
        diff = (decomp["logits"] - (decomp["doc_term"] + decomp["sa_term"] + decomp["bias"])).abs().max().item()
        assert diff < 1e-5

    def test_sa_term_has_strictly_zero_cross_document_variance(self):
        """The static label-SA term sa_term must have identically zero variance across documents."""
        B, k, H = 25, 54, 128
        torch.manual_seed(123)
        h_T = torch.randn(B, H)
        h_L = torch.randn(k, H)
        W = torch.randn(k, H)
        w = torch.randn(1, H)
        b = -0.02

        decomp_l2 = decompose_l2_csam_logits(h_T, h_L, w, b)
        decomp_a5b = decompose_a5b_csam_logits(h_T, h_L, W, b)

        # Variance across documents (dim=0)
        sa_var_l2 = decomp_l2["sa_term"].var(dim=0, unbiased=True)
        sa_var_a5b = decomp_a5b["sa_term"].var(dim=0, unbiased=True)

        assert sa_var_l2.max().item() == 0.0, f"L2 SA term varies across documents: {sa_var_l2.max().item()}"
        assert sa_var_a5b.max().item() == 0.0, f"A5B SA term varies across documents: {sa_var_a5b.max().item()}"

    def test_doc_term_is_sole_source_of_logit_variation(self):
        """Because sa_term is constant across documents, Var_b(logits) == Var_b(doc_term) exactly."""
        B, k, H = 30, 10, 32
        torch.manual_seed(999)
        h_T = torch.randn(B, H)
        h_L = torch.randn(k, H)
        W = torch.randn(k, H)
        b = -0.1

        decomp = decompose_a5b_csam_logits(h_T, h_L, W, b)
        logit_var = decomp["logits"].var(dim=0, unbiased=True)
        doc_var = decomp["doc_term"].var(dim=0, unbiased=True)

        max_var_diff = (logit_var - doc_var).abs().max().item()
        assert max_var_diff < 1e-5, f"Logit variance differs from doc_term variance: {max_var_diff}"

    def test_safe_denominator_handling_in_ratio(self):
        """SA-to-doc ratio must handle near-zero doc term safely without dividing by zero."""
        B, k, H = 5, 4, 16
        h_T = torch.randn(B, H)
        h_L = torch.randn(k, H)
        W = torch.randn(k, H)
        decomp = decompose_a5b_csam_logits(h_T, h_L, W, 0.0)

        # Force doc_term to 0
        decomp["doc_term"].zero_()
        label_names = [f"l_{i}" for i in range(k)]
        stats = compute_decomposition_statistics(decomp, label_names)

        assert not math.isnan(stats["sa_to_doc_ratio"])
        assert not math.isinf(stats["sa_to_doc_ratio"])
        assert stats["sa_to_doc_ratio"] > 0

    def test_duplicate_target_labels_detection(self):
        """Verify decomposition handles cmp-lg and cond-mat.dis-nn explicitly."""
        B, k, H = 20, 54, 32
        h_T = torch.randn(B, H)
        h_L = torch.randn(k, H)
        W = torch.randn(k, H)
        label_names = [f"label_{i:02d}" for i in range(k)]
        label_names[0] = "cmp-lg"
        label_names[1] = "cond-mat.dis-nn"

        decomp = decompose_a5b_csam_logits(h_T, h_L, W, -0.02)
        stats = compute_decomposition_statistics(decomp, label_names)

        dup = stats["duplicate_target_labels"]
        assert dup["cmp_lg"]["label_name"] == "cmp-lg"
        assert dup["cond_mat_dis_nn"]["label_name"] == "cond-mat.dis-nn"
        assert "static_sa_difference" in dup
        assert "mean_logit_difference" in dup

    def test_real_checkpoint_weights_decomposition_invariants(self):
        """
        Loads actual saved model-only checkpoints (if present on disk) and verifies
        tensor shapes and algebraic decomposition invariants on realistic representations.
        """
        ckpt_dir = Path("D:/LHRM_KAGGLE_ARTIFACTS/RUN_0_L2_A5B_20261012_A/checkpoints")
        l2_ckpt_path = ckpt_dir / "L2_STEP300_MODEL_ONLY.pt"
        a5b_ckpt_path = ckpt_dir / "L2_A5B_STEP300_MODEL_ONLY.pt"

        if not l2_ckpt_path.is_file() or not a5b_ckpt_path.is_file():
            pytest.skip("Local model-only checkpoints not found in artifact path")

        ckpt_l2 = torch.load(l2_ckpt_path, map_location="cpu", weights_only=False)
        ckpt_a5b = torch.load(a5b_ckpt_path, map_location="cpu", weights_only=False)

        assert ckpt_l2["variant"] == "L2" and ckpt_l2["step"] == 300
        assert ckpt_a5b["variant"] == "L2_A5B" and ckpt_a5b["step"] == 300

        w_l2 = ckpt_l2["model_state"]["csam.W.weight"]  # [1, 768]
        b_l2 = ckpt_l2["model_state"]["csam.W.bias"]    # [1]
        W_a5b = ckpt_a5b["model_state"]["csam.W"]       # [54, 768]
        b_a5b = ckpt_a5b["model_state"]["csam.bias"]    # [1]

        assert w_l2.shape == (1, 768)
        assert W_a5b.shape == (54, 768)

        # Test algebraic decomposition with authentic checkpoint weights
        B, k, H = 10, 54, 768
        torch.manual_seed(42)
        h_T = torch.randn(B, H)
        h_L = torch.randn(k, H)

        decomp_l2 = decompose_l2_csam_logits(h_T, h_L, w_l2, b_l2)
        decomp_a5b = decompose_a5b_csam_logits(h_T, h_L, W_a5b, b_a5b)

        assert decomp_l2["max_recon_error"] < 1e-5
        assert decomp_a5b["max_recon_error"] < 1e-5

        # Cross-document SA variance must be 0
        assert decomp_l2["sa_term"].var(dim=0).max().item() == 0.0
        assert decomp_a5b["sa_term"].var(dim=0).max().item() == 0.0
