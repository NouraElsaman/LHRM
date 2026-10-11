"""
Unit and integration tests for Variant L2_A5B (CSAM with unshared per-label classifier).

DR4 Protocol Requirements:
1. Output shape is [B, 54].
2. Correct per-label weight selection: d(logits[b, i]) / d(W[j]) == 0 for all i != j.
3. No cross-label readout leakage.
4. Exact L2 attention architecture retained (scaled similarity S/sqrt(H) and label-label SA A_ij).
5. Step-0 baseline-logit parity after controlled initialization: logits match L2 within float tolerance (< 1e-6).
6. Identical alpha attention and label self-attention matrices at step 0.
7. Label-specific classifier gradients are independent: non-uniform loss produces distinct gradient vectors.
8. Finite losses and gradients under backward pass.
9. Existing variants (L, L2, L2_NO_SA, B, R) remain completely unchanged.
10. Fail-closed guards protect historical run IDs.
"""
import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from lhrm_baseline.model import (
    CSAM_L,
    CSAM_L_A5B,
    CSAM_R,
    LHRMModel,
    init_a5b_from_l2,
)
from transformers import BertConfig, BertModel


@pytest.fixture
def tiny_bert():
    """Create a deterministic tiny BertModel for fast CPU testing."""
    config = BertConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=128,
    )
    torch.manual_seed(42)
    return BertModel(config)


class TestVariantL2A5B:
    def test_output_shape(self):
        """CSAM_L_A5B produces [B, k] logits."""
        B, k, H = 4, 54, 64
        csam = CSAM_L_A5B(hidden=H, num_labels=k, scale_similarity=True)
        h_T = torch.randn(B, H)
        h_L = torch.randn(k, H)
        logits, diag = csam(h_T, h_L)
        assert logits.shape == (B, k), f"Expected {(B, k)}, got {logits.shape}"
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_per_label_weight_selection_and_no_leakage(self):
        """
        Verify mathematical property:
        logits[b, i] = dot(W[i], h_final[b, i]) + b
        Therefore:
        d(logits[b, i]) / d(W[j]) == 0 for all i != j.
        """
        B, k, H = 2, 5, 16
        csam = CSAM_L_A5B(hidden=H, num_labels=k, scale_similarity=True)
        h_T = torch.randn(B, H, requires_grad=False)
        h_L = torch.randn(k, H, requires_grad=False)

        for target_label in range(k):
            csam.zero_grad()
            logits, _ = csam(h_T, h_L)
            # Differentiate scalar logit for target_label only
            target_logit = logits[0, target_label]
            target_logit.backward(retain_graph=True)

            grad_W = csam.W.grad  # [k, H]
            assert grad_W is not None
            # Gradient for target_label row must be non-zero
            assert grad_W[target_label].abs().sum() > 1e-6, f"Label {target_label} weight grad should be non-zero"
            # Gradient for all other rows must be EXACTLY zero (no cross-label leakage)
            for other_label in range(k):
                if other_label != target_label:
                    assert grad_W[other_label].abs().sum() == 0.0, (
                        f"Leakage detected: d(logits[{target_label}])/d(W[{other_label}]) != 0"
                    )

    def test_exact_l2_attention_architecture_retained(self):
        """
        Verify that L2_A5B uses:
        - Scaled document-label cross-attention S/sqrt(H).
        - Label-label self-attention A_ij (Eq. 6-7).
        """
        B, k, H = 3, 6, 32
        csam = CSAM_L_A5B(hidden=H, num_labels=k, scale_similarity=True, use_label_sa=True)
        assert csam.scale_similarity is True
        assert csam.use_label_sa is True

        h_T = torch.randn(B, H)
        h_L = torch.randn(k, H)
        logits, diag = csam(h_T, h_L, return_alpha=True)

        alpha = diag["alpha"]
        assert alpha is not None
        assert alpha.shape == (B, k)
        # Softmax rows must sum to 1
        assert torch.allclose(alpha.sum(dim=-1), torch.ones(B), atol=1e-5)

        # Verify that disabling SA produces different logits (confirming SA is active)
        csam_no_sa = CSAM_L_A5B(hidden=H, num_labels=k, scale_similarity=True, use_label_sa=False)
        csam_no_sa.W.data.copy_(csam.W.data)
        csam_no_sa.bias.data.copy_(csam.bias.data)
        logits_no_sa, _ = csam_no_sa(h_T, h_L)
        assert not torch.allclose(logits, logits_no_sa, atol=1e-3), (
            "L2_A5B must actively use label-label self-attention"
        )

    def test_step0_parity_with_l2_controlled_initialization(self):
        """
        Verify controlled step-0 initialization parity:
        When A5b unshared rows W[i] are set to the shared L2 weight W,
        logits, alpha, and losses match L2 within float tolerance (< 1e-6).
        """
        B, k, H = 4, 10, 64
        torch.manual_seed(123)

        l2_csam = CSAM_L(hidden=H, scale_similarity=True, use_label_sa=True)
        a5b_csam = CSAM_L_A5B(hidden=H, num_labels=k, scale_similarity=True, use_label_sa=True)

        # Controlled copy: shared W [1, H] -> unshared W [k, H]
        a5b_csam.W.data.copy_(l2_csam.W.weight.data.repeat(k, 1))
        a5b_csam.bias.data.copy_(l2_csam.W.bias.data)

        h_T = torch.randn(B, H)
        h_L = torch.randn(k, H)

        l2_logits, l2_diag = l2_csam(h_T, h_L, return_alpha=True)
        a5b_logits, a5b_diag = a5b_csam(h_T, h_L, return_alpha=True)

        max_logit_diff = (l2_logits - a5b_logits).abs().max().item()
        assert max_logit_diff < 1e-6, f"Step-0 logit diff {max_logit_diff} exceeds float tolerance"

        max_alpha_diff = (l2_diag["alpha"] - a5b_diag["alpha"]).abs().max().item()
        assert max_alpha_diff < 1e-6, f"Step-0 alpha diff {max_alpha_diff} exceeds float tolerance"

        # Verify loss parity under Focal Loss
        targets = torch.randint(0, 2, (B, k)).float()
        from lhrm_baseline.losses import get_loss
        loss_fn = get_loss("focal", focal_gamma=2.0, focal_alpha=0.25)

        l2_loss = loss_fn(l2_logits, targets).item()
        a5b_loss = loss_fn(a5b_logits, targets).item()
        assert abs(l2_loss - a5b_loss) < 1e-6, f"Step-0 loss diff {abs(l2_loss - a5b_loss)} exceeds tolerance"

    def test_init_a5b_from_l2_utility(self, tiny_bert):
        """Test the init_a5b_from_l2 utility on full LHRMModel instances."""
        k = 12
        l2_model = LHRMModel(tiny_bert, num_labels=k, variant="L2")

        # Create separate a5b model with different seed
        torch.manual_seed(999)
        config = tiny_bert.config
        fresh_bert = BertModel(config)
        a5b_model = LHRMModel(fresh_bert, num_labels=k, variant="L2_A5B")

        # Before controlled init, weights differ
        assert not torch.allclose(a5b_model.csam.W[0], l2_model.csam.W.weight[0])

        # Apply controlled init
        init_a5b_from_l2(l2_model, a5b_model)

        # After controlled init, each row of A5b W equals L2 W.weight
        for i in range(k):
            assert torch.allclose(a5b_model.csam.W[i], l2_model.csam.W.weight[0], atol=1e-7)
        assert torch.allclose(a5b_model.csam.bias, l2_model.csam.W.bias, atol=1e-7)

        # Forward pass parity (eval mode disables dropout for deterministic check)
        l2_model.eval()
        a5b_model.eval()
        doc_ids = torch.randint(0, 100, (2, 16))
        doc_mask = torch.ones(2, 16)
        label_ids = torch.randint(0, 100, (k, 8))
        label_mask = torch.ones(k, 8)

        with torch.no_grad():
            l2_logits, _ = l2_model(doc_ids, doc_mask, label_ids, label_mask)
            a5b_logits, _ = a5b_model(doc_ids, doc_mask, label_ids, label_mask)

        assert (l2_logits - a5b_logits).abs().max().item() < 1e-5

    def test_gradient_independence_across_labels(self):
        """
        Verify that after identical initialization, training on non-uniform label targets
        produces independent gradient vectors across the different rows of W.
        """
        B, k, H = 4, 8, 32
        csam = CSAM_L_A5B(hidden=H, num_labels=k, scale_similarity=True)
        h_T = torch.randn(B, H)
        h_L = torch.randn(k, H)

        logits, _ = csam(h_T, h_L)
        # Non-uniform loss: label 0 has positive target, label 1 has negative target
        targets = torch.zeros(B, k)
        targets[:, 0] = 1.0

        loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss.backward()

        grad_W = csam.W.grad
        assert grad_W is not None
        assert torch.isfinite(grad_W).all()

        # Row 0 and Row 1 gradients must NOT be identical
        diff_0_1 = (grad_W[0] - grad_W[1]).abs().max().item()
        assert diff_0_1 > 1e-4, "Label gradient rows should diverge under non-uniform loss"

    def test_existing_variants_intact(self, tiny_bert):
        """Verify that existing variants L, L2, L2_NO_SA, B, R initialize and behave unchanged."""
        k = 8
        m_l = LHRMModel(tiny_bert, num_labels=k, variant="L")
        assert isinstance(m_l.csam, CSAM_L)
        assert m_l.csam.scale_similarity is False

        m_l2 = LHRMModel(tiny_bert, num_labels=k, variant="L2")
        assert isinstance(m_l2.csam, CSAM_L)
        assert m_l2.csam.scale_similarity is True

        m_no_sa = LHRMModel(tiny_bert, num_labels=k, variant="L2_NO_SA")
        assert isinstance(m_no_sa.csam, CSAM_L)
        assert m_no_sa.csam.use_label_sa is False

        m_r = LHRMModel(tiny_bert, num_labels=k, variant="R")
        assert isinstance(m_r.csam, CSAM_R)

        m_b = LHRMModel(tiny_bert, num_labels=k, variant="B")
        assert isinstance(m_b.classifier, nn.Linear)

    def test_protected_run_ids(self):
        """Verify that historical experiment IDs cannot be overwritten."""
        from run_baseline import cmd_run0
        import argparse

        protected = [
            "RUN_0",
            "DR1",
            "DR2",
            "RUN_0_P1F_EXT1",
            "RUN_0_L2_NO_SA_ABLATION_20261011_A",
            "RUN_0_L2_NO_SA_REPLAY_20261011_B",
        ]
        for prot_id in protected:
            args = argparse.Namespace(
                config="configs/final/aapd_baseline.yaml",
                run_id=prot_id,
                steps=300,
                candidates="L2,L2_A5B",
                extended=False,
            )
            with pytest.raises(SystemExit):
                cmd_run0(args)

    def test_notebook_and_bundle_integrity(self):
        """Verify DR4 Kaggle notebook and bundle exist with matching SHA-256."""
        import json, os, hashlib
        import pytest
        from run_baseline import ROOT

        nb_path = os.path.join(ROOT, "notebooks", "lhrm_run0_l2_a5b_kaggle.ipynb")
        bundle_path = os.path.join(ROOT, "bundles", "lhrm_gpu_bundle_l2_a5b_20261012.zip")

        assert os.path.isfile(nb_path), f"Missing notebook: {nb_path}"
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        assert nb.get("nbformat") == 4
        assert len(nb["cells"]) >= 7
        assert "RUN_0_L2_A5B_20261012_A" in json.dumps(nb)

        if not os.path.isfile(bundle_path):
            pytest.skip("Bundle zip not present in this workspace (GPU bundles are not committed to git)")

        # Compute bundle SHA
        h = hashlib.sha256()
        with open(bundle_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        bundle_sha = h.hexdigest()

        assert bundle_sha in json.dumps(nb), "Bundle SHA mismatch in notebook JSON"

    def test_bundle_contents_quarantine(self):
        """Verify GPU bundle does not contain Test split or heavy model weights."""
        import os, zipfile
        import pytest
        from run_baseline import ROOT

        bundle_path = os.path.join(ROOT, "bundles", "lhrm_gpu_bundle_l2_a5b_20261012.zip")
        if not os.path.isfile(bundle_path):
            pytest.skip("Bundle zip not present in this workspace (GPU bundles are not committed to git)")

        with zipfile.ZipFile(bundle_path, "r") as z:
            names = z.namelist()

        # Required files present
        assert "run_baseline.py" in names
        assert "src/lhrm_baseline/model.py" in names
        assert "configs/final/aapd_baseline.yaml" in names
        assert "docs/experiments/DR4_A5B_PREREGISTRATION.md" in names

        # Strictly quarantine: no .pt checkpoints, no test split files
        for name in names:
            assert not name.endswith(".pt"), f"Bundle contains checkpoint: {name}"
            assert "test_" not in name.lower(), f"Bundle contains test file: {name}"

    def test_preflight_crosscheck_references(self):
        """Verify L2_A5B is registered in _CROSSCHECK_REFS and local crosscheck JSON."""
        import json, os
        from run_baseline import ROOT, _CROSSCHECK_REFS

        assert "L2_A5B" in _CROSSCHECK_REFS
        ref = _CROSSCHECK_REFS["L2_A5B"]
        assert ref["prior_p_at_1_hits"] == (63, 0)
        assert ref["alpha_one_hot_frac"] == (0.0, 0.0)

        crosscheck_json = os.path.join(ROOT, "docs", "phase1", "run0_local_step0_crosscheck.json")
        with open(crosscheck_json) as f:
            data = json.load(f)
        assert "L2_A5B" in data["variants"]

    def test_baseline_learning_rate_is_2e5(self):
        """Verify AAPD baseline configuration specifies learning_rate == 2e-5 (NOT 3e-5)."""
        import os
        from run_baseline import ROOT, load_yaml_config

        cfg_path = os.path.join(ROOT, "configs", "final", "aapd_baseline.yaml")
        cfg = load_yaml_config(cfg_path)
        lr = float(cfg.get("learning_rate", 0.0))
        assert abs(lr - 2e-5) < 1e-9, f"Expected lr=2e-5, got {lr}"

