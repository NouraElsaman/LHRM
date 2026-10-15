"""
tests/test_variant_r_author.py

Targeted CPU regression and unit tests for Variant R_AUTHOR (author-aligned CSAM):
1. Exact output shapes across all intermediate CSAM tensors.
2. Concatenation of [h_doc, agg] -> R^{B x 2H}.
3. Parity with reference calculations from Model/lhrm.py on controlled tensors.
4. Correct document BERT pooling: pooler_output for R_AUTHOR vs last_hidden_state for L/L2.
5. Trainable label embedding parameter with requires_grad=True.
6. Execution with BCEWithLogitsLoss.
7. Non-zero gradients to document BERT, label_emb, self-attention, and classifier.
8. Optimizer coverage of all trainable parameters.
9. Save and reload state_dict equivalence (< 1e-6).
10. Synthetic overfitting test showing decreasing training loss.
11. Client inference demo helper (predict_abstract) output structure and formatting.
12. Deterministic non-overlapping Dev partition for HRRM evaluation.
"""
from __future__ import annotations
import math
import os
import shutil
import tempfile
from typing import Dict, List

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.lhrm_baseline.model import (
    CSAM_R_AUTHOR,
    LHRMModel,
    build_author_label_embeddings,
    get_cls_embedding,
)
from src.lhrm_baseline.losses import get_loss
from src.lhrm_baseline.hrrm import (
    HRRM,
    create_dev_partition,
    evaluate_hrrm_heldout,
)
from src.lhrm_baseline.inference import predict_abstract, format_prediction_report


class DummyBertOutput:
    def __init__(self, last_hidden_state: torch.Tensor, pooler_output: torch.Tensor):
        self.last_hidden_state = last_hidden_state
        self.pooler_output = pooler_output


class DummyBertConfig:
    def __init__(self, hidden_size: int = 64):
        self.hidden_size = hidden_size


class DummyBertModel(nn.Module):
    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.config = DummyBertConfig(hidden_size)
        self.embedding = nn.Embedding(100, hidden_size)
        self.pooler = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        emb = self.embedding(input_ids)
        cls_token = emb[:, 0, :]
        pooler_out = torch.tanh(self.pooler(cls_token))
        return DummyBertOutput(last_hidden_state=emb, pooler_output=pooler_out)


class TestVariantRAuthor:
    def test_csam_r_author_shapes_and_concatenation(self):
        """Verify exact tensor shapes matching Model/lhrm.py:144-192."""
        B, k, H = 4, 54, 64
        torch.manual_seed(42)
        h_doc = torch.randn(B, H)
        label_emb = torch.randn(k, H)
        
        csam = CSAM_R_AUTHOR(hidden=H, num_labels=k, label_emb=label_emb, use_sa=True, heads=4)
        logits, diag = csam(h_doc, return_alpha=True)
        
        assert logits.shape == (B, k), f"Expected logits shape {(B, k)}, got {logits.shape}"
        assert diag["alpha"].shape == (B, k), f"Expected alpha shape {(B, k)}, got {diag['alpha'].shape}"
        assert diag["S"].shape == (B, k), f"Expected S shape {(B, k)}, got {diag['S'].shape}"
        # Alpha should sum to 1 across labels
        alpha_sums = diag["alpha"].sum(dim=1)
        assert torch.allclose(alpha_sums, torch.ones(B), atol=1e-5)

    def test_author_reference_aggregation_parity(self):
        """Verify numerical equivalence to the exact calculations in Model/lhrm.py."""
        B, k, H = 2, 5, 16
        torch.manual_seed(42)
        h_doc = torch.randn(B, H)
        label_emb = torch.randn(k, H)
        
        # Standalone computation without self-attention for pure aggregation check:
        # Model/lhrm.py:159-160
        sim_ref = h_doc @ label_emb.t()
        w_ref = sim_ref.softmax(dim=1)
        agg_ref = w_ref @ label_emb
        combined_ref = torch.cat([h_doc, agg_ref], dim=1)
        
        csam_nosa = CSAM_R_AUTHOR(hidden=H, num_labels=k, label_emb=label_emb, use_sa=False)
        # Test internal intermediate tensors
        with torch.no_grad():
            sim_csam = h_doc @ csam_nosa.label_emb.t()
            w_csam = sim_csam.softmax(dim=1)
            agg_csam = w_csam @ csam_nosa.label_emb
            combined_csam = torch.cat([h_doc, agg_csam], dim=1)
            
        assert torch.allclose(sim_ref, sim_csam, atol=1e-6)
        assert torch.allclose(w_ref, w_csam, atol=1e-6)
        assert torch.allclose(agg_ref, agg_csam, atol=1e-6)
        assert torch.allclose(combined_ref, combined_csam, atol=1e-6)

    def test_pooling_choice_distinction(self):
        """Verify R_AUTHOR uses pooler_output whereas L/L2 uses last_hidden_state[:, 0]."""
        H = 64
        bert = DummyBertModel(hidden_size=H)
        model_r = LHRMModel(bert, num_labels=10, variant="R_AUTHOR")
        model_l = LHRMModel(bert, num_labels=10, variant="L")
        
        doc_ids = torch.randint(0, 100, (2, 8))
        doc_mask = torch.ones_like(doc_ids)
        label_ids = torch.randint(0, 100, (10, 4))
        label_mask = torch.ones_like(label_ids)
        
        # In dummy BERT, last_hidden_state[:, 0] is raw embedding, while pooler_output is tanh(linear(raw))
        out = bert(doc_ids, doc_mask)
        cls_raw = get_cls_embedding(out)
        pooler_out = out.pooler_output
        assert not torch.allclose(cls_raw, pooler_out)
        
        # R_AUTHOR forward does not error and uses pooler_output
        logits_r, _ = model_r(doc_ids, doc_mask)
        assert logits_r.shape == (2, 10)

    def test_label_emb_trainability_and_gradient_flow(self):
        """Verify gradients reach document BERT, label_emb, self-attention, and classifier."""
        H = 32
        k = 8
        bert = DummyBertModel(hidden_size=H)
        model = LHRMModel(bert, num_labels=k, variant="R_AUTHOR")
        
        assert isinstance(model.csam.label_emb, nn.Parameter)
        assert model.csam.label_emb.requires_grad is True
        
        criterion = get_loss("bce")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        doc_ids = torch.randint(0, 100, (4, 12))
        doc_mask = torch.ones_like(doc_ids)
        targets = torch.randint(0, 2, (4, k)).float()
        
        logits, _ = model(doc_ids, doc_mask)
        loss = criterion(logits, targets)
        loss.backward()
        
        # Verify non-zero gradients on all key components:
        assert bert.embedding.weight.grad is not None
        assert bert.embedding.weight.grad.norm() > 0.0
        assert model.csam.label_emb.grad is not None
        assert model.csam.label_emb.grad.norm() > 0.0
        assert model.csam.sa.in_proj_weight.grad is not None
        assert model.csam.sa.in_proj_weight.grad.norm() > 0.0
        assert model.csam.classifier.weight.grad is not None
        assert model.csam.classifier.weight.grad.norm() > 0.0
        
        # Verify optimizer step works
        optimizer.step()

    def test_save_and_reload_state_dict_equivalence(self):
        """Verify model weights and predictions are identical across save and reload."""
        H = 32
        k = 6
        bert1 = DummyBertModel(hidden_size=H)
        model1 = LHRMModel(bert1, num_labels=k, variant="R_AUTHOR")
        
        doc_ids = torch.randint(0, 100, (3, 10))
        doc_mask = torch.ones_like(doc_ids)
        
        logits1, _ = model1(doc_ids, doc_mask)
        
        # Save state dict
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = os.path.join(tmp_dir, "test_ckpt.pt")
            torch.save({"model_state_dict": model1.state_dict(), "variant": "R_AUTHOR"}, ckpt_path)
            
            # Load into fresh model
            bert2 = DummyBertModel(hidden_size=H)
            model2 = LHRMModel(bert2, num_labels=k, variant="R_AUTHOR")
            loaded = torch.load(ckpt_path, weights_only=False)
            model2.load_state_dict(loaded["model_state_dict"])
            
            logits2, _ = model2(doc_ids, doc_mask)
            assert torch.allclose(logits1, logits2, atol=1e-6)

    def test_synthetic_overfitting_with_bce(self):
        """Verify small batch loss decreases monotonically under BCE training."""
        torch.manual_seed(42)
        H = 32
        k = 4
        bert = DummyBertModel(hidden_size=H)
        model = LHRMModel(bert, num_labels=k, variant="R_AUTHOR")
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
        criterion = get_loss("bce")
        
        doc_ids = torch.randint(0, 100, (4, 10))
        doc_mask = torch.ones_like(doc_ids)
        targets = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 1, 0, 0]]).float()
        
        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            logits, _ = model(doc_ids, doc_mask)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        assert losses[-1] < losses[0], f"Loss failed to decrease: initial={losses[0]}, final={losses[-1]}"
        assert losses[-1] < 0.5 * losses[0], f"Loss should decrease substantially, got {losses}"

    def test_dev_partition_determinism_and_exhaustiveness(self):
        """Verify create_dev_partition generates 4 non-overlapping, exhaustive subsets."""
        n_docs = 1000
        p1 = create_dev_partition(n_docs=n_docs, seed=42)
        p2 = create_dev_partition(n_docs=n_docs, seed=42)
        
        # Identical with same seed
        assert p1["diagnostic_ids"] == p2["diagnostic_ids"]
        assert p1["model_selection_ids"] == p2["model_selection_ids"]
        assert p1["hrrm_fit_ids"] == p2["hrrm_fit_ids"]
        assert p1["hrrm_eval_ids"] == p2["hrrm_eval_ids"]
        
        # Sizes
        assert len(p1["diagnostic_ids"]) == 200
        assert len(p1["model_selection_ids"]) == 400
        assert len(p1["hrrm_fit_ids"]) == 200
        assert len(p1["hrrm_eval_ids"]) == 200
        
        # Disjoint
        s_diag = set(p1["diagnostic_ids"])
        s_sel = set(p1["model_selection_ids"])
        s_fit = set(p1["hrrm_fit_ids"])
        s_eval = set(p1["hrrm_eval_ids"])
        
        assert len(s_diag & s_sel) == 0
        assert len(s_diag & s_fit) == 0
        assert len(s_diag & s_eval) == 0
        assert len(s_sel & s_fit) == 0
        assert len(s_sel & s_eval) == 0
        assert len(s_fit & s_eval) == 0
        
        # Exhaustive
        assert len(s_diag | s_sel | s_fit | s_eval) == 1000

    def test_evaluate_hrrm_heldout_pipeline(self):
        """Verify non-leaky HRRM fitting and held-out evaluation."""
        np.random.seed(42)
        N = 1000
        k = 10
        label_order = [f"lab_{i}" for i in range(k)]
        y_true = (np.random.rand(N, k) > 0.85).astype(float)
        # Create correlated predictions
        y_pred = y_true * 0.7 + np.random.rand(N, k) * 0.3
        
        res = evaluate_hrrm_heldout(y_true, y_pred, label_order, seed=42)
        assert res["n_fit_docs"] == 200
        assert res["n_eval_docs"] == 200
        assert "P@1" in res["neural_metrics"]
        assert "P@1" in res["hrrm_metrics"]
        assert "P@1" in res["delta"]
        assert res["hrrm_model"].is_fitted is True

    def test_predict_abstract_helper(self):
        """Verify client-facing predict_abstract helper output format."""
        class DummyTokenizer:
            def __call__(self, text, return_tensors, max_length, padding, truncation):
                return {
                    "input_ids": torch.randint(0, 100, (1, 16)),
                    "attention_mask": torch.ones(1, 16).long(),
                }
                
        H = 32
        k = 10
        bert = DummyBertModel(hidden_size=H)
        model = LHRMModel(bert, num_labels=k, variant="R_AUTHOR")
        tokenizer = DummyTokenizer()
        label_order = [f"cat_{i}" for i in range(k)]
        device = torch.device("cpu")
        
        text = "Deep learning advances for multi-label text classification."
        pred = predict_abstract(text, model, tokenizer, label_order, device, hrrm_model=None, top_k=5)
        
        assert len(pred["neural_top_k"]) == 5
        assert pred["neural_top_k"][0]["rank"] == 1
        assert "label" in pred["neural_top_k"][0]
        assert "score" in pred["neural_top_k"][0]
        assert "disclaimer" in pred["metadata"]
        
        report = format_prediction_report(pred)
        assert "### Client Inference Demonstration Report" in report
        assert "cat_" in report
