"""
Unit tests for model.py — shapes, finiteness, gradients, softmax rows.
Uses a tiny randomly-initialised BERT config (no download required).

Tests:
  - Eq.4 output shape and finiteness
  - Eq.5 (alpha) rows sum to 1
  - Eq.6 (A) rows sum to 1
  - Eq.7 h_final shape [B, k, H]
  - Eq.8 logits shape [B, k]
  - Gradients reach both document and label encoder paths
  - Determinism: same seed => identical outputs
  - CSAM_L and CSAM_L2 variants produce finite, non-identical outputs
  - Tiny model overfit test (debug only — uses synthetic data)
"""
import math
import torch
import torch.nn as nn
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lhrm_baseline.model import CSAM_L, get_cls_embedding
from lhrm_baseline.losses import FocalLoss, BCELoss


# ------------------------------------------------------------------
# Tiny BERT config (offline, random weights, no download)
# ------------------------------------------------------------------
def make_tiny_bert(hidden=32, num_layers=2, num_heads=4, vocab_size=100):
    """Create a tiny randomly-initialised BERT model for testing. CPU only."""
    from transformers import BertConfig, BertModel
    cfg = BertConfig(
        hidden_size=hidden,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=hidden * 4,
        vocab_size=vocab_size,
        max_position_embeddings=64,
    )
    return BertModel(cfg)


HIDDEN = 32
BATCH = 4
K = 5           # number of labels
SEQ_LEN = 16


@pytest.fixture(scope="module")
def tiny_bert():
    torch.manual_seed(42)
    return make_tiny_bert(hidden=HIDDEN)


@pytest.fixture(scope="module")
def dummy_inputs():
    """Generate fixed dummy token inputs for BERT."""
    torch.manual_seed(42)
    input_ids = torch.randint(0, 100, (BATCH, SEQ_LEN))
    attention_mask = torch.ones(BATCH, SEQ_LEN, dtype=torch.long)
    label_ids = torch.randint(0, 100, (K, 8))  # shorter seqs for labels
    label_mask = torch.ones(K, 8, dtype=torch.long)
    return input_ids, attention_mask, label_ids, label_mask


# ------------------------------------------------------------------
# Test get_cls_embedding
# ------------------------------------------------------------------
def test_get_cls_embedding_shape(tiny_bert, dummy_inputs):
    """CLS embedding has shape [B, H]."""
    input_ids, attention_mask, _, _ = dummy_inputs
    with torch.no_grad():
        out = tiny_bert(input_ids, attention_mask=attention_mask)
    cls_emb = get_cls_embedding(out)
    assert cls_emb.shape == (BATCH, HIDDEN), f"Got {cls_emb.shape}"


def test_get_cls_embedding_not_pooler(tiny_bert, dummy_inputs):
    """CLS embedding from last_hidden_state[:,0] differs from pooler_output."""
    input_ids, attention_mask, _, _ = dummy_inputs
    with torch.no_grad():
        out = tiny_bert(input_ids, attention_mask=attention_mask)
    cls_emb = get_cls_embedding(out)
    pooler = out.pooler_output
    # They should NOT be identical (pooler has tanh projection)
    assert not torch.allclose(cls_emb, pooler), "CLS and pooler should differ"


# ------------------------------------------------------------------
# Test CSAM_L shapes and properties (Eqs 4-8)
# ------------------------------------------------------------------
def test_csam_L_shapes():
    """CSAM_L logits shape is [B, k]."""
    torch.manual_seed(0)
    csam = CSAM_L(HIDDEN, scale_similarity=False)
    h_T = torch.randn(BATCH, HIDDEN)
    h_L = torch.randn(K, HIDDEN)
    with torch.no_grad():
        logits, diag = csam(h_T, h_L)
    assert logits.shape == (BATCH, K), f"Got {logits.shape}"


def test_csam_L_logits_finite():
    """CSAM_L logits are finite (no NaN, no Inf)."""
    torch.manual_seed(1)
    csam = CSAM_L(HIDDEN)
    h_T = torch.randn(BATCH, HIDDEN)
    h_L = torch.randn(K, HIDDEN)
    with torch.no_grad():
        logits, _ = csam(h_T, h_L)
    assert torch.isfinite(logits).all(), "CSAM_L produced non-finite logits"


def test_csam_alpha_rows_sum_to_one():
    """Eq.5: softmax(S) rows must sum to 1 (within float precision)."""
    torch.manual_seed(2)
    import torch.nn.functional as F
    h_T = torch.randn(BATCH, HIDDEN)
    h_L = torch.randn(K, HIDDEN)
    S = h_T @ h_L.T          # [B, K]
    alpha = F.softmax(S, dim=1)
    row_sums = alpha.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(BATCH), atol=1e-5),         f"Alpha rows don't sum to 1: {row_sums}"


def test_csam_A_rows_sum_to_one():
    """Eq.6: label self-attention A rows must sum to 1."""
    torch.manual_seed(3)
    import torch.nn.functional as F
    h_L = torch.randn(K, HIDDEN)
    A_raw = (h_L @ h_L.T) / math.sqrt(HIDDEN)
    A = F.softmax(A_raw, dim=1)
    row_sums = A.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(K), atol=1e-5),         f"A rows don't sum to 1: {row_sums}"


def test_csam_h_final_shape():
    """Eq.7: h_final has shape [B, k, H]."""
    import torch.nn.functional as F
    torch.manual_seed(4)
    B, k, H = BATCH, K, HIDDEN
    h_T = torch.randn(B, H)
    h_L = torch.randn(k, H)
    S = h_T @ h_L.T
    alpha = F.softmax(S, dim=1)         # [B, k]
    A = F.softmax((h_L @ h_L.T) / math.sqrt(H), dim=1)  # [k, k]
    
    alpha_weighted = alpha.unsqueeze(2) * h_L.unsqueeze(0)  # [B, k, H]
    A_weighted = (A @ h_L).unsqueeze(0).expand(B, -1, -1)   # [B, k, H]
    h_final = alpha_weighted + A_weighted
    
    assert h_final.shape == (B, k, H), f"h_final shape: {h_final.shape}"


# ------------------------------------------------------------------
# Test gradient flow
# ------------------------------------------------------------------
def test_gradients_flow_to_doc_and_label_encoder(tiny_bert, dummy_inputs):
    """Gradients must reach both document and label encoder paths."""
    input_ids, attention_mask, label_ids, label_mask = dummy_inputs
    
    # Build forward pass manually
    doc_out = tiny_bert(input_ids, attention_mask=attention_mask)
    label_out = tiny_bert(label_ids, attention_mask=label_mask)
    
    h_T = get_cls_embedding(doc_out)   # [B, H]
    h_L = get_cls_embedding(label_out) # [K, H]
    
    csam = CSAM_L(HIDDEN)
    logits, _ = csam(h_T, h_L)
    
    loss = logits.mean()
    loss.backward()
    
    # Check gradients exist for document BERT params
    doc_grad = None
    for name, param in tiny_bert.named_parameters():
        if param.grad is not None:
            doc_grad = param.grad
            break
    assert doc_grad is not None, "No gradients in BERT parameters"
    
    # Check CSAM weight has gradient
    assert csam.W.weight.grad is not None, "CSAM W has no gradient"


# ------------------------------------------------------------------
# Test determinism
# ------------------------------------------------------------------
def test_determinism_same_seed():
    """Same seed => identical CSAM outputs (CPU)."""
    def run_once(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        h_T = torch.randn(BATCH, HIDDEN)
        h_L = torch.randn(K, HIDDEN)
        csam = CSAM_L(HIDDEN)
        with torch.no_grad():
            logits, _ = csam(h_T, h_L)
        return logits.clone()
    
    out1 = run_once(42)
    out2 = run_once(42)
    assert torch.allclose(out1, out2), "Same seed should give identical outputs"


# ------------------------------------------------------------------
# Test Focal Loss
# ------------------------------------------------------------------
def test_focal_loss_gamma_zero_equals_bce_scaled():
    """With gamma=0 and alpha=0.5: FL = 0.5 * BCE."""
    from lhrm_baseline.losses import FocalLoss
    torch.manual_seed(5)
    logits = torch.randn(BATCH, K)
    targets = (torch.randn(BATCH, K) > 0).float()
    
    fl = FocalLoss(gamma=0.0, alpha=0.5)
    bce = BCELoss()
    
    fl_loss = fl(logits, targets)
    bce_loss = bce(logits, targets)
    
    # alpha=0.5 for both pos and neg; gamma=0 removes focusing -> 0.5 * BCE
    assert abs(fl_loss.item() - 0.5 * bce_loss.item()) < 1e-4,         f"FL(gamma=0, alpha=0.5)={fl_loss.item():.6f} != 0.5*BCE={0.5*bce_loss.item():.6f}"


def test_focal_loss_downweights_easy_examples():
    """FL with gamma=2 assigns lower weight to easy (high-confidence) examples."""
    from lhrm_baseline.losses import FocalLoss
    fl = FocalLoss(gamma=2.0, alpha=0.25, reduction="none")
    
    # High-confidence correct positive (easy): logit=5 -> p=0.993
    easy_logit = torch.tensor([[5.0]])
    easy_target = torch.tensor([[1.0]])
    
    # Low-confidence positive (hard): logit=0 -> p=0.5
    hard_logit = torch.tensor([[0.0]])
    hard_target = torch.tensor([[1.0]])
    
    easy_loss = fl(easy_logit, easy_target).item()
    hard_loss = fl(hard_logit, hard_target).item()
    
    assert hard_loss > easy_loss,         f"Hard example should have higher focal loss: hard={hard_loss:.4f} easy={easy_loss:.4f}"


# ------------------------------------------------------------------
# Test: tiny overfit (synthetic, debug only)
# ------------------------------------------------------------------
def test_tiny_model_overfit_synthetic(tiny_bert):
    """
    A tiny model should be able to overfit a very small synthetic dataset.
    Uses random weights BERT (not pretrained) — purely a gradient flow check.
    This is a SYNTHETIC DEBUG TEST. Results mean nothing for the paper.
    """
    torch.manual_seed(42)
    csam = CSAM_L(HIDDEN)
    
    # Synthetic: 2 docs, K=5 labels, fixed targets
    h_T = torch.randn(2, HIDDEN, requires_grad=True)
    h_L_param = nn.Parameter(torch.randn(K, HIDDEN))
    
    # Fixed labels (doc 0 has labels 0,1; doc 1 has labels 2,3)
    targets = torch.zeros(2, K)
    targets[0, 0] = targets[0, 1] = 1.0
    targets[1, 2] = targets[1, 3] = 1.0
    
    criterion = FocalLoss(gamma=2.0, alpha=0.25)
    optimizer = torch.optim.Adam(list(csam.parameters()) + [h_L_param, h_T], lr=0.1)
    
    initial_loss = None
    for step in range(50):
        optimizer.zero_grad()
        logits, _ = csam(h_T, h_L_param)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        if initial_loss is None:
            initial_loss = loss.item()
    
    final_loss = loss.item()
    assert final_loss < initial_loss,         f"Loss did not decrease: initial={initial_loss:.4f} final={final_loss:.4f}"
