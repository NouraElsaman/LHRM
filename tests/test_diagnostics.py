"""
tests/test_diagnostics.py
Phase 1F regression tests.

All tests run entirely locally (CPU, tiny random-config BERT, no downloads).
No GPU required. All assertions are structural / mathematical — never performance claims.

Tests:
  test_alpha_is_returned_and_read
  test_vacuous_v3_is_impossible
  test_v5_integer_tie
  test_prior_tiebreak_is_seeded_not_column_order
  test_v4_measured_in_fp32
  test_decision_l_viable
  test_decision_l_fails_l2_viable
  test_decision_neither_viable
  test_decision_v3_escalate
  test_decision_v2v5_only_extend
  test_decision_invalid_diagnostics
  test_artifact_completeness
  test_sha256sums_consistency
  test_alpha_missing_raises_diagnostic_error
  test_alpha_rows_sum_to_one
"""
import contextlib
import json
import math
import os
import struct
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from lhrm_baseline.diagnostics import (
    DiagnosticError,
    _validate_alpha,
    check_artifact_completeness,
    compute_diagnostics,
    evaluate_criteria,
    frequency_prior_hits_at_1,
    run0_decision,
    verify_sha256sums,
    REQUIRED_ARTIFACTS,
)


# ── Tiny model stubs ──────────────────────────────────────────────────────────

class _TinyCSAM(nn.Module):
    """Minimal CSAM that returns an alpha tensor (used to test the reader)."""

    def __init__(self, k: int = 4, H: int = 8):
        super().__init__()
        self.k = k
        self.H = H
        self.W = nn.Linear(H, 1, bias=True)
        self.scale_similarity = False  # mimic CSAM_L attribute

    def forward(self, h_T, h_L, return_alpha=False):
        S = h_T @ h_L.T                          # [B, k]
        alpha = F.softmax(S, dim=1)              # [B, k]
        A_raw = (h_L @ h_L.T) / math.sqrt(self.H)
        A = F.softmax(A_raw, dim=1)
        alpha_w = alpha.unsqueeze(2) * h_L.unsqueeze(0)
        A_w = (A @ h_L).unsqueeze(0).expand(h_T.size(0), -1, -1)
        h_final = alpha_w + A_w
        logits = self.W(h_final).squeeze(-1)
        diag = {
            "alpha": alpha.detach() if (not self.training or return_alpha) else None,
            "S":     S.detach()     if (not self.training or return_alpha) else None,
            "alpha_max_mean":    alpha.max(dim=1).values.mean().item(),
            "alpha_entropy_mean": -(alpha * (alpha + 1e-10).log()).sum(dim=1).mean().item(),
            "alpha_one_hot_frac":  (alpha.max(dim=1).values > 0.9).float().mean().item(),
        }
        return logits, diag


class _TinyModel(nn.Module):
    """Minimal LHRMModel-like stub with csam attribute."""

    def __init__(self, k=4, H=8):
        super().__init__()
        self.k = k
        self.H = H
        self.bert_emb = nn.Embedding(30522, H)
        self.csam = _TinyCSAM(k, H)

    def forward(self, doc_ids, doc_mask, label_ids, label_mask):
        h_T = self.bert_emb(doc_ids[:, 0])              # [B, H]
        h_L = self.bert_emb(label_ids[:, 0])            # [k, H]
        return self.csam(h_T, h_L)


class _TinyModelNoAlpha(nn.Module):
    """Model stub that always omits alpha from diag (simulates the old bug)."""

    def __init__(self, k=4, H=8):
        super().__init__()
        self.k = k
        self.H = H
        self.bert_emb = nn.Embedding(30522, H)
        self.csam = _TinyCSAM(k, H)

    def forward(self, doc_ids, doc_mask, label_ids, label_mask):
        h_T = self.bert_emb(doc_ids[:, 0])
        h_L = self.bert_emb(label_ids[:, 0])
        S = h_T @ h_L.T
        alpha = F.softmax(S, dim=1)
        W = nn.Linear(self.H, 1, bias=True)
        h_final = alpha.unsqueeze(2) * h_L.unsqueeze(0)
        logits = W(h_final).squeeze(-1)
        # BUG: alpha deliberately omitted / None
        diag = {"alpha": None, "S": None, "alpha_max_mean": 0.0,
                "alpha_entropy_mean": 0.0, "alpha_one_hot_frac": 0.0}
        return logits, diag


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_batch(B=3, seq_len=4, k=4):
    doc_ids  = torch.randint(0, 100, (B, seq_len))
    doc_mask = torch.ones(B, seq_len, dtype=torch.long)
    lab_ids  = torch.randint(0, 100, (k, seq_len))
    lab_mask = torch.ones(k, seq_len, dtype=torch.long)
    labels   = torch.zeros(B, k)
    for i in range(B):
        labels[i, i % k] = 1
    return doc_ids, doc_mask, lab_ids, lab_mask, labels


def _make_loader(B=6, seq_len=4, k=4):
    """Return a tiny DataLoader for compute_diagnostics."""
    from torch.utils.data import DataLoader, TensorDataset

    doc_ids  = torch.randint(0, 100, (B, seq_len))
    doc_mask = torch.ones(B, seq_len, dtype=torch.long)
    labels_t = torch.zeros(B, k)
    for i in range(B):
        labels_t[i, i % k] = 1

    class _DS(torch.utils.data.Dataset):
        def __len__(self): return B
        def __getitem__(self, i):
            return {"input_ids": doc_ids[i], "attention_mask": doc_mask[i],
                    "labels": labels_t[i]}

    def collate(batch):
        return {
            "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"]  for b in batch]),
            "labels":         torch.stack([b["labels"]          for b in batch]),
        }
    return DataLoader(_DS(), batch_size=2, shuffle=False, collate_fn=collate), labels_t.numpy()


# Patch forward_pass to route through the model stub
def _fp_patch(model, ids, msk, label_ids, label_mask):
    return model(ids, msk, label_ids, label_mask)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_alpha_is_returned_and_read(monkeypatch):
    """
    compute_diagnostics must receive a [B, k] alpha tensor with rows summing
    to 1 and mean_max_alpha >= 1/k.  Uses _TinyModel (eval mode).
    """
    import lhrm_baseline.diagnostics as diag_mod
    monkeypatch.setattr(diag_mod, "forward_pass", _fp_patch)
    k = 4; B = 6
    model = _TinyModel(k=k); model.eval()
    loader, val_labels = _make_loader(B=B, k=k)
    lab_ids  = torch.randint(0, 100, (k, 4))
    lab_mask = torch.ones(k, 4, dtype=torch.long)
    train_labels = np.zeros((10, k)); train_labels[:, 0] = 1

    result = compute_diagnostics(
        model=model, loader=loader,
        label_input_ids=lab_ids, label_attention_mask=lab_mask,
        val_labels=val_labels, device=torch.device("cpu"),
        train_labels=train_labels, n_labels=k, seed=42,
    )
    assert result["n_docs"] == B
    assert result["mean_max_alpha"] >= 1.0 / k - 1e-6, (
        f"mean_max_alpha={result['mean_max_alpha']:.6f} < 1/k={1/k:.6f}"
    )
    assert 0.0 <= result["alpha_one_hot_frac"] <= 1.0
    assert result["n_distinct_argmax"] >= 1


def test_alpha_missing_raises_diagnostic_error(monkeypatch):
    """
    If the model returns alpha=None in diag, compute_diagnostics must raise
    DiagnosticError — never silently default to zero.
    """
    import lhrm_baseline.diagnostics as diag_mod
    monkeypatch.setattr(diag_mod, "forward_pass", _fp_patch)
    k = 4; B = 4
    model = _TinyModelNoAlpha(k=k); model.eval()
    loader, val_labels = _make_loader(B=B, k=k)
    lab_ids  = torch.randint(0, 100, (k, 4))
    lab_mask = torch.ones(k, 4, dtype=torch.long)
    train_labels = np.zeros((10, k)); train_labels[:, 0] = 1

    with pytest.raises(DiagnosticError, match="diag\\['alpha'\\] is None"):
        compute_diagnostics(
            model=model, loader=loader,
            label_input_ids=lab_ids, label_attention_mask=lab_mask,
            val_labels=val_labels, device=torch.device("cpu"),
            train_labels=train_labels, n_labels=k, seed=42,
        )


def test_vacuous_v3_is_impossible():
    """
    Missing alpha → evaluate_criteria must return V3=False (UNMEASURED),
    not V3=True (the old vacuous bug where 0 < 0.9 was True).
    run0_decision must return INVALID_DIAGNOSTICS.
    """
    # Simulate a diagnostics dict that has no alpha (diagnostic_error set)
    bad_diag = {"step": 300, "diagnostic_error": "alpha is None",
                "nan_inf_count": 0}
    losses = [0.1] * 300

    crit = evaluate_criteria({}, bad_diag, losses, n_labels=54)
    assert crit["V3"] is False, "V3 must be False when alpha is unmeasured"
    assert crit["V3_value"] == "UNMEASURED"

    # Decision on a single candidate with diagnostic_error must be INVALID_DIAGNOSTICS
    decision = run0_decision({"L": {"diagnostic_error": "alpha is None"}})
    assert decision["status"] == "INVALID_DIAGNOSTICS"


def test_alpha_rows_sum_to_one():
    """_validate_alpha must pass for a valid softmax output and fail for garbage."""
    k = 54
    rng = np.random.default_rng(0)
    raw = rng.random((100, k))
    alpha = raw / raw.sum(axis=1, keepdims=True)   # valid rows
    _validate_alpha(alpha, k)  # must not raise

    # Bad: rows do not sum to 1
    bad = alpha.copy(); bad[0, 0] += 0.5
    with pytest.raises(DiagnosticError, match="do not sum to 1"):
        _validate_alpha(bad, k)

    # Bad: None
    with pytest.raises(DiagnosticError, match="alpha is None"):
        _validate_alpha(None, k)

    # Bad: mean_max < 1/k
    tiny = np.full((10, k), 1e-10, dtype=float)
    tiny[:] /= tiny.sum(axis=1, keepdims=True)
    # Even a valid softmax always has max >= 1/k; this path tests the guard
    # by forcing a low value (requires all rows uniform = exactly 1/k)
    uniform = np.ones((10, k), dtype=float) / k
    _validate_alpha(uniform, k)   # must pass (max = 1/k exactly)


def test_v5_integer_tie():
    """
    F2 fix: V5 must use integer hit comparison (strict >).
    63/200 vs 63/200: V5 False even if float32 representation differs.
    64/200 vs 63/200: V5 True.
    """
    # Case 1: tie — both 63 hits
    diag = {
        "p_at_1_hits": 63, "p_at_1": 63 / 200,
        "prior_p_at_1_hits": 63, "prior_p_at_1": 63 / 200,
        "alpha_one_hot_frac": 0.5, "logit_std_frac": 0.8,
        "nan_inf_count": 0,
    }
    losses = [0.2] * 50 + [0.15] * 50
    crit = evaluate_criteria({}, diag, losses, n_labels=54)
    assert crit["V5"] is False, (
        "V5 must be False when cand_hits == prior_hits (63 == 63)"
    )

    # Case 2: float32 would give 0.315 > 0.3149999... = True, but integer is also True
    # (this is still a pass; test that integer version agrees)
    diag2 = {**diag, "p_at_1_hits": 64, "prior_p_at_1_hits": 63}
    crit2 = evaluate_criteria({}, diag2, losses, n_labels=54)
    assert crit2["V5"] is True, "V5 must be True when cand_hits=64 > prior_hits=63"


def test_prior_tiebreak_is_seeded_not_column_order():
    """
    F2 fix: prior P@1 must use seeded tie-break.
    Permuting label columns must not change the hit count
    (because the seeded rng adapts to label order, not column index).
    """
    rng = np.random.default_rng(7)
    k = 10; N_train = 100; N_val = 50
    train_labels = (rng.random((N_train, k)) > 0.8).astype(float)
    val_labels   = (rng.random((N_val,   k)) > 0.5).astype(float)

    hits_orig = frequency_prior_hits_at_1(train_labels, val_labels, seed=42)

    # Permute columns in both train and val
    perm = rng.permutation(k)
    train_perm = train_labels[:, perm]
    val_perm   = val_labels[:, perm]
    hits_perm  = frequency_prior_hits_at_1(train_perm, val_perm, seed=42)

    # The same physical label (different column index) is selected in both cases,
    # so the hit count on the correspondingly-permuted val must match.
    assert hits_orig == hits_perm, (
        f"Prior hits changed after column permutation: {hits_orig} -> {hits_perm}. "
        "This indicates column-index tie-break, not seeded tie-break."
    )


def test_v4_measured_in_fp32():
    """
    F3 fix: compute_diagnostics must NOT use autocast.
    Verified by wrapping forward_pass with a tracer that records
    torch.is_autocast_enabled() at call time — must always be False.
    Also checks that result['precision_mode'] == 'fp32_no_autocast'.
    """
    import lhrm_baseline.diagnostics as diag_mod
    import sys

    autocast_active_during_calls = []
    _orig_fp = diag_mod.forward_pass

    def _traced_fp(model, ids, msk, label_ids, label_mask):
        autocast_active_during_calls.append(torch.is_autocast_enabled())
        return _fp_patch(model, ids, msk, label_ids, label_mask)

    k = 4; B = 4
    model = _TinyModel(k=k); model.eval()
    loader, val_labels = _make_loader(B=B, k=k)
    lab_ids  = torch.randint(0, 100, (k, 4))
    lab_mask = torch.ones(k, 4, dtype=torch.long)
    train_labels = np.zeros((10, k)); train_labels[:, 0] = 1

    # Patch via module attribute (compute_diagnostics uses the module-level name)
    diag_mod.forward_pass = _traced_fp
    try:
        result = compute_diagnostics(
            model=model, loader=loader,
            label_input_ids=lab_ids, label_attention_mask=lab_mask,
            val_labels=val_labels, device=torch.device("cpu"),
            train_labels=train_labels, n_labels=k, seed=42,
        )
    finally:
        diag_mod.forward_pass = _orig_fp

    assert autocast_active_during_calls, "forward_pass was never called"
    assert not any(autocast_active_during_calls), (
        "autocast was active during compute_diagnostics forward pass (F3 bug still present)"
    )
    assert result.get("precision_mode") == "fp32_no_autocast", (
        f"Expected 'fp32_no_autocast', got {result.get('precision_mode')!r}"
    )



# ── Decision function tests ───────────────────────────────────────────────────

def _make_crit(V1=True, V2=True, V3=True, V4=True, V5=True,
               argmax_true=0.0, one_hot=0.5):
    return {
        "V1": V1, "V2": V2, "V3": V3, "V4": V4, "V5": V5,
        "viable": V1 and V2 and V3 and V4 and V5,
        "escalate_v3": (not V3) and (argmax_true > 0.5),
        "V3_value": one_hot,
        "V4_value": 0.6,
        "V5_value": "70/63",
        "argmax_alpha_is_true_frac": argmax_true,
        "p_at_1_hits": 70,
        "prior_p_at_1_hits": 63,
    }


def test_decision_l_viable():
    """L passes all criteria → DECIDED with chosen_variant='L'."""
    results = {"L": _make_crit(), "L2": _make_crit()}
    d = run0_decision(results, candidates=("L", "L2"))
    assert d["status"] == "DECIDED"
    assert d["chosen_variant"] == "L"


def test_decision_l_fails_l2_viable():
    """L fails V4; L2 passes → DECIDED with chosen_variant='L2'."""
    results = {
        "L":  _make_crit(V4=False),
        "L2": _make_crit(),
    }
    d = run0_decision(results, candidates=("L", "L2"))
    assert d["status"] == "DECIDED"
    assert d["chosen_variant"] == "L2"


def test_decision_neither_viable():
    """Neither L nor L2 passes → NONE_VIABLE."""
    results = {
        "L":  _make_crit(V4=False, V5=False),
        "L2": _make_crit(V4=False),
    }
    d = run0_decision(results, candidates=("L", "L2"))
    assert d["status"] in ("NONE_VIABLE", "STOP_AND_ASK")
    assert d["chosen_variant"] is None


def test_decision_v3_escalate():
    """L fails V3 but argmax-alpha is true label >50% → ESCALATE."""
    results = {
        "L":  _make_crit(V3=False, argmax_true=0.7, one_hot=0.95),
        "L2": _make_crit(V4=False),
    }
    d = run0_decision(results, candidates=("L", "L2"))
    assert d["status"] == "ESCALATE"
    assert d["chosen_variant"] is None


def test_decision_v2v5_only_extend():
    """L fails only V2/V5; V1,V3,V4 pass → EXTEND eligible."""
    results = {
        "L":  _make_crit(V2=False, V5=False),
        "L2": _make_crit(V2=False, V5=False),
    }
    d = run0_decision(results, candidates=("L", "L2"))
    assert d["status"] == "EXTEND"


def test_decision_invalid_diagnostics():
    """diagnostic_error present → INVALID_DIAGNOSTICS immediately."""
    results = {
        "L": {"diagnostic_error": "alpha is None: model did not return diag['alpha']"},
    }
    d = run0_decision(results, candidates=("L", "L2"))
    assert d["status"] == "INVALID_DIAGNOSTICS"
    assert d["chosen_variant"] is None


def test_extension_not_eligible_if_v4_also_fails():
    """
    Extension rule check: EXTEND is only eligible if V2/V5 fail AND V1,V3,V4 ALL pass.
    If V4 also fails → NONE_VIABLE (extension not permitted).
    """
    results = {
        "L":  _make_crit(V2=False, V4=False, V5=False),  # V4 fails too
        "L2": _make_crit(V2=False, V4=False, V5=False),
    }
    d = run0_decision(results, candidates=("L", "L2"))
    assert d["status"] not in ("EXTEND",), (
        "EXTEND must NOT be granted if V4 also fails (only V2/V5-only failure eligible)"
    )


# ── Artifact completeness ─────────────────────────────────────────────────────

def test_artifact_completeness():
    """check_artifact_completeness returns False and lists missing files for empty dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ok, missing, _ = check_artifact_completeness(tmpdir)
        assert not ok
        assert len(missing) == len(REQUIRED_ARTIFACTS), (
            f"Expected {len(REQUIRED_ARTIFACTS)} missing; got {len(missing)}"
        )


def test_sha256sums_consistency():
    """verify_sha256sums passes when all files match, fails for hash mismatch or missing."""
    import hashlib
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a real file and a correct SHA256SUMS.txt
        test_file = os.path.join(tmpdir, "env.json")
        with open(test_file, "w") as f:
            f.write('{"ok": true}')
        h = hashlib.sha256(open(test_file, "rb").read()).hexdigest()
        sums_path = os.path.join(tmpdir, "SHA256SUMS.txt")
        with open(sums_path, "w") as f:
            f.write(f"{h}  env.json\n")
        ok, failures = verify_sha256sums(tmpdir)
        assert ok, f"Expected ok=True; failures={failures}"

        # Corrupt the file
        with open(test_file, "w") as f:
            f.write('{"ok": false}')
        ok2, failures2 = verify_sha256sums(tmpdir)
        assert not ok2
        assert any("HASH_MISMATCH" in f for f in failures2)

        # Missing file entry
        with open(sums_path, "w") as f:
            f.write(f"{h}  nonexistent.json\n")
        ok3, failures3 = verify_sha256sums(tmpdir)
        assert not ok3
        assert any("MISSING" in f for f in failures3)
