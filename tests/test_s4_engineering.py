"""
S4 Tests: Engineering items required before any GPU bundle.

Tests cover:
  N6: Tie-breaking invariance (permuting labels leaves metrics unchanged)
  N7: Dynamic padding (collate_fn pads to batch max, shapes correct)
  S4-3: HRRM API guards (unfitted refusal)
  S4-5: Label-order hash; train/dev/test text disjointness; label text token count
  S4-6: Script --help exits 0
  S4-7: End-to-end tiny smoke on synthetic (random-weight) model, DIAGNOSTIC ONLY
"""
import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from lhrm_baseline.hrrm import HRRM, build_hrrm_features, _seeded_rank_order
from lhrm_baseline.metrics import (
    precision_at_k,
    ndcg_at_k,
    _seeded_rank_order as metrics_rank_order,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_preds_file(n_docs=30, n_labels=10, seed=42):
    """Synthetic predictions JSON list with some tied scores."""
    rng = np.random.default_rng(seed)
    label_order = [f"label_{i}" for i in range(n_labels)]
    preds = []
    for d in range(n_docs):
        scores = [round(float(s), 2) for s in rng.random(n_labels)]  # ties via rounding
        true_k = max(1, int(n_labels * 0.2))
        true_labels = [
            label_order[i]
            for i in rng.choice(n_labels, size=true_k, replace=False)
        ]
        preds.append({
            "scores": scores,
            "labels": label_order,
            "true_labels": true_labels,
        })
    return preds, label_order


# ===========================================================================
# N6 Tie-breaking
# ===========================================================================

def test_seeded_rank_order_deterministic():
    """Same inputs and seed => same output."""
    scores = np.array([0.5, 0.3, 0.5, 0.1, 0.5])
    o1 = _seeded_rank_order(scores, seed=42)
    o2 = _seeded_rank_order(scores, seed=42)
    np.testing.assert_array_equal(o1, o2)


def test_seeded_rank_order_diverse_on_ties():
    """All-equal scores => multiple distinct orderings across seeds."""
    scores = np.ones(20)
    seen = set()
    for s in range(20):
        seen.add(tuple(_seeded_rank_order(scores, seed=s)))
    assert len(seen) > 1, "Tied scores must produce diverse orderings"


def test_metrics_label_permutation_invariance():
    """
    N6 core: permuting (y_true, y_score) columns identically leaves
    P@k and nDCG@k unchanged.
    """
    rng = np.random.default_rng(0)
    N, L, k = 50, 15, 3
    y_true = (rng.random((N, L)) > 0.8).astype(float)
    y_score = rng.random((N, L))

    p_orig = precision_at_k(y_true, y_score, k, seed=42)
    n_orig = ndcg_at_k(y_true, y_score, k, seed=42)

    perm = rng.permutation(L)
    p_perm = precision_at_k(y_true[:, perm], y_score[:, perm], k, seed=42)
    n_perm = ndcg_at_k(y_true[:, perm], y_score[:, perm], k, seed=42)

    assert abs(p_orig - p_perm) < 1e-6, f"P@{k} not invariant: {p_orig} vs {p_perm}"
    assert abs(n_orig - n_perm) < 1e-6, f"nDCG@{k} not invariant: {n_orig} vs {n_perm}"


def test_hrrm_features_column_permutation():
    """
    Permuting label columns consistently in the predictions file leaves
    the HRRM feature matrix shape unchanged and targets identical.
    """
    preds, label_order = make_preds_file(n_docs=20, n_labels=8, seed=7)
    rng = np.random.default_rng(99)
    perm = rng.permutation(len(label_order))
    perm_lo = [label_order[i] for i in perm]

    perm_preds = [
        {
            "scores": [p["scores"][i] for i in perm],
            "labels": perm_lo,
            "true_labels": p["true_labels"],
        }
        for p in preds
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(preds, f)
        path_orig = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(perm_preds, f)
        path_perm = f.name

    try:
        X, y, _, _ = build_hrrm_features(path_orig, label_order, seed=42)
        Xp, yp, _, _ = build_hrrm_features(path_perm, perm_lo, seed=42)
        assert X.shape == Xp.shape
        # Targets (correctness indicators) are label-name-based, must be equal
        np.testing.assert_array_equal(
            np.sort(y), np.sort(yp),
            err_msg="Target values changed under column permutation",
        )
    finally:
        os.unlink(path_orig)
        os.unlink(path_perm)


# ===========================================================================
# N7 Dynamic padding
# ===========================================================================

def test_collate_fn_pads_to_batch_max():
    """collate_fn pads input_ids and attention_mask to the longest sequence."""
    from lhrm_baseline.data import collate_fn

    batch = [
        {"input_ids": torch.tensor([1, 2, 3, 4]),        "attention_mask": torch.ones(4, dtype=torch.long), "labels": torch.tensor([1.0, 0.0])},
        {"input_ids": torch.tensor([1, 2, 3, 4, 5, 6]),  "attention_mask": torch.ones(6, dtype=torch.long), "labels": torch.tensor([0.0, 1.0])},
        {"input_ids": torch.tensor([1, 2]),               "attention_mask": torch.ones(2, dtype=torch.long), "labels": torch.tensor([1.0, 1.0])},
    ]
    out = collate_fn(batch)
    assert out["input_ids"].shape == (3, 6)
    assert out["attention_mask"].shape == (3, 6)
    assert out["labels"].shape == (3, 2)
    # Padding positions must have attention_mask = 0 and input_ids = 0
    assert out["attention_mask"][0, 4].item() == 0
    assert out["attention_mask"][0, 5].item() == 0
    assert out["input_ids"][0, 4].item() == 0
    assert out["attention_mask"][2, 2].item() == 0
    # Non-padded positions must have mask = 1
    assert out["attention_mask"][1, 5].item() == 1


def test_collate_fn_single_item():
    """A batch of one item should not be padded."""
    from lhrm_baseline.data import collate_fn
    batch = [
        {"input_ids": torch.tensor([101, 200, 102]), "attention_mask": torch.ones(3, dtype=torch.long), "labels": torch.tensor([1.0])}
    ]
    out = collate_fn(batch)
    assert out["input_ids"].shape == (1, 3)
    assert out["attention_mask"].shape == (1, 3)


# ===========================================================================
# S4-3 HRRM API guards
# ===========================================================================

def test_hrrm_predict_raises_if_unfitted():
    hrrm = HRRM(seed=42)
    with pytest.raises(AssertionError, match="fit"):
        hrrm.predict(np.array([[0.5, 1.0]]))


def test_hrrm_apply_raises_if_unfitted():
    preds, label_order = make_preds_file(n_docs=5, n_labels=4, seed=1)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(preds, f)
        preds_path = f.name
    try:
        hrrm = HRRM(seed=42)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f2:
            out_path = f2.name
        try:
            with pytest.raises(AssertionError, match="fit"):
                hrrm.apply_to_predictions(preds_path, label_order, out_path)
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)
    finally:
        os.unlink(preds_path)


# ===========================================================================
# S4-5 Data tests
# ===========================================================================

def test_aapd_label_order_hash_stable():
    """AAPD label order hash must be stable across runs."""
    mapping_path = os.path.join(ROOT, "Data", "label_mapping.json")
    if not os.path.exists(mapping_path):
        pytest.skip("Data/label_mapping.json not found")
    with open(mapping_path) as f:
        lm = json.load(f)
    col_order = lm.get("_col_order", [])
    assert len(col_order) == 54
    h = hashlib.sha256(json.dumps(col_order, sort_keys=False).encode()).hexdigest()
    hash_file = os.path.join(ROOT, "Data", "aapd_label_order.hash")
    if not os.path.exists(hash_file):
        with open(hash_file, "w") as hf:
            hf.write(h)
    with open(hash_file) as hf:
        expected = hf.read().strip()
    assert h == expected, f"AAPD label order hash changed: expected={expected}, got={h}"


def test_reuters_label_order_hash_stable():
    """Reuters label order hash must be stable across runs."""
    labels_path = os.path.join(ROOT, "data", "processed", "Reuters", "labels.json")
    if not os.path.exists(labels_path):
        pytest.skip("Reuters processed data not found")
    with open(labels_path) as f:
        labels = json.load(f)
    assert len(labels) == 90
    h = hashlib.sha256(json.dumps(labels, sort_keys=False).encode()).hexdigest()
    hash_file = os.path.join(ROOT, "data", "processed", "Reuters", "label_order.hash")
    if not os.path.exists(hash_file):
        with open(hash_file, "w") as hf:
            hf.write(h)
    with open(hash_file) as hf:
        expected = hf.read().strip()
    assert h == expected, f"Reuters label order hash changed: expected={expected}, got={h}"


def test_reuters_splits_text_disjoint():
    """Reuters train, dev, test must share zero documents by text content."""
    base = os.path.join(ROOT, "data", "processed", "Reuters")
    paths = {s: os.path.join(base, f"{s}_texts.json") for s in ["train", "dev", "test"]}
    if not all(os.path.exists(p) for p in paths.values()):
        pytest.skip("Reuters processed data not found")
    splits = {}
    for s, p in paths.items():
        with open(p, encoding="utf-8") as f:
            splits[s] = set(json.load(f))
    assert len(splits["train"] & splits["dev"]) == 0, "train/dev text overlap"
    assert len(splits["train"] & splits["test"]) == 0, "train/test text overlap"
    assert len(splits["dev"] & splits["test"]) == 0, "dev/test text overlap"


def test_label_texts_no_braces_aapd():
    """AAPD label texts must not contain Python-dict braces."""
    p = os.path.join(ROOT, "data", "processed", "AAPD", "label_texts.json")
    if not os.path.exists(p):
        pytest.skip("AAPD label_texts.json not found")
    with open(p, encoding="utf-8") as f:
        texts = json.load(f)
    bad = [i for i, t in enumerate(texts) if "{" in t or "}" in t]
    assert not bad, f"Brace syntax in label texts at indices: {bad}"


def test_label_texts_no_braces_reuters():
    """Reuters label texts must not contain Python-dict braces."""
    p = os.path.join(ROOT, "data", "processed", "Reuters", "label_texts.json")
    if not os.path.exists(p):
        pytest.skip("Reuters label_texts.json not found")
    with open(p, encoding="utf-8") as f:
        texts = json.load(f)
    bad = [i for i, t in enumerate(texts) if "{" in t or "}" in t]
    assert not bad, f"Brace syntax in label texts at indices: {bad}"


# ===========================================================================
# S4-6 Scripts --help
# ===========================================================================

@pytest.mark.parametrize("script_rel", [
    "scripts/prepare_aapd.py",
    "scripts/prepare_reuters.py",
    "run_baseline.py",
])
def test_script_help_exits_0(script_rel):
    """Script must parse cleanly and respond to --help with exit code 0."""
    script_path = os.path.join(ROOT, script_rel)
    if not os.path.exists(script_path):
        pytest.skip(f"{script_rel} not found")
    # Syntax check
    with open(script_path, "rb") as f:
        ast.parse(f.read())
    # --help
    r = subprocess.run(
        [sys.executable, script_path, "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, (
        f"{script_rel} --help exit={r.returncode}\n"
        f"stdout: {r.stdout[:200]}\nstderr: {r.stderr[:200]}"
    )


# ===========================================================================
# S4-7 End-to-end tiny smoke (DIAGNOSTIC ONLY, never reported)
# ===========================================================================

def test_end_to_end_tiny_smoke_cpu():
    """
    S4-7 DIAGNOSTIC ONLY: Forward pass + focal loss + backward on synthetic data.
    Uses a random tiny BERT (no pretrained weights). Never reported as a result.
    """
    try:
        from transformers import BertConfig, BertModel
        from lhrm_baseline.model import CSAM_L, get_cls_embedding
        from lhrm_baseline.losses import focal_loss
    except ImportError as e:
        pytest.skip(f"Import error: {e}")

    torch.manual_seed(42)
    B, k, H = 2, 5, 64

    cfg = BertConfig(
        hidden_size=H, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=128, max_position_embeddings=32, vocab_size=500,
    )
    bert = BertModel(cfg)
    csam = CSAM_L(hidden=H, scale_similarity=True)

    doc_ids  = torch.randint(0, 500, (B, 16))
    doc_mask = torch.ones(B, 16, dtype=torch.long)
    lbl_ids  = torch.randint(0, 500, (k, 8))
    lbl_mask = torch.ones(k, 8, dtype=torch.long)

    with torch.no_grad():
        h_T = get_cls_embedding(bert(input_ids=doc_ids, attention_mask=doc_mask))
        h_L = get_cls_embedding(bert(input_ids=lbl_ids, attention_mask=lbl_mask))

    logits, diag = csam(h_T, h_L)
    assert logits.shape == (B, k)
    assert not torch.isnan(logits).any(), "NaN in logits"

    labels = (torch.rand(B, k) > 0.8).float()
    loss = focal_loss(logits, labels, gamma=2.0, alpha=0.25, reduction="mean")
    assert not torch.isnan(loss), "NaN loss"
    assert loss.item() > 0

    loss.backward()
    assert csam.W.weight.grad is not None
    assert not torch.isnan(csam.W.weight.grad).any()
    assert 0.0 <= diag["alpha_max_mean"] <= 1.0
    assert 0.0 <= diag["alpha_one_hot_frac"] <= 1.0
