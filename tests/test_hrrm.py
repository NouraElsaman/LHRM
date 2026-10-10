"""
Unit tests for hrrm.py.
Tests: feature building, fit/predict, save/load round-trip, rank features correctness.
Uses synthetic predictions — no BERT, no real data.
"""
import json
import os
import tempfile
import numpy as np
import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lhrm_baseline.hrrm import HRRM, build_hrrm_features


def make_synthetic_predictions(n_docs=50, n_labels=10, seed=42):
    """Create synthetic predictions JSON file."""
    rng = np.random.default_rng(seed)
    label_order = [f"label_{i}" for i in range(n_labels)]
    preds = []
    for _ in range(n_docs):
        scores = rng.random(n_labels).tolist()
        n_true = rng.integers(1, 4)
        true_idx = rng.choice(n_labels, size=n_true, replace=False)
        true_labels = [label_order[i] for i in true_idx]
        preds.append({
            "scores": scores,
            "labels": label_order,
            "true_labels": true_labels,
        })
    return preds, label_order


def test_rank_features_are_1_to_k():
    """Each document's rank features should be 1..k (one per label, no repeats)."""
    preds, label_order = make_synthetic_predictions(n_docs=10, n_labels=5)
    k = len(label_order)
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(preds, f)
        preds_path = f.name
    
    try:
        X, y, doc_ids, n_docs = build_hrrm_features(preds_path, label_order)
        
        # Check shapes
        assert X.shape == (n_docs * k, 2), f"X shape: {X.shape}"
        assert y.shape == (n_docs * k,), f"y shape: {y.shape}"
        
        # Check rank features for each document
        for doc_idx in range(n_docs):
            doc_rows = [i for i, d in enumerate(doc_ids) if d == doc_idx]
            ranks = X[doc_rows, 1].astype(int)
            assert sorted(ranks) == list(range(1, k + 1)),                 f"Doc {doc_idx}: ranks {sorted(ranks)} != [1..{k}]"
    finally:
        os.unlink(preds_path)


def test_hrrm_fit_predict_round_trip():
    """HRRM can be fit and then predict."""
    preds, label_order = make_synthetic_predictions(n_docs=100, n_labels=10)
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(preds, f)
        preds_path = f.name
    
    try:
        X, y, _, _ = build_hrrm_features(preds_path, label_order)
        
        hrrm = HRRM(seed=42)
        hrrm.fit(X, y)
        
        preds_out = hrrm.predict(X)
        assert preds_out.shape == (len(y),), f"Predict shape: {preds_out.shape}"
        assert np.isfinite(preds_out).all(), "Non-finite predictions"
    finally:
        os.unlink(preds_path)


def test_hrrm_save_load_round_trip():
    """HRRM save/load produces identical predictions."""
    preds, label_order = make_synthetic_predictions(n_docs=80, n_labels=8)
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(preds, f)
        preds_path = f.name
    
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        model_path = f.name
    
    try:
        X, y, _, _ = build_hrrm_features(preds_path, label_order)
        
        hrrm = HRRM(seed=42)
        hrrm.fit(X, y)
        pred_before = hrrm.predict(X)
        
        hrrm.save(model_path)
        hrrm2 = HRRM.load(model_path, seed=42)
        pred_after = hrrm2.predict(X)
        
        np.testing.assert_array_almost_equal(pred_before, pred_after, decimal=6)
    finally:
        os.unlink(preds_path)
        if os.path.exists(model_path):
            os.unlink(model_path)


def test_hrrm_explicit_gb_config():
    """HRRM GradientBoostingRegressor has paper-specified params (A13)."""
    hrrm = HRRM(seed=42)
    gb = hrrm.model
    
    assert gb.n_estimators == 100, f"n_estimators: {gb.n_estimators}"
    assert abs(gb.learning_rate - 0.1) < 1e-9, f"lr: {gb.learning_rate}"
    assert gb.max_depth == 3, f"max_depth: {gb.max_depth}"
    assert abs(gb.subsample - 1.0) < 1e-9, f"subsample: {gb.subsample}"
    # F8: criterion='friedman_mse' is PAPER-EXPLICIT and IS valid in sklearn 1.9.0.
    # A FutureWarning is raised but value is accepted. Assert it is set explicitly.
    assert gb.criterion == "friedman_mse", f"criterion: {gb.criterion}"
    assert gb.n_iter_no_change is None, f"n_iter_no_change should be None: {gb.n_iter_no_change}"



def test_hrrm_targets_are_binary():
    """HRRM target y must be 0 or 1 (correctness indicator)."""
    preds, label_order = make_synthetic_predictions(n_docs=20, n_labels=5)
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(preds, f)
        preds_path = f.name
    
    try:
        X, y, _, _ = build_hrrm_features(preds_path, label_order)
        unique_vals = set(y.tolist())
        assert unique_vals.issubset({0.0, 1.0}), f"Non-binary targets: {unique_vals}"
    finally:
        os.unlink(preds_path)
