"""
Hierarchical Re-Ranking Module (HRRM).

Implements the non-leaky protocol (Section 8.4):
  TRAIN neural LHRM -> validate -> predict on val.csv -> FIT HRRM on val predictions
  -> TEST: predict test.csv -> APPLY HRRM -> evaluate

Features per (document, label) row:
  s = sigmoid(y_hat_i)     — predicted probability [ASSUMED A12]
  r = rank (1-based, descending) of label i within the document

Target:
  I(y_hat_i, Y) in {0, 1}  — whether label i is truly relevant [Eq.9]

Gradient Boosting config (PAPER-DEFINED per A13):
  n_estimators=100, learning_rate=0.1, max_depth=3, subsample=1.0,
  criterion="friedman_mse", loss="squared_error",
  validation_fraction=0.1, tol=1e-4, n_iter_no_change=None (no early stopping)

PAPER: Eqs. 9-12

TIE-BREAKING (N6/A18, NECESSARY ENGINEERING ASSUMPTION):
  Primary:   raw neural score (descending)
  Secondary: seeded-random permutation (seed = HRRM seed XOR doc_idx)
  NEVER:     column/label index order (biases toward low-index labels)
  Applies to: build_hrrm_features (rank feature), apply_to_predictions (final ranking),
              and metrics.py (P@k, nDCG@k).
"""
from __future__ import annotations
import json
import os
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


def _seeded_rank_order(scores: np.ndarray, seed: int) -> np.ndarray:
    """
    Return argsort indices (descending by score) with seeded-random tie-breaking.

    Tie-breaking rule (A18):
      1. Primary key:   raw score (descending)
      2. Secondary key: seeded-random permutation of label indices

    This ensures:
    - Deterministic across identical runs (given the same seed).
    - Independent of label column order (no index-favouritism).
    - Different permutations per document (seed XOR'd with doc_idx as needed).

    Args:
        scores: [k] float array of raw neural scores (logits or sigmoid)
        seed:   integer seed for the random permutation

    Returns:
        order: [k] argsort indices such that scores[order] is descending
    """
    k = len(scores)
    rng = np.random.default_rng(seed)
    # Random tiebreak values in [0,1); added as a fractional perturbation
    jitter = rng.random(k) * 1e-9  # negligible relative to score differences
    order = np.argsort(-(scores + jitter), kind="stable")
    return order


def build_hrrm_features(
    predictions_path: str,
    label_order: List[str],
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    """
    Build (score, rank) feature matrix and correctness targets
    from a predictions JSON file.
    
    Predictions JSON format (per document):
    {
      "scores": [float, ...],   # sigmoid scores in label_order
      "labels": [str, ...],     # label names in label_order
      "true_labels": [str, ...]
    }
    
    Returns:
        X: [N*k, 2] features (score, rank)
        y: [N*k] targets (0/1 correctness)
        doc_ids: list of doc_idx for each row
        n_docs: number of documents
    """
    with open(predictions_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    
    rows_X = []
    rows_y = []
    doc_ids = []
    
    for doc_idx, pred in enumerate(predictions):
        scores = np.array(pred["scores"], dtype=float)  # [k]
        true_set = set(pred["true_labels"])

        # Compute ranks: 1-based, descending score.
        # N6/A18 fix: tie-breaking by seeded-random permutation, NOT column index.
        rank_order = _seeded_rank_order(scores, seed=seed ^ doc_idx)
        ranks = np.empty_like(rank_order)
        ranks[rank_order] = np.arange(1, len(scores) + 1)

        for label_idx, label_name in enumerate(label_order):
            score = float(scores[label_idx])
            rank = int(ranks[label_idx])
            is_true = 1 if label_name in true_set else 0

            rows_X.append([score, rank])
            rows_y.append(is_true)
            doc_ids.append(doc_idx)

    X = np.array(rows_X, dtype=float)
    y = np.array(rows_y, dtype=float)

    return X, y, doc_ids, len(predictions)


class HRRM:
    """
    Hierarchical Re-Ranking Module.
    
    Wraps a GradientBoostingRegressor with the paper's explicit config.
    Provides fit, predict, save, load, and apply methods.
    
    Protocol (A17):
      fit()   — on validation predictions
      apply() — on test predictions (ONLY AFTER fit)
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.model = GradientBoostingRegressor(
            # ALL VALUES SET EXPLICITLY per A13 — do not rely on sklearn defaults
            n_estimators=100,                # PAPER-EXPLICIT
            learning_rate=0.1,              # PAPER-EXPLICIT
            max_depth=3,                    # PAPER-EXPLICIT
            subsample=1.0,                  # PAPER-EXPLICIT
            # F8 correction: criterion='friedman_mse' IS still valid in sklearn 1.9.0.
            # Verified: GradientBoostingRegressor(criterion='friedman_mse').get_params()
            # returns {'criterion': 'friedman_mse'}. A FutureWarning is raised
            # (param will be removed in a future version) but it is functional now.
            # PAPER-EXPLICIT: the paper lists this explicitly.
            criterion="friedman_mse",       # PAPER-EXPLICIT (F8 fix; sklearn 1.9.0)
            loss="squared_error",           # PAPER-EXPLICIT
            validation_fraction=0.1,        # PAPER-EXPLICIT (INERT: n_iter_no_change=None)
            tol=1e-4,                       # PAPER-EXPLICIT (INERT: n_iter_no_change=None)
            n_iter_no_change=None,          # NECESSARY ENGINEERING ASSUMPTION (no early stopping)
            random_state=seed,              # REPOSITORY-DEFINED
        )
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "HRRM":
        """Fit the re-ranking model on (score, rank) features."""
        assert X.shape[1] == 2, f"Expected 2 features (score, rank), got {X.shape[1]}"
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict adjusted scores."""
        assert self.is_fitted, "HRRM must be fit before predict"
        return self.model.predict(X)
    
    def apply_to_predictions(
        self,
        predictions_path: str,
        label_order: List[str],
        out_path: str,
    ) -> dict:
        """
        Apply fitted HRRM to a predictions file and save re-ranked results.
        
        Returns adjusted predictions with same format as input,
        plus 'adjusted_scores' field.
        """
        assert self.is_fitted, "HRRM must be fit before applying"
        
        with open(predictions_path, "r", encoding="utf-8") as f:
            predictions = json.load(f)
        
        k = len(label_order)
        adjusted_preds = []
        
        for doc_idx, pred in enumerate(predictions):
            scores = np.array(pred["scores"], dtype=float)

            # N6/A18 fix: seeded-random tie-breaking for rank features
            rank_order = _seeded_rank_order(scores, seed=self.seed ^ doc_idx)
            ranks = np.empty_like(rank_order)
            ranks[rank_order] = np.arange(1, k + 1)

            X_doc = np.column_stack([scores, ranks.astype(float)])  # [k, 2]
            adj_scores = self.predict(X_doc)  # [k]

            # Re-sort by adjusted score with seeded tie-breaking
            new_order = _seeded_rank_order(adj_scores, seed=self.seed ^ doc_idx ^ 0xDEAD)

            adjusted_preds.append({
                "scores": [float(adj_scores[i]) for i in range(k)],
                "adjusted_scores_sorted_labels": [label_order[i] for i in new_order],
                "true_labels": pred["true_labels"],
                "labels": pred["labels"],
            })
        
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(adjusted_preds, f, indent=2)
        
        return {"n_docs": len(predictions), "out_path": out_path}
    
    def save(self, path: str):
        """Save fitted model to disk."""
        joblib.dump(self.model, path)
        print(f"HRRM saved to: {path}")
    
    @classmethod
    def load(cls, path: str, seed: int = 42) -> "HRRM":
        """Load a fitted HRRM from disk."""
        hrrm = cls(seed=seed)
        hrrm.model = joblib.load(path)
        hrrm.is_fitted = True
        return hrrm
    
    def leaky_diagnostic(
        self,
        predictions_path: str,
        label_order: List[str],
    ) -> dict:
        """
        Reproduce the repository's LEAKY protocol for diagnostic purposes.
        FIT on random 70% of test rows, evaluate on ALL test rows.
        
        WARNING: This is the WRONG protocol. Use only to quantify leakage.
        NEVER report as a baseline result.
        """
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import GradientBoostingRegressor
        
        X, y, _, _ = build_hrrm_features(predictions_path, label_order)
        
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.3, random_state=self.seed
        )
        
        leaky_model = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=3,
            subsample=1.0, loss="squared_error",
            validation_fraction=0.1, tol=1e-4, n_iter_no_change=None,
            random_state=self.seed,
        )
        leaky_model.fit(X_train, y_train)
        adj_all = leaky_model.predict(X)
        
        return {"adj_scores": adj_all, "X": X, "y": y, "note": "LEAKY_DIAGNOSTIC"}
