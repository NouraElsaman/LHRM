"""
Evaluation module for LHRM baseline.
Loads predictions JSON and computes metrics with bootstrap CIs.
"""
from __future__ import annotations
import json
import os
from typing import List, Optional

import numpy as np

from .metrics import compute_all_metrics, bootstrap_ci


def load_predictions_matrix(
    predictions_path: str,
    label_order: List[str],
):
    """
    Load predictions JSON and return (y_true, y_score) matrices.
    
    Returns:
        y_true: [N, k] binary
        y_score: [N, k] sigmoid scores
    """
    with open(predictions_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    
    N = len(predictions)
    k = len(label_order)
    label2idx = {l: i for i, l in enumerate(label_order)}
    
    y_true = np.zeros((N, k), dtype=float)
    y_score = np.zeros((N, k), dtype=float)
    
    for i, pred in enumerate(predictions):
        # Scores in label_order
        scores = pred["scores"]
        assert len(scores) == k, f"Doc {i}: expected {k} scores, got {len(scores)}"
        y_score[i] = scores
        
        # True labels
        for label in pred["true_labels"]:
            if label in label2idx:
                y_true[i, label2idx[label]] = 1.0
    
    return y_true, y_score


def evaluate_predictions(
    predictions_path: str,
    label_order: List[str],
    k_values: List[int] = [1, 3, 5],
    n_bootstrap: int = 1000,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Full evaluation of a predictions file.
    Returns metrics dict with point estimates and 95% bootstrap CIs.
    """
    y_true, y_score = load_predictions_matrix(predictions_path, label_order)
    
    # Point estimates
    metrics = compute_all_metrics(y_true, y_score, k_values)
    
    # Bootstrap CIs
    cis = bootstrap_ci(y_true, y_score, k_values, n_bootstrap, seed)
    
    if verbose:
        print(f"Results from: {os.path.basename(predictions_path)}")
        for metric, val in metrics.items():
            ci = cis[metric]
            print(f"  {metric}: {val*100:.2f}%  "
                  f"95% CI [{ci['lower']*100:.2f}%, {ci['upper']*100:.2f}%]")
    
    return {
        "metrics": {k: v * 100 for k, v in metrics.items()},  # in percent
        "ci95": {m: {"lower": c["lower"]*100, "upper": c["upper"]*100}
                 for m, c in cis.items()},
        "n_docs": len(y_true),
        "predictions_path": predictions_path,
    }
