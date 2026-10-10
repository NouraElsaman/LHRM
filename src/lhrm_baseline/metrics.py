"""
Metrics: P@k and nDCG@k — implemented from paper Eqs. 13-16.

Eq.13: P@k = (1/k) * sum_{t in top-k} y_t
Eq.14: DCG@k = (1/k) * sum_{t in rank_k(y_hat)} y_t / log2(t+1)   [1/k cancels with IDCG]
Eq.15: IDCG@k = (1/k) * sum_{t=1..min(k,||y||_0)} 1/log2(t+1)     [same 1/k]
Eq.16: nDCG@k = DCG@k / IDCG@k

Note: The 1/k factors in Eqs.14-15 cancel exactly in Eq.16, so we implement:
  DCG@k = sum_{t=1..k} y_{rank_t} / log2(t+1)
  IDCG@k = sum_{t=1..min(k,||y||_0)} 1/log2(t+1)
  nDCG@k = DCG@k / IDCG@k

IMPORTANT difference from legacy code (precision_at_k in Model/lhrm.py):
- Legacy: divides by min(k, num_true_labels) — INFLATED for k>1
- This implementation: divides by k — CORRECT per Eq.13
- They agree at k=1 only.

TIE-BREAKING (N6/A18):
All ranking functions use seeded-random tie-breaking (never column-index order).
Seed defaults to 42; can be overridden for reproducibility checks.

Reference: Paper Eqs. 13-16; verified by unit tests in tests/test_metrics.py
"""
from __future__ import annotations
import numpy as np
from typing import List, Optional


def _seeded_rank_order(scores: np.ndarray, seed: int) -> np.ndarray:
    """
    Argsort (descending) with seeded-random tie-breaking.
    Shared logic with hrrm._seeded_rank_order.
    """
    rng = np.random.default_rng(seed)
    jitter = rng.random(len(scores)) * 1e-9
    return np.argsort(-(scores + jitter), kind="stable")


def precision_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int,
    seed: int = 42,
) -> float:
    """
    P@k = (1/k) * sum_{t in top-k positions} y_true_t

    Args:
        y_true:  [N, L] binary relevance matrix
        y_score: [N, L] predicted scores (higher = more relevant)
        k:       cutoff
        seed:    seed for tie-breaking (default 42)

    Returns:
        Mean P@k over N documents.

    NOTE: This is the CORRECT implementation (Eq.13 divisor = k).
    The legacy precision_at_k in Model/lhrm.py uses min(k, |y|) and is WRONG for k>1.
    TIE-BREAKING: seeded-random (N6/A18), never column-index order.
    """
    assert y_true.shape == y_score.shape, f"Shape mismatch: {y_true.shape} vs {y_score.shape}"
    N = y_true.shape[0]

    hits = np.array([
        y_true[i, _seeded_rank_order(y_score[i], seed=seed ^ i)[:k]].sum()
        for i in range(N)
    ], dtype=float)

    return float(np.mean(hits / k))


def dcg_at_k(rel: np.ndarray, k: int) -> float:
    """
    DCG@k = sum_{t=1..k} rel[t-1] / log2(t+1)
    rel is the relevance array in rank order (rel[0] = rank-1 item, etc.)
    """
    rel_k = rel[:k]
    if len(rel_k) == 0:
        return 0.0
    positions = np.arange(1, len(rel_k) + 1, dtype=float)  # [1, 2, ..., min(k, len(rel))]
    discounts = np.log2(positions + 1)
    return float(np.sum(rel_k.astype(float) / discounts))


def idcg_at_k(y_true_flat: np.ndarray, k: int) -> float:
    """
    IDCG@k = DCG of ideal ranking = sum_{t=1..min(k,||y||_0)} 1/log2(t+1)
    """
    n_relevant = int(y_true_flat.sum())
    ideal_k = min(k, n_relevant)
    if ideal_k == 0:
        return 0.0
    positions = np.arange(1, ideal_k + 1, dtype=float)
    return float(np.sum(1.0 / np.log2(positions + 1)))


def ndcg_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int,
    seed: int = 42,
) -> float:
    """
    nDCG@k = mean over docs of DCG@k / IDCG@k
    Documents with no relevant labels get nDCG = 0 (per A16).

    Args:
        y_true:  [N, L] binary relevance matrix
        y_score: [N, L] predicted scores
        k:       cutoff
        seed:    seed for tie-breaking (default 42)

    Returns:
        Mean nDCG@k over N documents.
    TIE-BREAKING: seeded-random (N6/A18), never column-index order.
    """
    assert y_true.shape == y_score.shape
    N = y_true.shape[0]

    scores = []
    for i in range(N):
        # N6/A18: seeded-random tie-breaking, per-document seed
        rank_idx = _seeded_rank_order(y_score[i], seed=seed ^ i)
        rel_ranked = y_true[i, rank_idx]  # relevance in rank order
        d = dcg_at_k(rel_ranked, k)
        ideal = idcg_at_k(y_true[i], k)

        if ideal == 0.0:
            scores.append(0.0)
        else:
            scores.append(d / ideal)

    return float(np.mean(scores))


def compute_f1_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute Micro/Macro/Weighted F1, Precision, Recall at a fixed threshold.

    THRESHOLD POLICY: threshold is applied uniformly (default 0.5).
    This is the PRELIMINARY fixed-threshold result.
    Do NOT use the test set to select or tune the threshold.
    Any threshold tuning must be done on dev only, with a separate protocol.

    Args:
        y_true:    [N, L] binary ground truth
        y_score:   [N, L] predicted scores in [0,1]
        threshold: decision boundary (default 0.5; always reported explicitly)

    Returns:
        dict with keys:
          threshold, micro_f1, micro_precision, micro_recall,
          macro_f1, macro_precision, macro_recall,
          weighted_f1, weighted_precision, weighted_recall
    """
    try:
        from sklearn.metrics import precision_recall_fscore_support
    except ImportError:
        return {"f1_error": "sklearn not available"}

    y_pred = (y_score >= threshold).astype(int)

    def _safe_prf(average):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average=average, zero_division=0
        )
        return float(p), float(r), float(f)

    mp, mr, mf = _safe_prf("micro")
    ap, ar, af = _safe_prf("macro")
    wp, wr, wf = _safe_prf("weighted")

    return {
        "threshold": threshold,
        "micro_f1": mf, "micro_precision": mp, "micro_recall": mr,
        "macro_f1": af, "macro_precision": ap, "macro_recall": ar,
        "weighted_f1": wf, "weighted_precision": wp, "weighted_recall": wr,
    }


def compute_per_label_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label_names: Optional[List[str]] = None,
    threshold: float = 0.5,
) -> list:
    """
    Per-label precision, recall, F1, TP, FP, FN, support.
    Returns list of dicts (one per label), sorted by label index.
    Zero-division: 0 (rare labels with no predictions or no support).
    """
    try:
        from sklearn.metrics import precision_recall_fscore_support
    except ImportError:
        return []

    y_pred = (y_score >= threshold).astype(int)
    n_labels = y_true.shape[1]
    if label_names is None:
        label_names = [str(i) for i in range(n_labels)]

    p_arr, r_arr, f_arr, sup_arr = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    tp = (y_pred & y_true.astype(int)).sum(axis=0)
    fp = (y_pred & (~y_true.astype(bool)).astype(int)).sum(axis=0)
    fn = ((~y_pred.astype(bool)).astype(int) & y_true.astype(int)).sum(axis=0)

    rows = []
    for i in range(n_labels):
        rows.append({
            "label": label_names[i],
            "index": i,
            "precision": float(p_arr[i]),
            "recall": float(r_arr[i]),
            "f1": float(f_arr[i]),
            "tp": int(tp[i]),
            "fp": int(fp[i]),
            "fn": int(fn[i]),
            "support": int(sup_arr[i]),
        })
    return rows


def compute_all_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k_values: List[int] = [1, 3, 5],
    threshold: float = 0.5,
    include_f1: bool = True,
) -> dict:
    """
    Compute P@k, nDCG@k, and (optionally) Micro/Macro/Weighted F1.

    threshold: applied uniformly for F1 (preliminary; not tuned on test).
    include_f1: if True, adds micro_f1, macro_f1, weighted_f1 and related keys.
    """
    results = {}
    for k in k_values:
        results["P@{}".format(k)] = precision_at_k(y_true, y_score, k)
    for k in k_values:
        if k > 1:
            results["nDCG@{}".format(k)] = ndcg_at_k(y_true, y_score, k)
    if include_f1:
        f1_metrics = compute_f1_metrics(y_true, y_score, threshold=threshold)
        results.update(f1_metrics)
    return results


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k_values: List[int] = [1, 3, 5],
    n_bootstrap: int = 1000,
    seed: int = 42,
    ci: float = 0.95
) -> dict:
    """
    Compute 95% bootstrap confidence intervals for all metrics.
    Resamples DOCUMENTS (rows), not (label, document) pairs.
    
    Returns dict: metric_name -> (lower, upper)
    """
    rng = np.random.default_rng(seed)
    N = y_true.shape[0]
    
    # First compute point estimates
    point = compute_all_metrics(y_true, y_score, k_values)
    
    # Bootstrap
    boot_results = {k: [] for k in point.keys()}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, N, size=N)
        bt = compute_all_metrics(y_true[idx], y_score[idx], k_values)
        for metric, val in bt.items():
            boot_results[metric].append(val)
    
    alpha = (1 - ci) / 2
    ci_results = {}
    for metric, vals in boot_results.items():
        arr = np.array(vals)
        ci_results[metric] = {
            "point": point[metric],
            "lower": float(np.quantile(arr, alpha)),
            "upper": float(np.quantile(arr, 1 - alpha)),
            "ci": ci,
            "n_bootstrap": n_bootstrap,
        }
    
    return ci_results


def legacy_precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """
    LEGACY (INCORRECT) precision_at_k from Model/lhrm.py.
    Divides by min(k, num_true_labels) instead of k.
    Included for comparison in unit tests only.
    DO NOT use in baseline results.
    """
    topk = np.argsort(-y_score, axis=1)[:, :k]
    hits = np.array([y_true[i, topk[i]].sum() for i in range(len(y_true))], dtype=float)
    denom = np.array([max(1, min(k, int(y_true[i].sum()))) for i in range(len(y_true))], dtype=float)
    return float(np.mean(hits / denom))
