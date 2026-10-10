"""
src/lhrm_baseline/diagnostics.py
Single source of truth for RUN_0 viability diagnostics.

Fixes implemented here (Phase 1F):
  F1: alpha tensor is read from diag["alpha"] (model must return it in eval mode).
      If alpha is missing or malformed → raise DiagnosticError → UNMEASURED.
  F2: V5 prior uses seeded tie-break (not column-order argmax); comparison is
      integer hit-count, strict >; float32/float64 representation cannot flip it.
  F3: all diagnostics run in fp32 with no autocast (torch.no_grad() only).

API used by both run_baseline.py CLI and the Kaggle notebook (thin wrappers only).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Import at module level so tests can monkeypatch it
from lhrm_baseline.train import forward_pass  # noqa: E402


# ── Custom exception ──────────────────────────────────────────────────────────

class DiagnosticError(RuntimeError):
    """
    Raised when alpha is missing, malformed, or NaN.
    The decision function converts this to status=INVALID_DIAGNOSTICS.
    Never let it silently default to 0 (that caused the vacuous-V3 bug).
    """


# ── Alpha validation ──────────────────────────────────────────────────────────

def _validate_alpha(alpha: np.ndarray, n_labels: int) -> None:
    """
    Raise DiagnosticError if alpha is malformed.
    Checks: shape, NaN/Inf, rows sum to 1±1e-3, mean_max ≥ 1/k.
    """
    if alpha is None:
        raise DiagnosticError(
            "alpha is None: model did not return diag['alpha']. "
            "Ensure CSAM_L.forward is called with return_alpha=True "
            "(eval mode or explicit flag)."
        )
    if alpha.ndim != 2:
        raise DiagnosticError(f"alpha must be 2-D [N, k]; got shape {alpha.shape}")
    if alpha.shape[1] != n_labels:
        raise DiagnosticError(
            f"alpha.shape[1]={alpha.shape[1]} != n_labels={n_labels}"
        )
    if not np.isfinite(alpha).all():
        raise DiagnosticError("alpha contains NaN or Inf values")
    row_sums = alpha.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        bad = int(np.sum(np.abs(row_sums - 1.0) > 1e-3))
        raise DiagnosticError(
            f"{bad} rows in alpha do not sum to 1 (±1e-3). "
            f"min_sum={row_sums.min():.6f} max_sum={row_sums.max():.6f}"
        )
    expected_min_max = 1.0 / n_labels
    actual_mean_max = float(alpha.max(axis=1).mean())
    if actual_mean_max < expected_min_max - 1e-6:
        raise DiagnosticError(
            f"mean_max_alpha={actual_mean_max:.6f} < 1/k={expected_min_max:.6f}. "
            "Impossible for a valid softmax — diagnostic is corrupted."
        )


# ── Frequency-prior ranker ────────────────────────────────────────────────────

def frequency_prior_hits_at_1(
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    seed: int = 42,
) -> int:
    """
    Compute frequency-prior P@1 hit count on val_labels.

    The prior always predicts the most-frequent training label for every doc.
    Tie-breaking: seeded random permutation (A16/A18) — NOT column-order argmax.
    Returns integer hit count (not float fraction) to avoid float32 artefacts.

    Preregistration: V5 compares integer hit counts, strict >.
    """
    label_freq = train_labels.sum(axis=0)  # [k] integer counts
    # Seeded tie-break: add tiny jitter before argmax
    rng = np.random.default_rng(seed)
    jitter = rng.random(label_freq.shape[0]) * 1e-9
    top1_idx = int(np.argmax(label_freq.astype(float) + jitter))
    # Hit count: how many val docs have that label = 1
    hits = int(val_labels[:, top1_idx].sum())
    return hits


# ── Main diagnostic function ──────────────────────────────────────────────────

def compute_diagnostics(
    model: torch.nn.Module,
    loader,
    label_input_ids: torch.Tensor,
    label_attention_mask: torch.Tensor,
    val_labels: np.ndarray,          # [N, k] ground-truth binary
    device: torch.device,
    train_labels: np.ndarray,        # [N_train, k] for prior computation
    n_labels: int,
    seed: int = 42,
    amp_for_training: bool = True,   # AMP used during training (no effect here)
) -> dict:
    """
    Compute all preregistered RUN_0 diagnostics in fp32 (no autocast).

    Returns a dict with keys needed by V1-V5 and the decision function:
      alpha_one_hot_frac       float  fraction of docs with max(alpha) > 0.9
      mean_max_alpha           float  mean of per-doc max(alpha); must be ≥ 1/k
      n_distinct_argmax        int    number of distinct argmax-alpha labels
      argmax_alpha_is_true_frac float fraction of docs where argmax-alpha ∈ true labels
      alpha_entropy_mean       float  mean per-doc entropy of alpha
      S_mean / S_std / S_min / S_max  float  distribution of similarity scores S_i
      per_label_logit_std      list[float] per-label logit std across N docs (length k)
      logit_std_frac           float  fraction of labels with std > 0.01
      p_at_1_hits              int    candidate P@1 hit count (integer)
      p_at_1                   float  candidate P@1 (hits / N)
      prior_p_at_1_hits        int    prior P@1 hit count (integer)
      prior_p_at_1             float  prior P@1 (hits / N)
      n_docs                   int    number of val documents evaluated

    Raises DiagnosticError if alpha is missing or malformed.
    AMP note: amp_for_training is recorded in the output for audit purposes only;
    it has no effect on this function's computation (always fp32).
    """
    from .train import forward_pass   # avoid circular import
    from .metrics import precision_at_k

    model.eval()
    all_alphas: list = []
    all_logits: list = []
    all_S:      list = []   # similarity scores S [B, k]

    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            msk  = batch["attention_mask"].to(device)

            # F3 fix: NO autocast; explicitly cast to float32 after forward
            # If model is in fp16/bf16 (e.g. loaded from ckpt), cast outputs.
            # Look up forward_pass through module globals so monkeypatching works in tests.
            _fp = globals().get("forward_pass", forward_pass)
            logits, diag = _fp(model, ids, msk,
                               label_input_ids, label_attention_mask)

            logits_np = logits.detach().float().cpu().numpy()  # [B, k]
            all_logits.append(logits_np)

            # F1 fix: require alpha tensor in diag; fail closed if missing
            alpha_t = diag.get("alpha")
            if alpha_t is None:
                raise DiagnosticError(
                    "diag['alpha'] is None in compute_diagnostics. "
                    "Call model in eval mode with return_alpha=True or set "
                    "model.csam.return_alpha = True before running diagnostics."
                )
            alpha_np = alpha_t.detach().float().cpu().numpy()  # [B, k]
            all_alphas.append(alpha_np)

            # S scores (before softmax) — stored in diag if model returns them
            S_t = diag.get("S")
            if S_t is not None:
                all_S.append(S_t.detach().float().cpu().numpy())

    # Stack
    all_logits_np = np.vstack(all_logits)   # [N, k]
    all_alphas_np = np.vstack(all_alphas)   # [N, k]
    N = all_alphas_np.shape[0]

    # Validate alpha
    _validate_alpha(all_alphas_np, n_labels)

    # ── Alpha statistics ───────────────────────────────────────────────────────
    per_doc_max_alpha = all_alphas_np.max(axis=1)    # [N]
    one_hot_frac  = float((per_doc_max_alpha > 0.9).mean())
    mean_max_alpha = float(per_doc_max_alpha.mean())

    per_doc_argmax = all_alphas_np.argmax(axis=1)    # [N]
    n_distinct_argmax = int(len(set(per_doc_argmax.tolist())))

    # argmax-alpha is a true label?
    argmax_is_true = np.array([
        int(val_labels[i, per_doc_argmax[i]]) for i in range(N)
    ], dtype=float)
    argmax_alpha_is_true_frac = float(argmax_is_true.mean())

    # Per-doc entropy of alpha
    alpha_entropy = -(all_alphas_np * np.log(all_alphas_np + 1e-10)).sum(axis=1)
    alpha_entropy_mean = float(alpha_entropy.mean())

    # ── S distribution ─────────────────────────────────────────────────────────
    if all_S:
        S_np = np.vstack(all_S)   # [N, k]
        S_stats = {
            "S_mean": float(S_np.mean()),
            "S_std":  float(S_np.std()),
            "S_min":  float(S_np.min()),
            "S_max":  float(S_np.max()),
        }
    else:
        S_stats = {"S_mean": None, "S_std": None, "S_min": None, "S_max": None}

    # ── Logit variance (V4) — fp32 ────────────────────────────────────────────
    per_label_logit_std = all_logits_np.std(axis=0)  # [k]
    logit_std_frac = float((per_label_logit_std > 0.01).mean())

    # ── P@1 — integer hit counts (F2 fix) ─────────────────────────────────────
    # Candidate: integer hit count using seeded-random tie-breaking
    top1_per_doc = np.array([
        int(val_labels[i, np.argmax(
            all_logits_np[i]
            + np.random.default_rng(seed ^ i).random(n_labels) * 1e-9
        )])
        for i in range(N)
    ], dtype=int)
    p_at_1_hits = int(top1_per_doc.sum())
    p_at_1 = p_at_1_hits / max(N, 1)

    # Prior: seeded tie-break (not column-order) — integer hit count
    prior_hits = frequency_prior_hits_at_1(train_labels, val_labels, seed=seed)
    prior_p_at_1 = prior_hits / max(N, 1)

    result = {
        # Alpha stats
        "alpha_one_hot_frac":        one_hot_frac,
        "mean_max_alpha":            mean_max_alpha,
        "n_distinct_argmax":         n_distinct_argmax,
        "argmax_alpha_is_true_frac": argmax_alpha_is_true_frac,
        "alpha_entropy_mean":        alpha_entropy_mean,
        # S distribution
        **S_stats,
        # Logit variance (V4)
        "per_label_logit_std":       per_label_logit_std.tolist(),
        "logit_std_frac":            logit_std_frac,
        # P@1 (integer + float)
        "p_at_1_hits":               p_at_1_hits,
        "p_at_1":                    p_at_1,
        "prior_p_at_1_hits":         prior_hits,
        "prior_p_at_1":              prior_p_at_1,
        # Meta
        "n_docs":                    N,
        "n_labels":                  n_labels,
        "seed":                      seed,
        "precision_mode":            "fp32_no_autocast",
        "amp_for_training":          amp_for_training,
    }
    return result


# ── V1-V5 evaluation ──────────────────────────────────────────────────────────

def evaluate_criteria(
    diag_step0: dict,
    diag_step300: dict,
    losses: List[float],
    n_labels: int,
) -> dict:
    """
    Apply pre-registered V1-V5 criteria.

    V1: nan_inf_count == 0              (from training log)
    V2: mean(last 50 steps) < mean(first 50 steps)
    V3: alpha_one_hot_frac < 0.9       (fp32 diagnostics, step 300)
    V4: logit_std_frac > 0.5           (fp32 diagnostics, step 300; > 50% of labels)
    V5: p_at_1_hits > prior_p_at_1_hits (integer, strict >; F2 fix)

    Returns dict with V1-V5 booleans, raw numbers, and UNMEASURED flags.
    """
    N = len(losses)
    first50 = losses[:50] if N >= 50 else losses
    last50  = losses[-50:] if N >= 50 else losses

    V1 = diag_step300.get("nan_inf_count", 0) == 0
    V2 = (sum(last50) / max(len(last50), 1)) < (sum(first50) / max(len(first50), 1))

    # V3/V4/V5 can be UNMEASURED if diagnostics are invalid
    if diag_step300.get("alpha_one_hot_frac") is None:
        V3_val = "UNMEASURED"
        V3 = False
    else:
        V3_val = float(diag_step300["alpha_one_hot_frac"])
        V3 = V3_val < 0.9

    if diag_step300.get("logit_std_frac") is None:
        V4_val = "UNMEASURED"
        V4 = False
    else:
        V4_val = float(diag_step300["logit_std_frac"])
        V4 = V4_val > 0.5

    # F2 fix: integer comparison
    cand_hits  = diag_step300.get("p_at_1_hits")
    prior_hits = diag_step300.get("prior_p_at_1_hits")
    if cand_hits is None or prior_hits is None:
        V5_val = "UNMEASURED"
        V5 = False
    else:
        V5 = int(cand_hits) > int(prior_hits)
        V5_val = f"{cand_hits}/{prior_hits}"

    # Escalation check: L fails V3 but argmax-alpha is a true label > 50% of docs
    argmax_true_frac = diag_step300.get("argmax_alpha_is_true_frac", 0.0)
    escalate_v3 = (not V3) and (isinstance(argmax_true_frac, float)
                                 and argmax_true_frac > 0.5)

    return {
        "V1": V1, "V2": V2, "V3": V3, "V4": V4, "V5": V5,
        "V3_value": V3_val,
        "V4_value": V4_val,
        "V5_value": V5_val,
        "mean_loss_first50":     sum(first50) / max(len(first50), 1),
        "mean_loss_last50":      sum(last50)  / max(len(last50), 1),
        "alpha_one_hot_frac":    diag_step300.get("alpha_one_hot_frac"),
        "mean_max_alpha":        diag_step300.get("mean_max_alpha"),
        "logit_std_frac":        diag_step300.get("logit_std_frac"),
        "p_at_1_hits":           cand_hits,
        "prior_p_at_1_hits":     prior_hits,
        "p_at_1":                diag_step300.get("p_at_1"),
        "prior_p_at_1":          diag_step300.get("prior_p_at_1"),
        "argmax_alpha_is_true_frac": argmax_true_frac,
        "escalate_v3":           escalate_v3,
        "viable":                V1 and V2 and V3 and V4 and V5,
    }


# ── Decision function ─────────────────────────────────────────────────────────

def run0_decision(
    candidate_results: dict,       # {variant: {V1, V2, V3, V4, V5, viable, ...}}
    candidates: list = ("L", "L2"),
    diag_hashes: dict = None,
) -> dict:
    """
    Scripted decision function (pre-registered rule).

    Status codes:
      DECIDED            — first viable candidate; chosen_variant = "L" or "L2"
      ESCALATE           — L fails V3 but argmax-alpha is true label >50%: user required
      EXTEND             — V2/V5-only failure at 300 steps; 1000-step extension eligible
                           (only if no other V fails for that candidate)
      NONE_VIABLE        — no candidate passed all criteria
      STOP_AND_ASK       — NONE_VIABLE with no EXTEND condition
      INVALID_DIAGNOSTICS — DiagnosticError was raised; measurement must be fixed

    Decision is NEVER based on which candidate scored higher.
    """
    # Check for invalid diagnostics first
    for v in candidates:
        r = candidate_results.get(v, {})
        if r.get("diagnostic_error"):
            return {
                "status":           "INVALID_DIAGNOSTICS",
                "chosen_variant":   None,
                "reason":           f"DiagnosticError for variant {v}: {r['diagnostic_error']}",
                "candidate_results": candidate_results,
                "rule_applied":     "F1/F2/F3 fix: diagnostics must be valid before decision",
                "diag_hashes":      diag_hashes,
            }

    # 1. L first: if viable (no V3 escalation), choose it
    for v in candidates:
        r = candidate_results.get(v, {})
        if r.get("viable", False) and not r.get("escalate_v3", False):
            return {
                "status":           "DECIDED",
                "chosen_variant":   v,
                "reason":           f"Variant {v} is first viable candidate in order {list(candidates)}",
                "candidate_results": candidate_results,
                "rule_applied":     "First viable of L→L2; never chosen by performance",
                "diag_hashes":      diag_hashes,
            }
        # Escalation: L fails V3 but argmax-alpha is a true label for >50% of val docs
        if v == candidates[0] and r.get("escalate_v3", False):
            return {
                "status":           "ESCALATE",
                "chosen_variant":   None,
                "reason":           (
                    f"Variant L fails V3 (alpha_one_hot_frac={r.get('V3_value')}) "
                    f"BUT argmax-alpha is the true label for "
                    f"{r.get('argmax_alpha_is_true_frac', 0)*100:.1f}% of val docs "
                    f"(> 50%). This may indicate the model is learning a valid prior "
                    f"but lacks doc-specific discrimination. User decision required."
                ),
                "candidate_results": candidate_results,
                "rule_applied":     "Pre-registered ESCALATE condition (not a silent L2 fallthrough)",
                "diag_hashes":      diag_hashes,
            }

    # 2. Check for V2/V5-only failure (extension eligible) — any candidate
    for v in candidates:
        r = candidate_results.get(v, {})
        v1 = r.get("V1", False); v2 = r.get("V2", False)
        v3 = r.get("V3", False); v4 = r.get("V4", False); v5 = r.get("V5", False)
        # Exactly V2 and/or V5 failing; all others passing
        if v1 and v3 and v4 and (not v2 or not v5):
            return {
                "status":           "EXTEND",
                "chosen_variant":   None,
                "reason":           (
                    f"Variant {v} fails only V2={v2}/V5={v5}; all other criteria pass. "
                    f"Pre-declared 1000-step extension is eligible for this candidate. "
                    f"User must approve before extension runs."
                ),
                "candidate_results": candidate_results,
                "rule_applied":     "Pre-registered EXTEND condition: V2/V5-only failure",
                "diag_hashes":      diag_hashes,
            }

    # 3. NONE_VIABLE
    return {
        "status":           "NONE_VIABLE",
        "chosen_variant":   None,
        "reason":           (
            f"No candidate ({list(candidates)}) passed all V1-V5 criteria. "
            f"See candidate_results for per-criterion evidence."
        ),
        "candidate_results": candidate_results,
        "rule_applied":     "STOP_AND_ASK: no viable candidate, no EXTEND condition, no ESCALATE",
        "diag_hashes":      diag_hashes,
    }


# ── Artifact completeness check ───────────────────────────────────────────────

REQUIRED_ARTIFACTS = [
    "run0_decision.json",
    "run0_val_sample_ids.json",
    "results_summary.json",
    "env.json",
    "resolved_config.json",
    "SHA256SUMS.txt",
    "variant_L/train.log",
    "variant_L/step0_diagnostics.json",
    "variant_L/step300_diagnostics.json",
    "variant_L/val_sample_predictions.json",
    "variant_L/checkpoints/step_150.pt",
    "variant_L/checkpoints/step_300.pt",
]

OPTIONAL_ARTIFACTS = [
    "variant_L2/train.log",
    "variant_L2/step0_diagnostics.json",
    "variant_L2/step300_diagnostics.json",
    "variant_L2/val_sample_predictions.json",
    "variant_L2/checkpoints/step_150.pt",
    "variant_L2/checkpoints/step_300.pt",
    "hrrm_code_check.json",
    "resume_check.json",
    "gradient_flow.json",
]


def check_artifact_completeness(run0_dir: str) -> Tuple[bool, list, list]:
    """
    Check that all preregistered artifacts are present.
    Returns (all_present, missing_required, missing_optional).
    """
    missing_req = [a for a in REQUIRED_ARTIFACTS
                   if not os.path.exists(os.path.join(run0_dir, a))]
    missing_opt = [a for a in OPTIONAL_ARTIFACTS
                   if not os.path.exists(os.path.join(run0_dir, a))]
    return (len(missing_req) == 0), missing_req, missing_opt


def verify_sha256sums(run0_dir: str) -> Tuple[bool, list]:
    """
    Verify SHA256SUMS.txt: every listed file must exist and have the correct hash.
    Returns (all_ok, failures).
    """
    sums_path = os.path.join(run0_dir, "SHA256SUMS.txt")
    if not os.path.exists(sums_path):
        return False, ["SHA256SUMS.txt is missing"]
    failures = []
    with open(sums_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("  ", 1)
            if len(parts) != 2:
                continue
            expected_hash, rel_path = parts
            full_path = os.path.join(run0_dir, rel_path)
            if not os.path.exists(full_path):
                failures.append(f"MISSING: {rel_path}")
            else:
                h = hashlib.sha256(open(full_path, "rb").read()).hexdigest()
                if h != expected_hash:
                    failures.append(f"HASH_MISMATCH: {rel_path}")
    return (len(failures) == 0), failures
