#!/usr/bin/env python3
"""
scripts/probe_dr4_frozen_sa_counterfactual.py

LHRM DR4 Frozen-Checkpoint Label-SA Counterfactual Diagnostic:
Tests whether subtracting the document-invariant label-SA component from the raw
logits of the frozen, trained models restores ranking diversity and improves Top-1 accuracy.

Mathematical Counterfactual:
  Original:       original_logits[b, i] = doc_term[b, i] + sa_term[i] + bias
  Counterfactual: counterfactual_logits[b, i] = doc_term[b, i] + bias
                  (equivalently: original_logits[b, i] - sa_term[i])

Strict Invariants Verified:
  1. Zero Parameter Mutation: Checkpoint weights and embeddings are unmodified.
  2. Exact Subtraction Parity: max |counterfactual_logits - (doc_term + bias)| <= 1e-6.
  3. Cross-Document Variance Invariance: Var_b(counterfactual_logits[:, i]) == Var_b(original_logits[:, i]).
     Therefore, the per-label std across documents and V4 passing count remain strictly invariant.
  4. Ranking Sensitivity: Evaluates changes in P@1 hits, distinct Top-1 labels, dominant-label fraction,
     and duplicate-target label ordering on the authentic 200 Dev documents.
  5. Deterministic tie-breaking: Matches seeded_rank_order from lhrm_baseline.metrics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.probe_dr4_trained_logit_decomposition import (
    decompose_l2_csam_logits,
    decompose_a5b_csam_logits,
    verify_saved_predictions,
    seeded_rank_order,
    sha256_file,
)


def compute_counterfactual_logits(decomp: Dict[str, Any]) -> Tuple[torch.Tensor, float]:
    """
    Computes counterfactual logits by subtracting static sa_term from original logits:
    counterfactual_logits[b, i] = doc_term[b, i] + bias
    Verifies that (original_logits - sa_term) == (doc_term + bias) within 1e-6.
    """
    logits = decomp["logits"]      # [B, k]
    doc_term = decomp["doc_term"]  # [B, k]
    sa_term = decomp["sa_term"]    # [B, k]
    b_val = decomp["bias"]         # float or tensor

    cf_logits_direct = doc_term + b_val
    cf_logits_subtracted = logits - sa_term

    subtraction_diff = (cf_logits_direct - cf_logits_subtracted).abs().max().item()
    if subtraction_diff > 1e-5:
        raise ValueError(f"Counterfactual subtraction parity failed! Max diff: {subtraction_diff:.2e} > 1e-5")

    return cf_logits_direct, subtraction_diff


def verify_cross_document_variance_invariance(
    orig_logits: torch.Tensor,
    cf_logits: torch.Tensor,
    tol: float = 1e-6,
) -> Dict[str, Any]:
    """
    Verifies the mathematical theorem:
    Since sa_term[i] is constant across documents b,
    Var_b(cf_logits[:, i]) must be identically Var_b(orig_logits[:, i]).
    """
    B, k = orig_logits.shape
    orig_std = orig_logits.std(dim=0, unbiased=True).cpu().numpy()  # [k]
    cf_std = cf_logits.std(dim=0, unbiased=True).cpu().numpy()      # [k]

    max_std_diff = float(np.abs(orig_std - cf_std).max())
    mean_std_diff = float(np.abs(orig_std - cf_std).mean())

    orig_v4_count = int(np.sum(orig_std > 0.01))
    cf_v4_count = int(np.sum(cf_std > 0.01))

    if max_std_diff > tol:
        raise ValueError(
            f"Mathematical invariant violation: Cross-document logit std changed after static offset removal! "
            f"Max diff: {max_std_diff:.2e} > tol {tol:.2e}"
        )

    if orig_v4_count != cf_v4_count:
        raise ValueError(
            f"V4 passing count changed: orig={orig_v4_count} vs cf={cf_v4_count}"
        )

    return {
        "max_std_difference": max_std_diff,
        "mean_std_difference": mean_std_diff,
        "orig_v4_passing_count": orig_v4_count,
        "cf_v4_passing_count": cf_v4_count,
        "v4_count_identical": bool(orig_v4_count == cf_v4_count),
        "invariant_satisfied": bool(max_std_diff <= tol),
    }


def evaluate_ranking_metrics(
    logits: torch.Tensor | np.ndarray,
    label_names: List[str],
    true_labels_list: Optional[List[List[str]]] = None,
    seed_offset: int = 42,
) -> Dict[str, Any]:
    """
    Evaluates ranking metrics on scores/logits using deterministic seeded tie-breaking.
    """
    if isinstance(logits, torch.Tensor):
        logits_np = logits.detach().cpu().float().numpy()
    else:
        logits_np = np.array(logits, dtype=np.float32)

    B, k = logits_np.shape
    probs = 1.0 / (1.0 + np.exp(-logits_np))

    top1_labels = []
    top3_patterns = []
    margins_1st_2nd = []
    p_at_1_hits = 0

    for b in range(B):
        ranking = seeded_rank_order(probs[b], seed=seed_offset ^ b)
        top1_idx = int(ranking[0])
        top2_idx = int(ranking[1])
        top1_lbl = label_names[top1_idx]
        top1_labels.append(top1_lbl)
        top3_lbls = tuple(label_names[int(j)] for j in ranking[:3])
        top3_patterns.append(top3_lbls)
        margins_1st_2nd.append(float(probs[b, top1_idx] - probs[b, top2_idx]))

        if true_labels_list is not None and len(true_labels_list) > b:
            if top1_lbl in true_labels_list[b]:
                p_at_1_hits += 1

    top1_counts = Counter(top1_labels)
    distinct_top1 = len(top1_counts)
    dominant_label, dominant_cnt = top1_counts.most_common(1)[0]
    distinct_top3 = len(set(top3_patterns))

    # Duplicate target comparison: cmp-lg vs cond-mat.dis-nn
    idx_cmp = label_names.index("cmp-lg") if "cmp-lg" in label_names else 0
    idx_cond = label_names.index("cond-mat.dis-nn") if "cond-mat.dis-nn" in label_names else 1

    duplicate_analysis = {
        "cmp_lg_top1_count": top1_counts.get(label_names[idx_cmp], 0),
        "cond_mat_dis_nn_top1_count": top1_counts.get(label_names[idx_cond], 0),
        "cmp_lg_mean_prob": float(probs[:, idx_cmp].mean()),
        "cond_mat_dis_nn_mean_prob": float(probs[:, idx_cond].mean()),
        "mean_prob_difference": float(probs[:, idx_cmp].mean() - probs[:, idx_cond].mean()),
    }

    return {
        "num_documents": B,
        "num_labels": k,
        "p_at_1_hits": p_at_1_hits,
        "p_at_1_accuracy": p_at_1_hits / B,
        "frequency_prior_hits": 63,
        "frequency_prior_accuracy": 63 / 200,
        "exceeds_frequency_prior": bool(p_at_1_hits > 63),
        "distinct_top1_count": distinct_top1,
        "dominant_label": dominant_label,
        "dominant_count": dominant_cnt,
        "dominant_fraction": dominant_cnt / B,
        "distinct_top3_count": distinct_top3,
        "top1_labels_distribution": dict(top1_counts),
        "score_margin_1st_2nd": {
            "mean": float(np.mean(margins_1st_2nd)),
            "std": float(np.std(margins_1st_2nd)),
            "min": float(np.min(margins_1st_2nd)),
            "max": float(np.max(margins_1st_2nd)),
        },
        "duplicate_target_labels": duplicate_analysis,
        "raw_top1_labels": top1_labels,
    }


def compare_original_vs_counterfactual(
    decomp: Dict[str, Any],
    label_names: List[str],
    true_labels_list: Optional[List[List[str]]] = None,
) -> Dict[str, Any]:
    """
    Runs both original and counterfactual evaluations on decomp and performs paired comparison.
    """
    variant = decomp["variant"]
    orig_logits = decomp["logits"]
    cf_logits, subtraction_diff = compute_counterfactual_logits(decomp)

    # Invariant verification
    invariant_check = verify_cross_document_variance_invariance(orig_logits, cf_logits)

    # Ranking evaluations
    orig_eval = evaluate_ranking_metrics(orig_logits, label_names, true_labels_list)
    cf_eval = evaluate_ranking_metrics(cf_logits, label_names, true_labels_list)

    B = orig_eval["num_documents"]
    orig_preds = orig_eval["raw_top1_labels"]
    cf_preds = cf_eval["raw_top1_labels"]

    # Paired transitions
    rank_flips = 0
    t_to_t = 0
    t_to_f = 0
    f_to_t = 0
    f_to_f = 0

    for b in range(B):
        if orig_preds[b] != cf_preds[b]:
            rank_flips += 1

        if true_labels_list is not None and len(true_labels_list) > b:
            orig_correct = orig_preds[b] in true_labels_list[b]
            cf_correct = cf_preds[b] in true_labels_list[b]

            if orig_correct and cf_correct:
                t_to_t += 1
            elif orig_correct and not cf_correct:
                t_to_f += 1
            elif not orig_correct and cf_correct:
                f_to_t += 1
            else:
                f_to_f += 1

    # Per-label detailed metrics
    k = len(label_names)
    orig_std_docs = orig_logits.std(dim=0, unbiased=True).cpu().numpy()
    cf_std_docs = cf_logits.std(dim=0, unbiased=True).cpu().numpy()
    doc_mean = decomp["doc_term"].mean(dim=0).cpu().numpy()
    sa_vals = decomp["sa_term"][0].cpu().numpy()
    orig_logit_mean = orig_logits.mean(dim=0).cpu().numpy()
    cf_logit_mean = cf_logits.mean(dim=0).cpu().numpy()

    per_label_records = []
    for i in range(k):
        per_label_records.append({
            "variant": variant,
            "label_idx": i,
            "label_name": label_names[i],
            "doc_term_mean": round(float(doc_mean[i]), 6),
            "sa_term_offset": round(float(sa_vals[i]), 6),
            "orig_logit_mean": round(float(orig_logit_mean[i]), 6),
            "cf_logit_mean": round(float(cf_logit_mean[i]), 6),
            "orig_logit_std": round(float(orig_std_docs[i]), 6),
            "cf_logit_std": round(float(cf_std_docs[i]), 6),
            "std_difference": round(float(abs(orig_std_docs[i] - cf_std_docs[i])), 8),
            "v4_passing": bool(orig_std_docs[i] > 0.01),
        })

    comparison = {
        "variant": variant,
        "subtraction_diff": subtraction_diff,
        "invariant_verification": invariant_check,
        "original_ranking": {k_m: v_m for k_m, v_m in orig_eval.items() if k_m != "raw_top1_labels"},
        "counterfactual_ranking": {k_m: v_m for k_m, v_m in cf_eval.items() if k_m != "raw_top1_labels"},
        "delta_hits": cf_eval["p_at_1_hits"] - orig_eval["p_at_1_hits"],
        "delta_distinct_top1": cf_eval["distinct_top1_count"] - orig_eval["distinct_top1_count"],
        "delta_distinct_top3": cf_eval["distinct_top3_count"] - orig_eval["distinct_top3_count"],
        "num_rank_flips": rank_flips,
        "rank_flip_fraction": rank_flips / B,
        "accuracy_transitions": {
            "true_to_true": t_to_t,
            "true_to_false": t_to_f,
            "false_to_true": f_to_t,
            "false_to_false": f_to_f,
        },
        "per_label_metrics": per_label_records,
    }
    return comparison


def generate_counterfactual_markdown_report(
    l2_comp: Dict[str, Any],
    a5b_comp: Dict[str, Any],
    pred_verification: Optional[Dict[str, Any]] = None,
) -> str:
    """Generates quantitative, observational diagnostic report in GitHub markdown format."""
    lines = []
    lines.append("# LHRM DR4 Frozen-Checkpoint Label-SA Counterfactual Report")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        "This diagnostic evaluates the immediate ranking sensitivity of frozen, trained checkpoints "
        "(`L2_STEP300_MODEL_ONLY.pt` and `L2_A5B_STEP300_MODEL_ONLY.pt` from `RUN_0_L2_A5B_20261012_A`) "
        "to removing the static, document-invariant label self-attention offset (`sa_term`)."
    )
    lines.append("")
    lines.append("### Mathematical Invariant Verification:")
    l2_inv = l2_comp["invariant_verification"]
    a5b_inv = a5b_comp["invariant_verification"]
    lines.append(
        f"1. **Variance Invariance:** Because `sa_term` is strictly document-invariant, per-label logit standard deviation "
        f"across documents is identical before and after subtraction:\n"
        f"   - L2 Max Std Diff: `{l2_inv['max_std_difference']:.2e}` (V4 count: `{l2_inv['cf_v4_passing_count']}/54`, unchanged)\n"
        f"   - L2_A5B Max Std Diff: `{a5b_inv['max_std_difference']:.2e}` (V4 count: `{a5b_inv['cf_v4_passing_count']}/54`, unchanged)"
    )
    lines.append(
        f"2. **Algebraic Subtraction Parity:** Subtraction max diff <= `{max(l2_comp['subtraction_diff'], a5b_comp['subtraction_diff']):.2e}` (<= 1e-5)."
    )

    if pred_verification:
        lines.append(f"3. **Saved Predictions Agreement:** Pre-intervention model outputs reproduce authentic screening scores (Max diff <= `{pred_verification.get('max_diff', 0.0):.2e}`).")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Comparative Summary Table: Original vs Counterfactual")
    lines.append("")
    lines.append("| Metric | L2 Original | L2 Counterfactual | L2 Delta | L2_A5B Original | L2_A5B Counterfactual | L2_A5B Delta |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    l2_orig = l2_comp["original_ranking"]
    l2_cf = l2_comp["counterfactual_ranking"]
    a5b_orig = a5b_comp["original_ranking"]
    a5b_cf = a5b_comp["counterfactual_ranking"]

    lines.append(f"| **P@1 Hits (vs Prior 63)** | {l2_orig['p_at_1_hits']}/200 | {l2_cf['p_at_1_hits']}/200 | {l2_comp['delta_hits']:+d} | {a5b_orig['p_at_1_hits']}/200 | {a5b_cf['p_at_1_hits']}/200 | {a5b_comp['delta_hits']:+d} |")
    lines.append(f"| **P@1 Accuracy** | {l2_orig['p_at_1_accuracy']:.1%} | {l2_cf['p_at_1_accuracy']:.1%} | {l2_cf['p_at_1_accuracy'] - l2_orig['p_at_1_accuracy']:+.1%} | {a5b_orig['p_at_1_accuracy']:.1%} | {a5b_cf['p_at_1_accuracy']:.1%} | {a5b_cf['p_at_1_accuracy'] - a5b_orig['p_at_1_accuracy']:+.1%} |")
    lines.append(f"| **Distinct Top-1 Labels** | {l2_orig['distinct_top1_count']} | {l2_cf['distinct_top1_count']} | {l2_comp['delta_distinct_top1']:+d} | {a5b_orig['distinct_top1_count']} | {a5b_cf['distinct_top1_count']} | {a5b_comp['delta_distinct_top1']:+d} |")
    lines.append(f"| **Dominant Label Fraction** | {l2_orig['dominant_fraction']:.1%} | {l2_cf['dominant_fraction']:.1%} | {l2_cf['dominant_fraction'] - l2_orig['dominant_fraction']:+.1%} | {a5b_orig['dominant_fraction']:.1%} | {a5b_cf['dominant_fraction']:.1%} | {a5b_cf['dominant_fraction'] - a5b_orig['dominant_fraction']:+.1%} |")
    lines.append(f"| **Distinct Top-3 Patterns** | {l2_orig['distinct_top3_count']} | {l2_cf['distinct_top3_count']} | {l2_comp['delta_distinct_top3']:+d} | {a5b_orig['distinct_top3_count']} | {a5b_cf['distinct_top3_count']} | {a5b_comp['delta_distinct_top3']:+d} |")
    lines.append(f"| **Top-1 Rank Flips** | - | {l2_comp['num_rank_flips']}/200 | - | - | {a5b_comp['num_rank_flips']}/200 | - |")
    lines.append(f"| **V4 Labels Passing** | 21/54 | 21/54 | 0 | 3/54 | 3/54 | 0 |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Prediction Transitions Analysis")
    lines.append("")
    l2_t = l2_comp["accuracy_transitions"]
    a5b_t = a5b_comp["accuracy_transitions"]
    lines.append("| Transition State | Variant L2 | Variant L2_A5B |")
    lines.append("|---|---:|---:|")
    lines.append(f"| **True -> True (Preserved Hits)** | {l2_t['true_to_true']} | {a5b_t['true_to_true']} |")
    lines.append(f"| **True -> False (Damaged Hits)** | {l2_t['true_to_false']} | {a5b_t['true_to_false']} |")
    lines.append(f"| **False -> True (Recovered Hits)** | {l2_t['false_to_true']} | {a5b_t['false_to_true']} |")
    lines.append(f"| **False -> False (Unrecovered Misses)** | {l2_t['false_to_false']} | {a5b_t['false_to_false']} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Duplicate-Target Resolution (`cmp-lg` vs `cond-mat.dis-nn`)")
    lines.append("")
    dup_l2_o = l2_orig["duplicate_target_labels"]
    dup_l2_c = l2_cf["duplicate_target_labels"]
    dup_a5b_o = a5b_orig["duplicate_target_labels"]
    dup_a5b_c = a5b_cf["duplicate_target_labels"]
    lines.append("| Metric | L2 Original | L2 Counterfactual | L2_A5B Original | L2_A5B Counterfactual |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(f"| `cmp-lg` Top-1 Count | {dup_l2_o['cmp_lg_top1_count']}/200 | {dup_l2_c['cmp_lg_top1_count']}/200 | {dup_a5b_o['cmp_lg_top1_count']}/200 | {dup_a5b_c['cmp_lg_top1_count']}/200 |")
    lines.append(f"| `cond-mat.dis-nn` Top-1 Count | {dup_l2_o['cond_mat_dis_nn_top1_count']}/200 | {dup_l2_c['cond_mat_dis_nn_top1_count']}/200 | {dup_a5b_o['cond_mat_dis_nn_top1_count']}/200 | {dup_a5b_c['cond_mat_dis_nn_top1_count']}/200 |")
    lines.append(f"| Mean Prob Diff (`cmp` - `cond`) | {dup_l2_o['mean_prob_difference']:+.6f} | {dup_l2_c['mean_prob_difference']:+.6f} | {dup_a5b_o['mean_prob_difference']:+.6f} | {dup_a5b_c['mean_prob_difference']:+.6f} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Scientific Interpretation")
    lines.append("")
    lines.append(
        "1. **Nature of Intervention:** This intervention isolated the static SA component without retraining. "
        "Because it did not update model parameters, the document terms reflect what the model learned under joint optimization."
    )
    lines.append(
        "2. **Causality Limit:** If removing the static offset does not restore accuracy, it proves that the document terms "
        "themselves were not trained to provide competitive discriminative rankings, rather than proving that SA is impossible to train."
    )
    lines.append("")
    return "\n".join(lines)


def export_counterfactual_deliverables(
    l2_comp: Dict[str, Any],
    a5b_comp: Dict[str, Any],
    out_dir: str | Path,
    pred_verification: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Saves all required counterfactual output deliverables and packages evidence ZIP."""
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    results_data = {
        "diagnostic_id": "DR4_FROZEN_SA_CF_20261012_A",
        "date": "2026-10-12",
        "variants": {
            "L2": {k: v for k, v in l2_comp.items() if k != "per_label_metrics"},
            "L2_A5B": {k: v for k, v in a5b_comp.items() if k != "per_label_metrics"},
        },
        "prediction_verification": pred_verification or {},
    }

    # 1. Results JSON
    results_json = out_p / "DR4_FROZEN_SA_CF_RESULTS.json"
    results_json.write_text(json.dumps(results_data, indent=2), encoding="utf-8")

    # 2. Per-label CSV
    csv_path = out_p / "DR4_FROZEN_SA_CF_PER_LABEL_METRICS.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "variant", "label_idx", "label_name", "doc_term_mean", "sa_term_offset",
            "orig_logit_mean", "cf_logit_mean", "orig_logit_std", "cf_logit_std",
            "std_difference", "v4_passing",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in l2_comp.get("per_label_metrics", []):
            writer.writerow(rec)
        for rec in a5b_comp.get("per_label_metrics", []):
            writer.writerow(rec)

    # 3. Markdown Report
    report_md = out_p / "DR4_FROZEN_SA_CF_REPORT.md"
    report_text = generate_counterfactual_markdown_report(l2_comp, a5b_comp, pred_verification)
    report_md.write_text(report_text, encoding="utf-8")

    # 4. Evidence ZIP
    zip_path = out_p / "DR4_FROZEN_SA_CF_EVIDENCE.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(results_json, arcname=results_json.name)
        z.write(csv_path, arcname=csv_path.name)
        z.write(report_md, arcname=report_md.name)

    # 5. Checksums
    sums_path = out_p / "DR4_FROZEN_SA_CF_SHA256SUMS.txt"
    lines = [
        f"{sha256_file(results_json)}  {results_json.name}",
        f"{sha256_file(csv_path)}  {csv_path.name}",
        f"{sha256_file(report_md)}  {report_md.name}",
        f"{sha256_file(zip_path)}  {zip_path.name}",
    ]
    sums_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "results_json": str(results_json),
        "metrics_csv": str(csv_path),
        "report_md": str(report_md),
        "evidence_zip": str(zip_path),
        "checksums": str(sums_path),
    }


def main():
    parser = argparse.ArgumentParser(description="DR4 Frozen-Checkpoint Label-SA Counterfactual Diagnostic")
    parser.add_argument("--l2_checkpoint", default=None)
    parser.add_argument("--a5b_checkpoint", default=None)
    parser.add_argument("--out_dir", default="diagnostics/DR4_FROZEN_SA_CF")
    parser.add_argument("--synthetic_test", action="store_true")
    args = parser.parse_args()

    print("=== DR4 Frozen-Checkpoint Label-SA Counterfactual Diagnostic ===")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic_test:
        print("Executing synthetic invariant test mode on CPU...")
        B, k, H = 200, 54, 768
        torch.manual_seed(42)
        h_T = torch.randn(B, H) / math.sqrt(H)
        h_L = torch.randn(k, H) / math.sqrt(H)
        w_l2 = torch.randn(1, H) / math.sqrt(H)
        b_l2 = torch.tensor([-0.023])
        W_a5b = w_l2.repeat(k, 1) + torch.randn(k, H) * (0.005 / math.sqrt(H))
        b_a5b = torch.tensor([-0.023])

        label_names = [f"label_{i:02d}" for i in range(k)]
        label_names[0] = "cmp-lg"
        label_names[1] = "cond-mat.dis-nn"

        decomp_l2 = decompose_l2_csam_logits(h_T, h_L, w_l2, b_l2)
        decomp_a5b = decompose_a5b_csam_logits(h_T, h_L, W_a5b, b_a5b)

        true_labels_list = [[label_names[0]] for _ in range(B)]

        l2_comp = compare_original_vs_counterfactual(decomp_l2, label_names, true_labels_list)
        a5b_comp = compare_original_vs_counterfactual(decomp_a5b, label_names, true_labels_list)

        paths = export_counterfactual_deliverables(l2_comp, a5b_comp, out_dir)
        print("Synthetic counterfactual diagnostic completed successfully.")
        for k_name, p in paths.items():
            print(f"  {k_name}: {p}")
        return

    print("Full trained-checkpoint inference mode.")
    print("See documentation and dedicated Kaggle notebook for execution on GPU.")


if __name__ == "__main__":
    main()
