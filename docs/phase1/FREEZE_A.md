# FREEZE-A: Pre-RUN-0 Configuration Snapshot

**Date:** 2026-10-06
**Status:** FROZEN (do not modify)
**Purpose:** Records the exact configuration used for RUN 0 (screening smoke run on Reuters).
  Architecture variant (CF6) is NOT yet chosen — to be filled in after RUN 0.

---

## Dataset: Reuters

| Parameter | Value | Source |
|---|---|---|
| Zenodo ZIP MD5 | 47e9dc181f1446e742c91d519987531e | Verified |
| Split protocol | A_R1b: merge+dedup+resplit, seed=42 | prepare_reuters.py |
| train | 6,910 docs | A_R1b |
| val | 767 docs | A_R1b |
| test | 3,019 docs | Zenodo test (untouched) |
| n_labels | 90 | reuters_labels_definitions.json |
| label_order_hash | (see data/processed/Reuters/label_order.hash) | |
| max_len | 512 | A10-Reuters (BERT hard limit; 5.88% truncated) |
| dynamic_padding | True | N7 fix |
| text_field | text | A_R2 |
| token_length_stats | p50=128, p90=387, p95=555, p99=899, max=1633 | |

## Dataset: AAPD

| Parameter | Value | Source |
|---|---|---|
| Status | BLOCKED | UNRESOLVED_LABEL_MAPPING |
| Reason | arXiv API join shows systematic label mis-assignment (5/5 test docs wrong) | S1 Route 4 |
| Large join (300 docs) | IN PROGRESS (background task) | |

---

## Model Configuration

| Parameter | Value | Source |
|---|---|---|
| bert_model | bert-base-uncased | PAPER-EXPLICIT |
| hidden_size | 768 | BERT default |
| Architecture | TBD (chosen by V1-V5 screening in RUN 0) | A6 |
| W shape | nn.Linear(768, 1, bias=True) shared | A5 |
| No Q/K/V projections in Eq. 6 | confirmed | A7 |
| Document dependence via alpha only | analytic fact | A8 |

---

## Training Configuration

| Parameter | Value | Source |
|---|---|---|
| num_epochs | 10 | A15 (PAPER-EXPLICIT for Reuters) |
| batch_size | 8 | reuters_baseline.yaml |
| learning_rate | 2.0e-05 | reuters_baseline.yaml |
| weight_decay | 0.01 | reuters_baseline.yaml |
| loss | focal_loss(gamma=2.0, alpha=0.25) | A9 (ASSUMED) |
| seed | 42 | |
| amp | True | |

---

## HRRM Configuration

| Parameter | Value | Source |
|---|---|---|
| n_estimators | 100 | A13 (PAPER-EXPLICIT) |
| learning_rate | 0.1 | A13 (PAPER-EXPLICIT) |
| max_depth | 3 | A13 (PAPER-EXPLICIT) |
| subsample | 1.0 | A13 (PAPER-EXPLICIT) |
| criterion | friedman_mse | A13 (PAPER-EXPLICIT) |
| fitting_data | validation predictions | A17 (non-leaky) |
| tie_breaking | seeded-random | A18 |

---

## Evaluation

| Parameter | Value | Source |
|---|---|---|
| Metrics | P@1, P@3, P@5, nDCG@3, nDCG@5 | Paper Eqs. 13-16 |
| Metric formula | Correct (divisor=k for P@k) | metrics.py |
| Evaluation model | final epoch | A19 |
| best.pt | DIAGNOSTIC ONLY | A19 |
| tie_breaking | seeded-random (seed=42 XOR doc_idx) | A18 |

---

## Open Items Before RUN A Can Begin

| ID | Item |
|---|---|
| CF6 | Architecture variant (L vs L2) — chosen by RUN 0 V1-V5 |
| CF8 | FREEZE-B (after architecture choice) |

---

## Decisions Registered at Freeze Time

A1, A3, A5, A6, A7, A8, A9, A10, A12, A13, A15, A17, A18, A19, A20, A_R1b, A_R2
(All in decision_registry.md v3)

---

## Test Suite Status at Freeze

53 passed, 1 skipped, 0 failed (2026-10-06)
Skipped: test_end_to_end_tiny_smoke_cpu (requires pretrained BERT weights for tiny model)
