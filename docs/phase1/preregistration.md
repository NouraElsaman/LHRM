# Pre-Registration Document

**Date:** 2026-10-06
**Status:** PRE-REGISTERED (do not modify after RUN A begins)

---

## 1. Primary Configuration

| Item | Value |
|---|---|
| Model | LHRM (Eqs. 1-8) with BERT-base-uncased |
| Architecture variant | Chosen by V1-V5 screening on Reuters (not by metric) |
| Loss | Focal loss (gamma=2.0, alpha=0.25) -- ASSUMED (A9) |
| AAPD max_len | 512 (A10) |
| Reuters max_len | PENDING measurement |
| Dynamic padding | Yes (N7 fix) |
| Seed | 42 |
| HRRM | Fitted on validation predictions (A17) |
| Tie-breaking | Seeded-random (A18) |
| Evaluation model | Final epoch checkpoint (A19) |
| best.pt | DIAGNOSTIC ONLY -- never reported as headline |

---

## 2. Pre-Registered Status Rule

MATCH:     |Delta| <= 1.0 pp AND paper value inside our 95% CI
CLOSE:     |Delta| <= 2.5 pp
DIFFERENT: |Delta| > 2.5 pp
NOT REPRODUCIBLE: Required ingredient unavailable or run cannot complete

Judge pattern across P@1/P@3/P@5/nDCG@3/nDCG@5.

---

## 3. Paper Reference Values (LHRM, Table 3)

AAPD: P@1=86.90, P@3=62.07, P@5=42.02, nDCG@3=82.11, nDCG@5=85.63
Reuters: P@1=94.47, P@3=37.51, P@5=23.31, nDCG@3=95.67, nDCG@5=96.10

---

## 4. Discrepancy Investigation Order (pre-registered)

If results are DIFFERENT, investigate one factor at a time:
1. Labels/split (A_R1b, D3)
2. Preprocessing (A_R2, max_len)
3. Label definitions (A7, A5b)
4. Metric formula (Eqs. 13-16 vs legacy)
5. HRRM protocol (A17; compare leaky diagnostic)
6. Loss assumption (A20 BCE control)
7. Architecture reading (A5b, Variant R fallback)
8. max_len sensitivity
9. Training duration (epochs A15)
10. Seed sensitivity

---

## 5. Run Order (pre-registered)

RUN 0: DIAGNOSTIC GPU smoke (Reuters; screening V1-V5; DIAGNOSTIC_ONLY)
FREEZE-B after RUN 0
RUN A: Reuters baseline (train=6,910 A_R1b; annotate TRAIN_SPLIT_DEVIATION)
RUN A-BCE: Reuters BCE control (A20)
RUN B: AAPD baseline (ONLY if VERIFIED_LABEL_MAPPING; currently BLOCKED)
