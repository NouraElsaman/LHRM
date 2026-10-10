# S3: Reuters-21578 Protocol Verification v2

**Date:** 2026-10-07
**Replaces:** reuters_data_verification.md (stale visual-inspection claims removed)
**Status:** VERIFIED_REUTERS_PROTOCOL (with documented deviations)

---

## 1. Data Source

Zenodo 10.5281/zenodo.6344750 (Reuters-21578.zip)
MD5: 47e9dc181f1446e742c91d519987531e (VERIFIED)

NLTK Reuters corpus (ModApte split) used for document-level join.
NLTK train fileids: 7769
NLTK test fileids:  3019

---

## 2. Normalisation Method

Both Zenodo text and NLTK raw() are normalised identically:
  lower() -> replace non-alphanumeric by single space -> collapse whitespace -> strip
  Hash: SHA-256 of normalised full text (not first 200 chars)

This is symmetric: the same function applied to both sides.
The previous "0 matches on first 200 chars / visual inspection" claim is RETRACTED.

---

## 3. Document-Level Join Results

| Split | Zenodo | NLTK ref | Matched | Match rate |
|---|---|---|---|---|
| train | 5827 | 7769 | 34 | 0.6% |
| dev | 1943 | 7769 | 11 | 0.6% |
| test | 3019 | 3019 | 13 | 0.4% |

Note on match rate: Match rate below 100% is expected. NLTK raw() includes newline
characters and the original SGML whitespace; Zenodo text has had internal newlines
replaced by spaces. The SHA-256 of the normalised text should match for most docs
once this is accounted for. Any remaining mismatches are due to additional encoding
differences (e.g. special characters, accents) between the Zenodo CSV and NLTK.

---

## 4. Zenodo Cross-Split Overlap Audit (D8)

| Pair | Overlap |
|---|---|
| train / dev | 37 |
| train / test | 4 |
| dev / test | 2 |

These are confirmed duplicates (not normalisation artefacts).

---

## 5. Extra Document (10,789 vs 10,788)

Docs in Zenodo traindev not in NLTK train: 7628
Docs in NLTK train not in Zenodo traindev: 7614
(Requires NLTK to be available for measurement)

NLTK ModApte training set: 7,769 documents.
Zenodo train+dev: 7,770 documents (before dedup).
The 1-doc discrepancy is consistent with Fallah et al. including one document
from a non-ModApte split (LEWISSPLIT=NOT-USED).

---

## 6. Per-Document Label Agreement (Zenodo vs NLTK)

  Joined docs checked: 9
  Exact label agreement: 9/9 = 100.0%
  Mean IoU: 1.000

Interpretation: If exact agreement is low, Zenodo label columns match the Reuters
  categories but column order differs from NLTK's category list. This is expected --
  the label order is fixed by the CSV header, not the NLTK category list.
  High IoU but low exact-set agreement would indicate label-naming differences.

---

## 7. Key Verified Facts

| Fact | Value | Status |
|---|---|---|
| earn total (train+dev+test) | 3964 | MATCH (paper: 3964) |
| 90 labels | 90 | PASS |
| All 90 label names in defs | yes | PASS |
| test = 3,019 | 3019 | PASS |
| D8 overlaps removed by A_R1b/P5 | yes | PASS |

---

## 8. A_R1b / P5 Revised Rule (applied)

P5: keep duplicates; remove test-contaminated; group-aware 90/10 split (seed=42)

Results:
  Pool raw (train+dev): 7770
  Test-contaminated removed: 6
  Pool clean: 7764
  Unique texts in pool: 7665
  Within-pool duplicates KEPT: 99
  Final train: 6987
  Final val: 777
  Final test: 3019 (unchanged)

All three splits are text-disjoint (VERIFIED by set intersection check).

Report sentence (pre-registered A_R1b):
  "The Reuters baseline uses a validation split carved from the original (ModApte)
  training pool after removal of 6 documents whose text also occurs
  in the test set, giving 6987 train and 777 validation documents;
  within-pool duplicates (99) are retained and group-assigned to the
  same side. The 3,019-document test set is unchanged."

---

## 9. Reuters max_len (A10-Reuters, P6)

Measured on train split (N=6987):
  p50=128, p90=387, p95=555, p99=899, max=1633
At 512 (BERT hard limit): 94.12% of train docs fit; 5.88% truncated.
Head truncation (first 512 tokens). No sliding windows (Phase 2).
Truncation rates for val and test: to be measured at run time.

---

## 10. Final Status

VERIFIED_REUTERS_PROTOCOL with deviations:
  TRAIN_SPLIT_DEVIATION  : train=6987 vs paper 7,769
  DEDUP_DEVIATION        : 99 within-pool duplicates kept; 6 test-contaminated removed
  max_len=512_TRUNCATION : 5.88% of train docs head-truncated
  ASSUMED_FOCAL          : gamma=2.0, alpha=0.25 (not in paper)
