# S1: AAPD Label Mapping — Purity-of-Mapping Analysis (P8 Correction)

**Date:** 2026-10-07
**Replaces:** The "column purity" metric in s1_large_join_analysis.md (which measured
  whether a column name matches its own true category -- irrelevant if headers are
  mis-assigned).

---

## Correct Question

For each CSV column C, which true arXiv label L does it consistently map to,
and with what purity (fraction of matched docs where C_active AND true_label = L)?
Is the overall map injective (each column maps to a different true label)?

---

## Data (29 matched documents from S1 large join)

Source: docs/phase1/s1_arxiv_join_results.json

---

## Observed Column -> True-Label Map

For columns with n >= 2 matched documents:

| CSV Column    | True arXiv Cat | Purity | n_docs | Injective? |
|---|---|---|---|---|
| cmp-lg        | cs.IT          | 14/14 = 100% | 14 | yes |
| cond-mat.dis-nn | cs.IT        | 14/14 = 100% | 14 | SAME as cmp-lg |
| cond-mat.stat-mech | cs.LG+stat.ML | 2/2 (both) | 2  | ambiguous |
| cs.cc         | cs.LG+stat.ML  | 2/2 (both) | 2  | SAME as cond-mat.stat-mech |
| cs.ds         | cs.CL          | 4/4 = 100% | 4  | yes |
| cs.fl         | cs.CR          | 5/5 = 100% | 5  | yes (second: cs.IT 4/5) |
| cs.hc         | cs.DC          | 2/2 = 100% | 2  | yes |
| cs.it         | cs.IR          | 3/3 = 100% | 3  | yes |
| cs.ma         | cs.CY          | 2/2 = 100% | 2  | yes |
| math.lo       | cs.DL          | 3/3 = 100% | 3  | yes |

Note: "cond-mat.stat-mech" and "cs.cc" both co-occur 100% with (cs.LG, stat.ML)
  and always appear together -- another byte-identical pair candidate.

---

## Injectivity Assessment

Among the 10 columns with n>=2:
- cmp-lg and cond-mat.dis-nn both map to cs.IT -> NOT injective for this pair.
- cond-mat.stat-mech and cs.cc both map to (cs.LG, stat.ML) -> NOT injective.
- All other mapped columns have unique targets -> injective.

This is consistent with a systematic PERMUTATION of the 54 labels with some
column pairs pointing to the same true label (possibly because the true labels
themselves have near-identical document distributions -- e.g., cs.LG and stat.ML
often co-occur).

---

## Coverage

29 matched documents cover only 25 of 54 CSV columns (as active columns).
The mapping for the remaining 29 columns is unknown from this join.
Columns with n=1 are single-document anecdotal; not reported as verified.

---

## Verdict

Evidence is CONSISTENT WITH a systematic permutation of the 54 label names.
The mapped columns show 100% purity (or near-100%) in ALL cases with n>=2.
This strongly suggests the mapping is deterministic, not random noise per document.

However:
1. Coverage is only 25/54 columns and only 29 documents.
2. Acceptance threshold is purity >= 99.5% over >= 5,000 matched training documents
   (unchanged from preregistration).
3. The current evidence is INSUFFICIENT for VERIFIED_LABEL_MAPPING.
4. The full arXiv metadata join (Option A, Kaggle) is required to close D3.

---

## Implication for D3 Decision

The Nature of the error is a systematic permutation, not random per-document noise.
This means:
- Option A (Kaggle join) is likely to succeed in recovering the true mapping.
- The corrected AAPD could in principle be used for baseline training.
- BUT: the 5,000-document threshold and 99.5% purity must be met first.

See user_decision_D3.md for options.
