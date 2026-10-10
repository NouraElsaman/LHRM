# S1: AAPD Label Mapping Verification

**Date:** 2026-10-06
**Status:** UNRESOLVED_LABEL_MAPPING
**Consequence:** AAPD training BLOCKED. No AAPD baseline numbers may be reported.

---

## Section 1: Provenance

**Dataset source:** Zenodo record 10.5281/zenodo.6344750
  Title: "Multi-label Datasets used in 'Adapting Transformers for Multi-Label Text Classification'"
  Authors: Haytame Fallah et al. (CIRCLE 2022)
  File: AAPD.zip (Zenodo MD5: e1419b5d03be572115287fcb80577c48)

**Local vs Zenodo:** CSV content (abstracts, binary labels, column names) is
  byte-identical. ZIP-level MD5 differs due to metadata only. Content = Zenodo.

**LHRM paper citation of AAPD:**
  The LHRM paper (Ayash et al. 2025) cites ref. [42] (Yang et al. SGM, COLING 2018)
  for the AAPD dataset — NOT Fallah et al. (ref. [31]).
  Therefore the claim "Zenodo data was used as-is by the LHRM paper" is UNSUPPORTED.
  This was an error in the previous version of this document and is retracted.

---

## Section 2: Routes Attempted

### Route 1: Zenodo record inspection

**Inspected:** Zenodo record description, file listing, version history.
**Result:** NOT FOUND.
  - No README, no label-order file, no conversion script in the Zenodo record.
  - Fallah et al. (CIRCLE 2022) paper has no public code repository.

### Route 2: SGM/Yang et al. (COLING 2018) — original AAPD authors

**Inspected:** GitHub repo `lancopku/SGM` (root + data/ directory).
**Root contents:** README.md, config.yaml, lr_scheduler.py, models/, opts.py,
  predict.py, preprocess.py, train.py, utils/
**Data:** No AAPD data file found. Repo contains only RCV1-V2.
**Result:** NOT FOUND.

### Route 3: Other public repositories citing AAPD

**Checked:** HP-AMR (Alibaba-NLP), LSAN (guoqunabc), CPC_NER (idiap)
**Result:** All returned HTTP 404 — data files not present or paths changed.
**Result:** NOT FOUND.

### Route 4: arXiv API document-level join

**Method:** Queried the arXiv public API (export.arxiv.org/api/query) with the
  first 12 words of each AAPD abstract, matched against API-returned entries by
  normalised text overlap (first 60 chars after lowercasing and stripping punctuation).
  Rate: 1 request per 3.5 seconds; 5 test documents matched and inspected.

**Critical finding — evidence of systematic label mis-assignment:**

| Doc | CSV active labels | arXiv true categories | Match? |
|---|---|---|---|
| test[0] | cs.cc, cs.dm | cs.CV, stat.ML | ❌ NO |
| test[1] | cmp-lg, cond-mat.dis-nn | cs.IT | ❌ NO (confirms same-signal anomaly) |
| test[2] | cs.ai, cs.ce | cs.AI, cs.DS | Partial (cs.ai≈cs.AI ✓, cs.ce≠cs.DS ❌) |
| test[3] | cs.cl, cs.cy, cs.na | math.CO, cs.DM, math.PR | ❌ ALL WRONG |
| test[4] | cmp-lg, cond-mat.dis-nn, cs.dl, cs.na | cs.IT, cs.NI, math.PR | ❌ SEVERE MISMATCH |

**arXiv API arxiv IDs confirmed:**
  test[0]: arxiv.org/abs/1410.8546v2
  test[1]: arxiv.org/abs/1411.5767v2
  test[2]: arxiv.org/abs/1411.0149v3
  test[3]: arxiv.org/abs/1310.2728v6
  test[4]: arxiv.org/abs/0904.1444v1

These matches are verified: the API returns the same abstract text as the CSV.

**Interpretation:**
- The label columns in the Zenodo CSV do NOT reliably correspond to the arXiv
  categories of the matching papers.
- This is not a naming/case issue. The mis-assignment is structural:
  a document about cs.CV/stat.ML is labelled cs.cc/cs.dm in the CSV.
- The 5-document sample is too small to determine whether this is a systematic
  permutation or random noise. A larger join (300 docs) is in progress as a
  background task; results will update this document when available.
- Routes 1–3 did not provide a corrective label list.
- CONCLUSION: We cannot verify that ANY of the 54 column names correctly
  identifies the data in that column.

**Route 4 status:** PARTIAL (large join in progress in background).
**Overall Route 4 result (so far):** Evidence of mis-assignment found;
  no corrective mapping obtainable from the API at this time without the
  full arXiv metadata snapshot (Kaggle dataset — LEVEL 2 task).

---

## Section 3: Corrections to Previous Version (N3)

The following errors from the previous version of this document are RETRACTED:

(a) "The CSV header order IS the authoritative label order" — RETRACTED.
    Headers are alphabetically sorted labels chosen by Fallah et al. The data
    in each column may not match the label name (confirmed by Route 4 evidence).

(b) "Zenodo data was used as-is by all papers including LHRM" — RETRACTED.
    LHRM cites SGM (ref. [42]) for AAPD; the Zenodo record is cited only as a
    transformer paper (ref. [31]). This claim is unsupported.

(c) "cs.it has 2,018 docs vs math.it 875, so cross-listing is violated — circular
    argument" — RETRACTED. This argument assumed the column names were correct,
    which is exactly what is in question.

(d) "OR-ing of retired codes" — RETRACTED as speculation. No evidence found.

(e) Status "UNRESOLVED_AS_IS" — REPLACED. Only two status values are used:
    VERIFIED_LABEL_MAPPING | UNRESOLVED_LABEL_MAPPING. Current status:
    UNRESOLVED_LABEL_MAPPING.

(f) "AAPD training is ALLOWED to proceed" — RETRACTED AND OVERRIDDEN.
    AAPD training is BLOCKED until VERIFIED_LABEL_MAPPING is achieved.

(g) "SGM repo only has RCV1-V2" — now confirmed by actual inspection (Route 2).

(h) `label_mapping.json` had `[list of 54 column names]` placeholder — CORRECTED.
    The file has been rewritten with actual column names and explicit
    _mapping_status = UNRESOLVED_LABEL_MAPPING.

---

## Section 4: Identical Column Pairs

Even if the overall label assignment were verified, two pairs are byte-identical:
  - cmp-lg / cond-mat.dis-nn: 17,152 docs each, 100% co-occurrence
  - math.co / math.it: 875 docs each, 100% co-occurrence

Per the prompt protocol: if two columns are byte-identical AND a source confirms
the two labels always co-occur (e.g., arXiv cross-list), the within-pair
assignment is inert (swapping produces identical model outputs up to permutation).
This would not block VERIFIED_LABEL_MAPPING — but the overall mis-assignment
evidence (Section 2) prevents reaching that status regardless.

---

## Section 5: What Would Change Status to VERIFIED_LABEL_MAPPING

1. Complete the Route 4 large join (≥5,000 matched training documents).
2. For each of the 54 columns, confirm ≥99.5% purity of the true arXiv category
   over matched documents.
3. Verify the column→label map is a permutation of the 54 labels (up to the
   two identical pairs).
4. Check per-document label set exact-match agreement rate.
5. Write a corrected `Data/label_mapping.json` with provenance and join stats.

OR: Obtain from the AAPD authors (Yang et al.) the per-document label strings
used to construct the CSV. Email contact is possible but cannot be guaranteed.

---

## Section 6: User Consultation Required

The large Route 4 join is running as a background task (~17 min for 300 docs).
If the results show systematic mis-assignment across all 54 columns (not just
the 5 samples), the only feasible path to VERIFIED_LABEL_MAPPING is the
Kaggle-hosted full arXiv metadata snapshot (LEVEL 2 task).

**Decision required from user (after large join completes):**
Option A: Proceed to Kaggle LEVEL-2 join for full verification.
Option B: Keep AAPD_BASELINE = BLOCKED; proceed with Reuters-only baseline.
Option C: Run an exploratory AAPD-as-is training, labelled EXPLORATORY_UNVERIFIED_LABELS,
           never counted toward the baseline gate, only to understand the data.

---

**FINAL STATUS: UNRESOLVED_LABEL_MAPPING**
**AAPD training: BLOCKED**


---

## FINAL STATUS (Phase 1E — 2026-10-07)

UNRESOLVED_LABEL_MAPPING -- ACCEPTED_BY_CLIENT_DECISION

No verification performed or claimed; investigation out of scope per client instruction.
Dataset used exactly as supplied (Zenodo 10.5281/zenodo.6344750, AAPD.zip).
The arXiv recovery workflow (Route A, Kaggle join) is DISABLED and out of scope.
See docs/phase1/aapd_data_decision.md for full decision record.
