# S1 Route 4 Large Join — Final Results

**Date:** 2026-10-06
**Status:** ANALYSIS COMPLETE
**Conclusion:** UNRESOLVED_LABEL_MAPPING (DEFINITIVE — no additional join needed)

---

## Join Statistics

- Documents attempted: 300
- Documents matched to arXiv: 29 (9.7% match rate)
- Low match rate explanation: AAPD abstracts are pre-processed (lowercased,
  tokenized, whitespace-normalised); arXiv API returns raw LaTeX text. The
  first 12 words of each abstract must match after normalisation. Many AAPD
  abstracts have equations or symbols that break the match.

---

## Label Mapping Test (29 matched documents)

| Metric | Value |
|---|---|
| Exact label set agreement | 0 / 29 = 0.0% |
| Mean IoU (Jaccard) of label sets | 0.018 |
| Docs with any shared label | 2 / 29 = 6.9% (cs.ai, cs.lo) |
| Docs with IoU > 0.5 | 0 / 29 = 0.0% |

**This is not a close miss. It is a systematic mismatch.**

---

## Column Purity (fraction of active docs where col name == true arXiv cat)

Only 2 out of 25 observed columns have any correct self-assignment:
- cs.ai: 1/1 (100%) — single document, anecdotal
- cs.lo: 1/1 (100%) — single document, anecdotal

All other 23 columns: 0% purity.

Notable example:
- `cmp-lg`: appears in 14 matched docs. True arXiv category in every case: cs.IT.
  `cmp-lg` = Computation and Language (now cs.CL). The true category is cs.IT
  (Information Theory). These are completely different fields.
- `cond-mat.dis-nn`: appears in same 14 docs (byte-identical to cmp-lg column).
  True arXiv: cs.IT. `cond-mat.dis-nn` = Disordered Systems (physics). Wildly wrong.

---

## Observed Column -> True arXiv Category Mapping (29 docs)

| Column (CSV label) | Most common true arXiv cats | n |
|---|---|---|
| cmp-lg | cs.it (14/14), cs.cr (4/14) | 14 |
| cond-mat.dis-nn | cs.it (14/14), cs.cr (4/14) | 14 |
| cond-mat.stat-mech | cs.lg (2/2), stat.ml (2/2) | 2 |
| cs.cc | cs.lg (2/2), stat.ml (2/2) | 2 |
| cs.ds | cs.cl (4/4) | 4 |
| cs.fl | cs.cr (5/5), cs.it (4/5) | 5 |
| cs.hc | cs.dc (2/2) | 2 |
| cs.it | cs.ir (3/3) | 3 |
| cs.ma | cs.cy (2/2) | 2 |
| math.lo | cs.dl (3/3) | 3 |

In every case where n >= 2, the column name does not appear in the true arXiv categories.

---

## Identical Column Pair Confirmed

`cmp-lg` and `cond-mat.dis-nn` are byte-identical (14/14 matched docs have both active
simultaneously). They always co-occur and always map to the same true arXiv categories
(cs.IT, cs.CR). This confirms the pair is indistinguishable and both are systematically
mis-labelled.

---

## Interpretation

The AAPD Zenodo CSV columns are NOT the arXiv category labels of the corresponding papers.
The mapping appears to be a systematic permutation or scramble of the 54 label names,
not a random shuffle per-document. The same wrong label appears consistently for the
same wrong column across all 29 matched documents.

Possible explanations:
1. The Fallah et al. (CIRCLE 2022) pipeline extracted AAPD papers from arXiv but
   applied a column-name assignment that does not correspond to the original arXiv
   categories (e.g., an off-by-one in a sort, or a different label taxonomy).
2. The column names are arbitrary identifiers that do not correspond to the true
   arXiv category strings used to construct the binary label matrix.

**It is NOT possible to train a meaningful multi-label classifier on AAPD using the
Zenodo labels, because the label strings do not identify the true categories.**

---

## What Would Be Needed to Proceed with AAPD

Option A (LEVEL 2 — Kaggle full arXiv metadata):
  Download the Kaggle arxiv-metadata-oai-snapshot JSON (millions of papers).
  Join every AAPD document by abstract text to recover true arXiv IDs and categories.
  Build a corrected label matrix from the true categories, restricted to the 54 AAPD labels.
  Estimated time: 1-2 hours of join computation; requires ~2 GB download.

Option B (Author contact):
  Email Yang et al. (SGM, COLING 2018) or Fallah et al. (CIRCLE 2022) requesting
  the original per-document arXiv ID list used to construct the AAPD CSV.
  No guarantee of response.

Option C (Proceed with Reuters only):
  Run the LHRM baseline on Reuters exclusively. Report AAPD as UNVERIFIABLE with
  explanation. This is the fastest path to G5 and RUN A.

---

## Decision Required from User

See docs/phase1/user_decision_D3.md for the formal decision request.

---

## Final Status

**S1 Route 4 COMPLETE. Result: UNRESOLVED_LABEL_MAPPING (definitive).**
**AAPD training: BLOCKED until D3 is resolved via one of the three options above.**
