# User Decision Required: D3 — AAPD Label Mapping

**Date:** 2026-10-06
**Urgency:** Blocks AAPD training only. Reuters baseline is unblocked.

---

## Summary of Finding

The AAPD Zenodo CSV label columns are SYSTEMATICALLY INCORRECT.

Evidence from arXiv API join (29 matched documents):
- Exact label agreement: 0/29 (0.0%)
- Mean IoU: 0.018
- Column `cmp-lg` always maps to true arXiv `cs.IT` (Information Theory) — wrong field
- Column `cond-mat.dis-nn` always maps to true arXiv `cs.IT` — also wrong
- Column `cs.it` always maps to true arXiv `cs.IR` (Information Retrieval) — wrong label
- No column with n>=2 matches its own name

The labels in the CSV cannot be used to train a meaningful model.
This is not a naming-convention issue. The labels are structurally wrong.

---

## Your Options

**Option A: Kaggle arXiv Metadata Join (LEVEL 2 task)**
- Download: https://www.kaggle.com/datasets/Cornell-University/arxiv
- ~2 GB compressed JSON; 2.3M papers
- Join AAPD abstracts to recover true arXiv IDs and categories
- Build a corrected label matrix restricted to the 54 AAPD label strings
- Estimated compute: 1-4 hours; ~16 GB RAM needed for full join
- Result: Corrected AAPD dataset with verified labels
- Risk: If abstract text in AAPD is too different from Kaggle snapshot, match rate may be low

**Option B: Author Contact**
- Email: Yue Yang (yang.yue@pku.edu.cn, SGM author) or Haytame Fallah (CIRCLE 2022)
- Request: "Please share the per-document arXiv ID list used to construct the AAPD.zip CSV"
- Timeline: Unknown (days to never)
- Risk: No guarantee of response

**Option C: Reuters Only (Proceed Now)**
- Run the full LHRM baseline on Reuters only
- Report AAPD as NOT REPRODUCIBLE (label provenance unverifiable)
- This unblocks G5 immediately and gives us Reuters results in ~1-2 hours on GPU
- AAPD can be revisited later as a LEVEL 2 task

---

## Recommendation

**Option C first, then Option A in parallel or after.**

Reuters-only baseline is a valid reproducibility result — the paper reports both
datasets separately and Reuters is fully verified. Running Option C does not preclude
running Option A later. Option A can be started as a background task while RUN A runs.

---

## Status if No Decision is Made

Reuters baseline (RUN A) proceeds regardless.
AAPD remains BLOCKED. No AAPD metrics will be reported.
