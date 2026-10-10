# Data Audit Report

**Date:** 2026-10-05
**Tool:** Python script (scratch/audit_data.py)
**Status:** COMPLETE — verified 2026-10-05

---

## D1: AAPD Shapes — VERIFIED ✅

| Split | Rows | Label Cols | Mean Labels | Min | Max | Zero-Label |
|---|---|---|---|---|---|---|
| train | 53,840 | 54 | 2.407 | 2 | 8 | 0 |
| dev | 1,000 | 54 | 2.400 | 2 | 7 | 0 |
| test | 1,000 | 54 | 2.421 | 2 | 7 | 0 |
| **TOTAL** | **55,840** | | | | | |

**Interpretation:**
- Paper says "55,840 total, 54,840 train / 1,000 test"
- Reconciliation: train (53,840) + dev (1,000) = 54,840 → paper's "train"
- Our protocol: train=train.csv, val=dev.csv, test=test.csv
- Minimum labels = 2: zero-label edge cases do not exist (A16 moot for AAPD)

## D2: Case Sensitivity — VERIFIED ✅ (with consequence)

- JSON keys: original-case (e.g., `cs.AI`, `cs.CL`)
- CSV headers: all-lowercase (e.g., `cs.ai`, `cs.cl`)
- Exact case-sensitive matches: **6/54 only**
- Case-insensitive matches: **54/54** (all found)

**Bug in original code (R9):** `context.get(label, '')` uses lowercase label as key
on original-case JSON → returns empty string for 48/54 labels.
**Fix:** Case-insensitive lookup using `{k.lower(): k for k in defs.keys()}`.

## D3: Label Column Misalignment — CONFIRMED ⚠️ BLOCKING

### Evidence
| Anomaly | Value |
|---|---|
| cmp-lg frequency | 17,152 (31.9%) |
| cond-mat.dis-nn frequency | 17,152 (31.9%) |
| Are they identical columns? | **YES — byte-identical** |
| math.co frequency | 875 (1.6%) |
| math.it frequency | 875 (1.6%) |
| Are they identical? | **YES — byte-identical** |

### Sample abstracts under `cmp-lg` column (random selection)
1. "...quantum and classical correlations in three qubits..."
2. "...frequency diverse radar waveforms, frequency diverse array..."
3. "...discrete time Wiener phase noise channel with integrate and dump..."

**Assessment:** These are clearly **information theory / physics** abstracts, not
"Logic in Computer Science" (the label's definition in the JSON).

### Real AAPD dataset (Yang et al. 2018, SGM paper)
The public AAPD dataset from "SGM: Sequence Generation Model for Multi-label
Classification" (Yang et al., EMNLP 2018) uses arXiv category labels where
`cs.IT` (Information Theory, shared with math.IT) is consistently one of the
most frequent classes (~30% of documents). This aligns with the observed
17,152-count column.

### Hypothesis
The two most frequent CSV columns (`cmp-lg`, `cond-mat.dis-nn`) are actually
`cs.IT` and `math.IT`, which always co-occur in arXiv practice.
The label **header names** may not correspond to the actual **column contents**.

### Impact
- LHRM's CSAM module uses label definition text to condition attention
- If `cmp-lg` column contains `cs.IT` abstracts but is conditioned on the
  "Logic in Computer Science" definition, the semantic alignment is broken
- P@k and nDCG can still be computed but the CSAM semantics are poisoned

### Action Required
**BLOCKING for full LHRM semantic validation.**
To resolve:
1. Obtain the original AAPD dataset from Yang et al. 2018 (via EMNLP 2018 ACL Anthology)
2. Cross-reference column order with the 54 arXiv labels listed in the SGM paper
3. If correct mapping confirmed: create `data/label_mapping.json` and rerun prepare_aapd.py

The model can be trained and metrics computed, but results will be marked
`UNVERIFIED_LABEL_MAPPING` until this is resolved.

## D4: Token Length Statistics — VERIFIED ✅

Word-level statistics (training set, N=53,840):
| Percentile | Word count |
|---|---|
| P50 | 158 |
| P75 | 208 |
| P90 | 257 |
| P95 | 286 |
| P99 | 330 |
| P100 | 504 |

- **67.1%** of training abstracts exceed 128 words
- **10.1%** exceed 256 words

**A10 Decision (max_len=256):** P90 word count = 257 > 128, so 128 is insufficient
by the non-performance rule. 256 is chosen (covers ~90% at word level).
Actual truncation at 256 tokens (BERT subword) will be somewhat higher due
to subword splitting, especially for mathematical notation.

## D5: Reuters Data — NOT PRESENT ⚠️

- `Label_Definition/reuters_labels_definitions.json`: 90 entries (plain strings)
- No Reuters corpus in repository
- Reuters-21578 is available via NLTK: `nltk.download('reuters')`
  - Or: https://kdd.ics.uci.edu/databases/reuters21578/reuters21578.html
- Paper split: 7,769 train / 3,019 test (note: sum=10,788, paper says 9,788 — likely typo)
- ModApte split (standard): 7,769 train / 3,019 test

**Action:** User must run `python -c "import nltk; nltk.download('reuters')"` or
provide the Reuters dataset in `data/raw/Reuters/`.

## D6: Reuters Count Inconsistency — DOCUMENTED

Paper text: "9,788 documents" but Table 3: 7,769 + 3,019 = **10,788**
Assessment: Typo for 10,788. Standard ModApte split is 10,788 docs.
Verification: Count when data is obtained.
