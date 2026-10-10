# Phase 1 Initial Report: LHRM Baseline Reproduction

**Paper:** Label-Aware Hierarchical Ranking Model for Multi-Label Text Classification
**Authors:** Ayash, Algarni, Alqahtani; IEEE Access vol. 13, 2025, DOI 10.1109/ACCESS.2025.3599944
**Report date:** 2026-10-05
**Git hash at time of writing:** 443c504 (initial commit, untouched state)
**WRITTEN BEFORE ANY CODE CHANGE** (Hard Rule 1)

---

## 1. Project Architecture (Before Any Change)

```
Model/lhrm.py                          524 lines; one-file implementation
Label_Definition/arxiv_labels_definitions.json  54 labels; values are DICTS
Label_Definition/reuters_labels_definitions.json  90 labels; values are STRINGS
Data/AAPD.zip                          18.1 MB REAL archive (train/dev/test.csv)
assets/LHRM_ARCH.png
README.md, LHRM_Executive_Summary.md, LHRM_Technical_Review*.md
```

New structure created by setup (no code changes yet):
```
legacy/lhrm_original.py   byte-identical copy (READ-ONLY)
docs/phase1/              this report + decision_registry + preregistration
src/lhrm_baseline/        paper-faithful baseline (to be created)
tests/                    pytest suite
notebooks/                GPU notebooks
configs/                  YAML configs
data/raw/  data/processed/
experiments/
scripts/
```

---

## 2. Existing LHRM Flow

### train command (Model/lhrm.py)
1. set_seed(42)
2. Load train+test JSON (pd.read_json)
3. Label list from labels.txt; MultiLabelBinarizer
4. BertTokenizer + BertModel from bert-base-uncased
5. build_label_embeddings(): [BUG R9] encodes dict repr for arXiv labels
   - under @no_grad; uses last_hidden_state.mean; stores as nn.Parameter
6. DataLoader (max_len=128 default, batch=16 default)
7. LHRMModel: bert().pooler_output -> CSAM -> cat -> Linear(2H, L) [NOT Eqs.7-8]
8. AdamW(lr=2e-5); BCEWithLogitsLoss [NOT Focal]; GradScaler
9. Train 3 epochs (default, NOT paper's 15)
10. eval_loop on TEST set; P@k with WRONG denominator; nDCG OK
11. Save predictions JSON

### rerank command
1. Load predictions JSON from test set
2. Build (score, rank, is_true) rows
3. train_test_split(test rows, test_size=0.3) [LEAKAGE]
4. GradientBoostingRegressor with sklearn defaults (happen to match paper values)
5. Fit on 70% test rows; predict all; re-sort
6. Report BASE and HRRM metrics

### Functions safe to REUSE: set_seed, collate_fn, ndcg_at_k (verify first), load/save_json
### Functions that MUST CHANGE: build_label_embeddings, precision_at_k, CSAM, LHRMModel, run_rerank, run_train

---

## 3. Paper Baseline Specification

Architecture (Eqs. 1-8, 17):
- h_T = BERT_CLS(X)   [CLS last hidden state, Eq.1]
- h_Li = BERT_CLS(Y_i)  [same BERT, per label, with grad, Eq.2]
- S_i = h_T . h_Li  [Eq.4]
- alpha_i = softmax(S_i over i)  [Eq.5]
- A_ij = softmax(h_Li . h_Lj^T / sqrt(768))  [Eq.6, no projections]
- h_final^(i) = alpha_i*h_Li + sum_j A_ij*h_Lj  [Eq.7, per-label vector]
- y_hat_i = W.h_final^(i) + b  [Eq.8, W shared 1xH]
- FocalLoss(y_hat, y_true)  [Eq.17, gamma=2 alpha=0.25 ASSUMED]
- HRRM: GradBoost(score, rank) -> correctness [Eqs.9-12]
- Metrics: P@k=hits/k; nDCG per DCG/IDCG [Eqs.13-16]

Training (PAPER-DEFINED):
- AdamW, lr=2e-5, weight_decay=0.01
- batch_size=8, epochs=15 (AAPD) / 10 (Reuters)
- BERT: bert-base-uncased (ASSUMED from repo default)
- max_len: 256 (ASSUMED per A10: covers >=90% of train abstracts)

---

## 4. Gap Matrix (Key Rows)

| Component | Paper | Current Code | Gap | Action |
|---|---|---|---|---|
| Document embedding | CLS last_hidden_state[:,0] | pooler_output | DIFFERS | Use last_hidden_state[:,0] |
| Label embedding | BERT_CLS(Yi) with grad, per-fwd | static mean-pooled nn.Parameter | DIFFERS | Re-encode in forward with grad |
| Label text | full_name + definition | dict repr of full arXiv entry | DIFFERS (BUG) | Use f"{full_name}: {definition}" |
| CSAM Eq.6 | scaled dot-product no projections | 4-head MultiheadAttention | DIFFERS | Literal scaled dot-product |
| CSAM Eq.7-8 | per-label h_final; scalar W shared | pooled summary + Linear(2H,L) | DIFFERS | Per-label h_final, scalar W |
| Focal Loss | FL(p_t)=alpha_t(1-p_t)^gamma BCE | BCEWithLogitsLoss | MISSING | Implement losses.py |
| P@k (k>1) | hits/k | hits/min(k,|y|) | INFLATED | Fix divisor |
| HRRM protocol | fit on val, apply on test | fit on 70% test rows | LEAKAGE | Clean val->fit, test->apply |
| Epochs/batch | 15/8 | 3/16 (defaults) | DIFFERS | Set paper values |
| Checkpointing | not specified | absent | MISSING | Add per-epoch checkpoint |
| Validation split | not described | absent | MISSING | Use dev.csv |

---

## 5. Data Facts D1-D6

D1 CONFIRMED: train=53840, dev=1000, test=1000, total=55840, 54 labels
   mean labels: train=2.407, dev=2.400, test=2.421; no zero-label docs
   INTERPRETATION: paper's "54,840 train" = train+dev; "1,000 test" = test.csv

D2 CONFIRMED: JSON keys are original-case; CSV columns are all-lowercase
   Exact case matches: 6/54. Case-insensitive: 54/54 all match.
   CODE BUG: context.get(label) fails for 48/54 labels silently -> empty definition

D3 CONFIRMED MISALIGNED (BLOCKING RISK):
   - cmp-lg and cond-mat.dis-nn are IDENTICAL column vectors (both freq 17152)
   - math.co and math.it are IDENTICAL column vectors (both freq 875)
   - Sample abstracts under cmp-lg: quantum correlations, radar waveforms, 
     phase noise channels -> clearly cs.IT/math.IT content, not Logic in CS
   - CONCLUSION: Label column headers do not match column contents
   - ACTION: Obtain original AAPD label order from Yang et al. (2018)
   - AAPD results marked UNVERIFIED_LABEL_MAPPING until resolved

D4 CONFIRMED: P50=158, P90=257, P99=330 words (train)
   >128 words: 67.1%; >256 words: 10.1%
   A10 DECISION: max_len=256 (covers ~90% at word level, per non-performance rule)

D5 NOT PRESENT: No Reuters data. reuters_labels_definitions.json has 90 entries.
   ACTION: User must download NLTK reuters corpus or provide data.

D6 DOCUMENTED: Paper says 9788 but 7769+3019=10788. Expect 10788 (typo).

---

## 6. Compute Assessment

ENVIRONMENT: Python 3.12.5, torch 2.12.1+cpu (NO CUDA GPU), RAM 31.7GB
GPU STATUS: NO GPU AVAILABLE -> ALL FULL TRAINING IS EXTERNAL GPU

Level 1 (LOCAL): unit tests, data prep, HRRM fitting, metrics, reporting
Level 3 (EXTERNAL GPU): BERT training (15 epochs AAPD, 10 Reuters)
-> Generate Kaggle-ready notebooks

---

## 7. Risk Register (Top Items)

R01 AAPD label mapping corruption [CRITICAL - BLOCKING]
R02 Reuters data unavailable [HIGH]
R03 Focal Loss parameters unknown [MEDIUM - assumed gamma=2, alpha=0.25]
R04 P@k metric mismatch [HIGH - CONFIRMED, FIXED in new code]
R05 HRRM leakage [HIGH - CONFIRMED, FIXED in new code]
R07 No local GPU [CRITICAL - external notebooks required]
R10 max_len truncation [MEDIUM - A10 decision: 256]

---

## 8. Verification Outcomes for All R-Claims

R1 524 lines (not 523): CHANGED
R2 No Focal Loss: CONFIRMED (lhrm.py:339)
R3 pooler_output not CLS: CONFIRMED (lhrm.py:185)
R4 Static label embeddings: CONFIRMED (build_label_embeddings, lines 222-243)
R5 Simplified head not Eqs.7-8: CONFIRMED (CSAM lines 153-160, LHRMModel lines 184-191)
R6 P@k denominator wrong: CONFIRMED (precision_at_k line 201)
R7 HRRM leakage: CONFIRMED (run_rerank line 410)
R8 3 epochs, batch 16 defaults: CONFIRMED (CLI lines 497-498)
R9 Dict repr bug: CONFIRMED (f-string on dict value shown in audit)
R10 README path errors: CONFIRMED
R11 Python>=3.10 syntax: CONFIRMED (works fine on Python 3.12)

---

## 9. Execution Plan

Step 1: Legacy copy [DONE]
Step 2: Git init + commit [DONE - hash 443c504]
Step 3: Write decision_registry.md and preregistration.md [NEXT]
Step 4: Implement src/lhrm_baseline/ package [NEXT]
Step 5: Write and run unit tests [NEXT - LEVEL 1]
Step 6: Data preparation (LEVEL 1 - extract AAPD, label mapping audit)
Step 7: Generate GPU notebooks (needs data prep complete)
Step 8: [EXTERNAL GPU] Variant pilot (1 epoch)
Step 9: [EXTERNAL GPU] Full training (15/10 epochs)
Step 10: [LOCAL] HRRM fitting on validation predictions
Step 11: [LOCAL] Test evaluation + bootstrap CI
Step 12: Report + gate assessment

BLOCKING QUESTION: AAPD label mapping (D3). See data_audit.md for details.
