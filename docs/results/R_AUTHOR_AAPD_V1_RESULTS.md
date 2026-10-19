# LHRM R_AUTHOR V1.0 ? Scientific Results & Evidence Report

## 1. Executive Summary & Artifact Provenance

This document provides the authoritative, audited scientific and empirical results for the **LHRM R_AUTHOR V1.0** model release on the **AAPD (ArXiv Academic Paper Dataset)** benchmark.

- **Model Architecture**: Author-style Context-Self-Attention-Module (`R_AUTHOR`). BERT-base-uncased document encoder (768d), static precomputed BERT label embeddings (54 labels ? 768d), cross-attention label-aware document representation with residual addition and LayerNorm, CSAM fusion block, and linear classification head.
- **Loss Function & Training Protocol**: Standard Binary Cross-Entropy (`BCEWithLogitsLoss`), AdamW optimizer, linear warmup with decay, batch size 8, gradient accumulation 1, 15 complete epochs (100,950 total optimizer steps) on Kaggle NVIDIA T4 GPU.
- **Run ID**: `AAPD_R_AUTHOR_BCE_RECOVERY_20261015_A`
- **Selected Checkpoint**: `best_dev_model_only.pt` (Epoch 6, Step 40,380)
- **Checkpoint SHA-256**: `988bb64fca2b8ea21030585de7268a1cfb1fc8676b91b4d6e6f38c4f22c95643`
- **Label Order SHA-256**: `a3d662f3920259a94ed61d83f98f21cabe7f908d98a5662ccd9adc2413cc000b` (exact canonical 54-label schema)

---

## 2. Dataset Partitioning & Training Dynamics

### 2.1 Dataset Splits
- **Training Set**: 53,840 documents (full AAPD training partition)
- **Validation (Dev) Set**: 1,000 documents (split into 400 model selection, 200 HRRM fit, 200 HRRM eval, 200 diagnostic)
- **Frozen Test Set**: 1,000 documents (evaluated once, strictly neural-only, zero test-time tuning)

### 2.2 15-Epoch Training History (`epoch_metrics.csv`)

| Epoch | Train Loss | Global Steps | Val P@1 | Val P@3 | Val P@5 | Val nDCG@3 | Val nDCG@5 | Val Micro-F1 | Selection P@1 | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.089996 | 6730 | 0.8460 | 0.6030 | 0.4100 | 0.8032 | 0.8421 | 0.6962 | 0.8650 | Completed |
| 2 | 0.062288 | 13460 | 0.8540 | 0.6130 | 0.4142 | 0.8133 | 0.8508 | 0.7264 | 0.8500 | Completed |
| 3 | 0.052821 | 20190 | 0.8700 | 0.6207 | 0.4160 | 0.8257 | 0.8585 | 0.7427 | 0.8750 | Completed |
| 4 | 0.044969 | 26920 | 0.8650 | 0.6213 | 0.4208 | 0.8243 | 0.8618 | 0.7474 | 0.8650 | Completed |
| 5 | 0.038326 | 33650 | 0.8750 | 0.6267 | 0.4216 | 0.8318 | 0.8662 | 0.7493 | 0.8575 | Completed |
| 6 | 0.032777 | 40380 | 0.8760 | 0.6223 | 0.4176 | 0.8282 | 0.8613 | 0.7432 | 0.8950 | **Selected Best** |
| 7 | 0.028080 | 47110 | 0.8560 | 0.6117 | 0.4148 | 0.8131 | 0.8512 | 0.7359 | 0.8575 | Completed |
| 8 | 0.023954 | 53840 | 0.8660 | 0.6140 | 0.4136 | 0.8173 | 0.8524 | 0.7401 | 0.8725 | Completed |
| 9 | 0.020679 | 60570 | 0.8570 | 0.6180 | 0.4136 | 0.8189 | 0.8519 | 0.7395 | 0.8625 | Completed |
| 10 | 0.017831 | 67300 | 0.8680 | 0.6137 | 0.4156 | 0.8177 | 0.8555 | 0.7413 | 0.8750 | Completed |
| 11 | 0.015382 | 74030 | 0.8540 | 0.6197 | 0.4144 | 0.8192 | 0.8516 | 0.7441 | 0.8625 | Completed |
| 12 | 0.013552 | 80760 | 0.8540 | 0.6137 | 0.4124 | 0.8152 | 0.8493 | 0.7451 | 0.8675 | Completed |
| 13 | 0.011907 | 87490 | 0.8560 | 0.6130 | 0.4124 | 0.8138 | 0.8491 | 0.7336 | 0.8775 | Completed |
| 14 | 0.010475 | 94220 | 0.8620 | 0.6173 | 0.4144 | 0.8203 | 0.8545 | 0.7380 | 0.8650 | Completed |
| 15 | 0.009489 | 100950 | 0.8610 | 0.6240 | 0.4152 | 0.8257 | 0.8555 | 0.7503 | 0.8750 | Completed |

> [!NOTE]
> Checkpoint selection was governed strictly by **Selection P@1** on the 400-document selection subset of Dev. Epoch 6 achieved the peak Selection P@1 of **0.8950** (Step 40,380) with Full Dev P@1 of **0.8760**. Subsequent epochs experienced training loss convergence (down to 0.009489 at Epoch 15) accompanied by slight overfit on Dev ranking metrics.

---

## 3. Independent Frozen Test Set Evaluation (`test_summary_metrics.json`)

The independent AAPD Test set (1,000 documents) was evaluated **once** using the frozen Epoch 6 checkpoint (`best_dev_model_only.pt`). The evaluation protocol was strictly **Neural-Only** (no tree re-ranking, no test-time threshold sweep, threshold fixed at default 0.5).

### 3.1 Overall Performance Summary

| Metric Group | Metric | Value | Reference / Notes |
|:---|:---|:---:|:---|
| **Ranking Metrics** | **P@1** | **0.8490** (84.90%) | Primary precision metric |
| | **P@3** | **0.6160** (61.60%) | Top-3 ranking precision |
| | **P@5** | **0.4144** (41.44%) | Top-5 ranking precision |
| | **nDCG@3** | **0.8104** | Normalized Discounted Cumulative Gain @ 3 |
| | **nDCG@5** | **0.8429** | Normalized Discounted Cumulative Gain @ 5 |
| **Classification (t=0.5)** | **Micro-F1** | **0.727787** (72.78%) | Exact mathematical recalculation: 2?TP / (2?TP + FP + FN) |
| | Micro-Precision | 0.769691 (76.97%) | TP / (TP + FP) |
| | Micro-Recall | 0.690211 (69.02%) | TP / (TP + FN) |
| | **Macro-F1** | **0.555253** (55.53%) | Unweighted mean across 54 labels |
| | Macro-Precision | 0.631180 (63.12%) | Unweighted mean across 54 labels |
| | Macro-Recall | 0.525600 (52.56%) | Unweighted mean across 54 labels |
| | Weighted-F1 | 0.712893 (71.29%) | Support-weighted mean across 54 labels |
| | Weighted-Precision | 0.761388 (76.14%) | Support-weighted mean across 54 labels |
| | Weighted-Recall | 0.690211 (69.02%) | Support-weighted mean across 54 labels |
| **Global Prediction Counts** | True Positives (TP) | **1671** | Correct label assignments |
| | False Positives (FP) | **500** | Over-predicted label assignments |
| | False Negatives (FN) | **750** | Missed ground-truth labels |
| | Total Ground Truth | **2421** | Sum of ground-truth label occurrences |

### 3.2 Exact Mathematical Verification
$$\text{Micro-Precision} = \frac{1671}{1671 + 500} = \frac{1671}{2171} = 0.7696913864578535$$
$$\text{Micro-Recall} = \frac{1671}{1671 + 750} = \frac{1671}{2421} = 0.6902106567534076$$
$$\text{Micro-F1} = \frac{2 \cdot 1671}{2 \cdot 1671 + 500 + 750} = \frac{3342}{4592} = 0.7277874564459931$$

---

## 4. HRRM Held-Out Dev Empirical Negative Result

The Hierarchical Recurrent Ranking Module (HRRM) / GBDT re-ranker was fitted on a non-leaky 200-document slice of Dev and evaluated on a separate 200-document held-out slice of Dev (`docs/results/hrrm_heldout_comparison.json`).

> [!WARNING]
> **Empirical Negative Finding**: HRRM produced a slight degradation or zero gain across all ranking and classification metrics on the held-out Dev partition:
> - $\Delta P@1$: **-0.005** (0.870 $\rightarrow$ 0.865)
> - $\Delta P@3$: **0.000** (0.615 $\rightarrow$ 0.615)
> - $\Delta P@5$: **-0.006** (0.413 $\rightarrow$ 0.407)
> - $\Delta \text{Micro-F1}$: **-0.000949** (0.7328 $\rightarrow$ 0.7319)
> - $\Delta \text{Macro-F1}$: **-0.030388** (0.5187 $\rightarrow$ 0.4883)
>
> In accordance with fail-closed scientific integrity:
> 1. HRRM was **NOT deployed to the frozen Test set**.
> 2. No HRRM Test evaluation was conducted, preventing leakage or speculative evaluation.
> 3. Production client inference utilizes Neural-Only inference.

---

## 5. Complete 54-Label Breakdown on Frozen Test Set

The canonical 54-label AAPD schema is preserved in exact index order (SHA-256: `a3d662f3920259a94ed61d83f98f21cabe7f908d98a5662ccd9adc2413cc000b`).

| Index | Label Name | Precision | Recall | F1-Score | Support | TP | FP | FN |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | `cmp-lg` | 0.9387 | 0.9080 | 0.9231 | 337 | 306 | 20 | 31 |
| 1 | `cond-mat.dis-nn` | 0.9387 | 0.9080 | 0.9231 | 337 | 306 | 20 | 31 |
| 2 | `cond-mat.stat-mech` | 0.7419 | 0.7877 | 0.7641 | 146 | 115 | 40 | 31 |
| 3 | `cs.ai` | 0.7538 | 0.4414 | 0.5568 | 111 | 49 | 16 | 62 |
| 4 | `cs.cc` | 0.5840 | 0.7766 | 0.6667 | 94 | 73 | 52 | 21 |
| 5 | `cs.ce` | 0.8125 | 0.5000 | 0.6190 | 78 | 39 | 9 | 39 |
| 6 | `cs.cg` | 0.7625 | 0.8592 | 0.8079 | 71 | 61 | 19 | 10 |
| 7 | `cs.cl` | 0.7143 | 0.5147 | 0.5983 | 68 | 35 | 14 | 33 |
| 8 | `cs.cr` | 0.8033 | 0.8033 | 0.8033 | 61 | 49 | 12 | 12 |
| 9 | `cs.cv` | 0.7324 | 0.8125 | 0.7704 | 64 | 52 | 19 | 12 |
| 10 | `cs.cy` | 0.6279 | 0.4500 | 0.5243 | 60 | 27 | 16 | 33 |
| 11 | `cs.db` | 0.5532 | 0.5909 | 0.5714 | 44 | 26 | 21 | 18 |
| 12 | `cs.dc` | 0.7368 | 0.5283 | 0.6154 | 53 | 28 | 10 | 25 |
| 13 | `cs.dl` | 0.6341 | 0.4815 | 0.5474 | 54 | 26 | 15 | 28 |
| 14 | `cs.dm` | 0.7179 | 0.6512 | 0.6829 | 43 | 28 | 11 | 15 |
| 15 | `cs.ds` | 0.9348 | 0.8431 | 0.8866 | 51 | 43 | 3 | 8 |
| 16 | `cs.fl` | 0.8182 | 0.9184 | 0.8654 | 49 | 45 | 10 | 4 |
| 17 | `cs.gt` | 0.7059 | 0.5455 | 0.6154 | 44 | 24 | 10 | 20 |
| 18 | `cs.hc` | 0.7500 | 0.4500 | 0.5625 | 40 | 18 | 6 | 22 |
| 19 | `cs.ir` | 0.7333 | 0.7097 | 0.7213 | 31 | 22 | 8 | 9 |
| 20 | `cs.it` | 0.6250 | 0.5000 | 0.5556 | 40 | 20 | 12 | 20 |
| 21 | `cs.lg` | 0.8333 | 0.9722 | 0.8974 | 36 | 35 | 7 | 1 |
| 22 | `cs.lo` | 0.6667 | 0.8276 | 0.7385 | 29 | 24 | 12 | 5 |
| 23 | `cs.ma` | 0.6923 | 0.6923 | 0.6923 | 39 | 27 | 12 | 12 |
| 24 | `cs.mm` | 0.8077 | 0.7778 | 0.7925 | 27 | 21 | 5 | 6 |
| 25 | `cs.ms` | 0.5625 | 0.9000 | 0.6923 | 20 | 18 | 14 | 2 |
| 26 | `cs.na` | 0.5556 | 0.2500 | 0.3448 | 20 | 5 | 4 | 15 |
| 27 | `cs.ne` | 0.8000 | 0.5000 | 0.6154 | 24 | 12 | 3 | 12 |
| 28 | `cs.ni` | 0.9091 | 0.4545 | 0.6061 | 22 | 10 | 1 | 12 |
| 29 | `cs.pf` | 0.6667 | 0.5000 | 0.5714 | 20 | 10 | 5 | 10 |
| 30 | `cs.pl` | 0.7333 | 0.6875 | 0.7097 | 16 | 11 | 4 | 5 |
| 31 | `cs.ro` | 0.6154 | 0.4000 | 0.4848 | 20 | 8 | 5 | 12 |
| 32 | `cs.sc` | 0.5185 | 0.5833 | 0.5490 | 24 | 14 | 13 | 10 |
| 33 | `cs.se` | 0.4000 | 0.2857 | 0.3333 | 14 | 4 | 6 | 10 |
| 34 | `cs.si` | 0.7857 | 0.6875 | 0.7333 | 16 | 11 | 3 | 5 |
| 35 | `cs.sy` | 0.5000 | 0.1667 | 0.2500 | 18 | 3 | 3 | 15 |
| 36 | `math.co` | 0.4545 | 0.2500 | 0.3226 | 20 | 5 | 6 | 15 |
| 37 | `math.it` | 0.5000 | 0.3000 | 0.3750 | 20 | 6 | 6 | 14 |
| 38 | `math.lo` | 0.8889 | 0.6154 | 0.7273 | 13 | 8 | 1 | 5 |
| 39 | `math.na` | 0.7857 | 0.7333 | 0.7586 | 15 | 11 | 3 | 4 |
| 40 | `math.nt` | 0.6000 | 0.4000 | 0.4800 | 15 | 6 | 4 | 9 |
| 41 | `math.oc` | 0.1111 | 0.0714 | 0.0870 | 14 | 1 | 8 | 13 |
| 42 | `math.pr` | 0.7000 | 0.4667 | 0.5600 | 15 | 7 | 3 | 8 |
| 43 | `math.st` | 0.2941 | 0.5556 | 0.3846 | 9 | 5 | 12 | 4 |
| 44 | `nlin.ao` | 1.0000 | 0.1429 | 0.2500 | 14 | 2 | 0 | 12 |
| 45 | `physics.data-an` | 0.6667 | 0.2857 | 0.4000 | 7 | 2 | 1 | 5 |
| 46 | `physics.soc-ph` | 0.0000 | 0.0000 | 0.0000 | 3 | 0 | 0 | 3 |
| 47 | `q-bio.nc` | 0.2500 | 0.0909 | 0.1333 | 11 | 1 | 3 | 10 |
| 48 | `q-bio.qm` | 0.5000 | 0.2222 | 0.3077 | 9 | 2 | 2 | 7 |
| 49 | `quant-ph` | 0.6667 | 0.7500 | 0.7059 | 8 | 6 | 3 | 2 |
| 50 | `stat.ap` | 0.0000 | 0.0000 | 0.0000 | 9 | 0 | 1 | 9 |
| 51 | `stat.me` | 0.0000 | 0.0000 | 0.0000 | 8 | 0 | 3 | 8 |
| 52 | `stat.ml` | 0.5000 | 0.3333 | 0.4000 | 6 | 2 | 2 | 4 |
| 53 | `stat.th` | 0.5000 | 0.5000 | 0.5000 | 4 | 2 | 2 | 2 |

---

## 6. Client Report Artifacts (Archival Reference)

The following formatted client reports were produced during the evaluation phase and are preserved in the authoritative local archive (`D:\LHRM_KAGGLE_ARTIFACTS\AAPD_R_AUTHOR_BCE_RECOVERY_20261015_A\reports`):

1. **Final Test Arabic Report (PDF)**:
   - File: `reports/test/LHRM_AAPD_FINAL_TEST_54_LABELS_CLIENT_REPORT_AR.pdf`
   - Size: 61,838 bytes
   - SHA-256: `3462e5411d5ca6e6d4c21ca8b3ed6fd7caec7db714c1b75d22145b999b715eda`
2. **Final Test Excel Results (XLSX)**:
   - File: `reports/test/LHRM_AAPD_FINAL_TEST_54_LABELS_CLIENT_RESULTS.xlsx`
   - Size: 20,239 bytes
   - SHA-256: `643d6e009d18c7c7d9bc3dd7f8810e2311f12d7b334e00670973b4a60bf4d981`
3. **Dev 54-Label Arabic Report (PDF)**:
   - File: `reports/LHRM_AAPD_Per_Label_54_Client_Report_AR.pdf`
   - Size: 89,580 bytes
   - SHA-256: `3f63e3fb3eb170b0f97335ccc5659be02b13955824ef83c995dea9c50cfec917`
4. **Dev 54-Label Excel Results (XLSX)**:
   - File: `reports/LHRM_AAPD_Per_Label_54_Client_Results.xlsx`
   - Size: 15,464 bytes
   - SHA-256: `9c17125b7da6acb108fd4f5ba074f66d96bdea13ece74078d029a6a0190d3674`
5. **Frozen Test Evidence Archive (ZIP)**:
   - File: `reports/test/LHRM_R_AUTHOR_EPOCH6_TEST_EVIDENCE.zip`
   - Size: 585,138 bytes
   - SHA-256: `58c5ab09e1e3d80ecf53ccc10421bd40fea7f8e4774e4010390255799066db73`

> [!NOTE]
> Large binary documents (`.pdf`, `.xlsx`, `.zip`) are preserved off-Git in the local archive. The machine-readable JSON and CSV files in `docs/results/` serve as the public, authoritative versioned sources in Git.

---

## 7. Scientific Caveats & Limitations

1. **Baseline Comparisons**: R_AUTHOR achieves high ranking accuracy ($P@1 = 84.90\%$, Micro-$F_1 = 72.78\%$), but this project does **NOT** claim to have outperformed a compute-matched BERT+BCE baseline under identical hyperparameter conditions.
2. **Label Semantics**: The raw AAPD label names (e.g., `cs.LG`, `cond-mat.stat-mech`) represent ArXiv subject category codes. They have not been mapped to verified human-edited taxonomic definitions.
3. **Score Calibration**: Model outputs undergo raw sigmoid activation. Predicted probabilities are uncalibrated ranking scores; the fixed threshold of 0.5 was not tuned on the test set.
4. **Single Seed**: All results represent a single training execution (seed 42). Multi-seed variance bounds remain an open research item.
