# DR4 Frozen-Checkpoint Label-SA Counterfactual: Final Scientific Analysis

**Diagnostic ID:** `DR4_FROZEN_SA_CF_20261012_A`  
**Execution Date:** 2026-10-12  
**Status:** COMPLETED & VERIFIED ON KAGGLE GPU  
**Classification:** Strictly Inference-Only Counterfactual Probe (Frozen Trained Checkpoints)  
**Parent Screening Run:** `RUN_0_L2_A5B_20261012_A`  
**Parent Decomposition:** `DR4_POSTTRAIN_DECOMPOSITION_20261012`  

---

## 1. Executive Summary and Scientific Disposition

The DR4 Frozen-Checkpoint Label-Self-Attention Counterfactual diagnostic evaluated the direct ranking sensitivity of the trained step-300 models (`L2_STEP300_MODEL_ONLY.pt` and `L2_A5B_STEP300_MODEL_ONLY.pt`) by subtracting the document-invariant static label self-attention offset ($\text{sa\_term}[i] = W_i \cdot \text{SA}[i]$) from the raw logits at inference time across the 200 frozen AAPD Dev screening documents.

### Key Empirical Findings:
1. **Mathematical Invariant Holds:** The cross-document logit standard deviation for each of the 54 labels is strictly invariant under static offset removal (maximum observed difference across all 108 evaluated label vectors: $4.28 \times 10^{-8}$). The V4 passing label count remained identically **21/54** for L2 and **3/54** for L2_A5B.
2. **Top-1 Accuracy Unchanged:** Both variants achieved exactly **63/200** correct Top-1 predictions before and after the counterfactual intervention, exactly tying the dataset frequency prior ($31.50\%$). Net accuracy improvement was **0**.
3. **Severe Score Margin Compression:** Removing the large static SA offsets compressed the score margins between the top labels to negligible numerical tolerances (mean 1st-to-2nd margin in L2 collapsed from $0.0173$ down to $3.42 \times 10^{-6}$).
4. **Duplicate-Target Dominance Preserved:** In L2_A5B, the counterfactual intervention shifted 199/200 Top-1 predictions from `cmp-lg` back to `cond-mat.dis-nn` (the identical ground-truth target column), producing a high rate of rank flips ($199/200$) with zero change in Top-1 correctness.
5. **Scientific Disposition:** **Diagnostic completed; recovery not demonstrated.** Removing static self-attention offsets at inference time does not resolve ranking collapse. The document-dependent component itself ($\text{doc\_term}_{b, i} = \alpha[b, i] \cdot (W_i \cdot h_L[i])$) lacks sufficient dynamic range and discriminative power to overcome static label ordering.

---

## 2. Authenticated Artifacts & SHA-256 Provenance

All deliverables were downloaded directly from Kaggle execution and passed bitwise SHA-256 verification against the signed manifest:

### Evidence Archive Provenance
- **Archive Path:** `D:\LHRM_KAGGLE_ARTIFACTS\RUN_0_L2_A5B_20261012_A\counterfactual\DR4_FROZEN_SA_CF_EVIDENCE.zip`
- **Archive SHA-256:** `a32eab0a33a39bb079259e175b61dd442d33386cb8412f59f69cda93bd92a610`
- **Checksums Manifest:** `D:\LHRM_KAGGLE_ARTIFACTS\RUN_0_L2_A5B_20261012_A\counterfactual\DR4_FROZEN_SA_CF_SHA256SUMS.txt`
- **Executed Notebook Artifact:** `D:\LHRM_KAGGLE_ARTIFACTS\RUN_0_L2_A5B_20261012_A\counterfactual\notebook\lhrm_dr4_frozen_sa_cf_EXECUTED_20261012.ipynb`

### Constituent Files Inside Verified ZIP
| Filename | Verified SHA-256 | Verification Status |
|---|---|---|
| `DR4_FROZEN_SA_CF_RESULTS.json` | `e5cb5326c6615f1ea7c3c95431d4547b3766885e72d504222bcfcd4f59e7e3e2` | MATCH |
| `DR4_FROZEN_SA_CF_PER_LABEL_METRICS.csv` | `1773222c06bfc12c04d66778f45e2690d0ff94ece5ca9a5d0132a1ffe4cc5398` | MATCH (108 rows) |
| `DR4_FROZEN_SA_CF_REPORT.md` | `70fe20047a03b7a0fbdcdc5288c348c3a9f63752ed918fcbbe1a240e5f53f363` | MATCH |

---

## 3. The Counterfactual Intervention & Pre-Intervention Agreement

### 3.1 Mathematical Definition
For document $b \in \{1, \dots, 200\}$ and label $i \in \{1, \dots, 54\}$:
$$\text{original\_logits}[b, i] = \text{doc\_term}[b, i] + \text{sa\_term}[i] + b_{\text{scalar}}$$
$$\text{counterfactual\_logits}[b, i] = \text{doc\_term}[b, i] + b_{\text{scalar}} \equiv \text{original\_logits}[b, i] - \text{sa\_term}[i]$$

Where:
- $\text{doc\_term}[b, i] = \alpha[b, i] \cdot (W_i \cdot h_L[i])$
- $\text{sa\_term}[i] = W_i \cdot \text{SA}[i] = W_i \cdot \left(\operatorname{softmax}\left(\frac{h_L h_L^T}{\sqrt{H}}\right) h_L\right)_i$
- $b_{\text{scalar}}$ is the shared scalar bias ($-0.0233$).

### 3.2 Pre-Intervention Parity Gates
Before computing counterfactual outcomes, the execution passed all non-negotiable verification gates:
1. **Model Forward-Pass Agreement:** Maximum difference between model forward pass logits and algebraic decomposition: $\le 10^{-5}$.
2. **Algebraic Subtraction Parity:** $\max |(\text{original\_logits} - \text{sa\_term}) - (\text{doc\_term} + b_{\text{scalar}})|$:
   - L2: $6.48 \times 10^{-7}$
   - L2_A5B: $3.87 \times 10^{-7}$
3. **Reference Predictions Agreement:** Pre-intervention reconstructed model probabilities matched authentic screening predictions from `val_sample_predictions.json` with maximum difference:
   - L2: $1.04 \times 10^{-7}$
   - L2_A5B: $7.45 \times 10^{-8}$
   - Both variants strictly reproduced the 63/200 Top-1 integer hit baseline.

---

## 4. Comprehensive Numerical Results

| Metric | L2 Original | L2 Counterfactual | L2 Delta | L2_A5B Original | L2_A5B Counterfactual | L2_A5B Delta |
|---|---:|---:|---:|---:|---:|---:|
| **P@1 Hits (out of 200)** | **63/200** | **63/200** | **+0** | **63/200** | **63/200** | **+0** |
| **P@1 Accuracy** | 31.50% | 31.50% | +0.00% | 31.50% | 31.50% | +0.00% |
| **Frequency Prior Reference** | 63/200 | 63/200 | 0 | 63/200 | 63/200 | 0 |
| **Exceeds Frequency Prior?** | False | False | False | False | False | False |
| **Distinct Top-1 Predicted Labels** | 1 | 2 | +1 | 2 | 1 | -1 |
| **Dominant Top-1 Label** | `cond-mat.dis-nn` | `cond-mat.dis-nn` | Unchanged | `cmp-lg` | `cond-mat.dis-nn` | Inverted |
| **Dominant Label Count** | 200/200 (100.0%) | 199/200 (99.5%) | -1 doc | 199/200 (99.5%) | 200/200 (100.0%) | +1 doc |
| **Distinct Ordered Top-3 Patterns** | 1 | 4 | +3 | 2 | 3 | +1 |
| **Total Top-1 Rank Flips** | - | 1/200 (0.5%) | - | - | 199/200 (99.5%) | - |
| **V4 Passing Labels (>0.01 std)** | **21/54** | **21/54** | **0** | **3/54** | **3/54** | **0** |
| **Max Per-Label Std Difference** | - | $4.10 \times 10^{-8}$ | - | - | $4.28 \times 10^{-8}$ | - |
| **Mean Top-1 vs Top-2 Margin** | $0.01726$ | $3.42 \times 10^{-6}$ | $-0.01726$ | $0.00033$ | $0.00033$ | $+0.00000$ |

---

## 5. Detailed Forensic Observations

### 5.1 Invariance of Cross-Document Variance (V4 Gate)
Removing $\text{sa\_term}[i]$ at inference time provided empirical verification of the mathematical identity:
$$\operatorname{Var}_b(\text{counterfactual\_logits}[:, i]) \equiv \operatorname{Var}_b(\text{original\_logits}[:, i]) \quad \forall i$$
The maximum observed difference across all 54 labels was $4.10 \times 10^{-8}$ (L2) and $4.28 \times 10^{-8}$ (L2_A5B), fully accounted for by 32-bit floating-point subtraction rounding. The number of labels passing the V4 threshold remained strictly 21/54 for L2 and 3/54 for L2_A5B. This demonstrates that static offset removal neither creates nor destroys document-dependent variability for any individual label.

### 5.2 Paired Document Accuracy Transitions
To determine whether individual document classifications improved despite the unchanged aggregate score (63/200), we analyzed paired accuracy transitions for all 200 screening documents:

| Transition Category | Description | Variant L2 | Variant L2_A5B |
|---|---|---:|---:|
| **True $\to$ True** | Correct under original, remains correct under counterfactual | 63 | 63 |
| **True $\to$ False** | Correct under original, damaged by counterfactual | 0 | 0 |
| **False $\to$ True** | Incorrect under original, recovered by counterfactual | 0 | 0 |
| **False $\to$ False** | Incorrect under original, remains incorrect | 137 | 137 |

**Critical Observation:** In both variants, not a single document out of the 137 incorrect predictions was recovered by removing the static SA offsets ($\text{False} \to \text{True} = 0$).

### 5.3 Duplicate-Target Collapse (`cmp-lg` vs `cond-mat.dis-nn`)
In AAPD, `cmp-lg` (label index 0) and `cond-mat.dis-nn` (label index 1) possess identical binary target columns across the Train and Dev splits:
- In original L2, `cond-mat.dis-nn` held a static SA advantage ($+0.0757$) over `cmp-lg`, capturing 200/200 Top-1 predictions. Under the counterfactual, `cond-mat.dis-nn` maintained a tiny document-term advantage of $+5.3 \times 10^{-6}$, retaining 199/200 Top-1 slots (with 1 document predicting `cs.se`).
- In original L2_A5B, `cmp-lg` held a static SA advantage of $+0.002807$, capturing 199/200 Top-1 slots. Under the counterfactual, the removal of the SA offset exposed a slight document-term advantage for `cond-mat.dis-nn` (mean probability difference $-0.000333$), flipping 199/200 predictions back to `cond-mat.dis-nn`.
- Because both labels share identical targets, this massive rank flip ($99.5\%$ of documents) produced zero net change in accuracy.

### 5.4 Extreme Margin Compression & Near-Tie Sensitivity
In Variant L2, the mean probability margin between the Top-1 and Top-2 ranked labels collapsed from $0.01726$ to $3.42 \times 10^{-6}$ (with minimum margin $8.94 \times 10^{-8}$).
This demonstrates that:
1. The dynamic range of the document-dependent term across labels is negligible.
2. Apparent diversity in Top-3 patterns (4 distinct patterns) is an artifact of floating-point noise and tie-breaking jitter rather than genuine semantic discrimination.

---

## 6. What This Diagnostic Rules Out vs. What Remains Unresolved

### 6.1 What Is Decisively Ruled Out
- **H1 (Static SA as the Exclusive Cause of Failure) is REFUTED:** While the static label-SA offset is undeniably large ($48\times$ to $52\times$ document terms), removing it at inference time is completely insufficient to restore ranking diversity or accuracy.
- **Hypothesis that Document-Term Accuracies are Masked:** If the document-dependent path $\alpha[b, i] \cdot (W_i \cdot h_L[i])$ had learned accurate document-dependent rankings that were merely submerged beneath the static SA offsets, removing those offsets would have immediately exposed those rankings, producing $\text{False} \to \text{True}$ transitions. The fact that zero misses were recovered proves that the document terms themselves are not providing discriminative multi-label rankings.

### 6.2 Difference Between Inference Subtraction and Training Without SA
This intervention operated exclusively on frozen weights trained with self-attention. It does not establish how parameters would evolve if trained from scratch without SA (which was evaluated in DR3/DR3b and also failed). However, it definitively proves that the failure of the trained checkpoints cannot be repaired by post-hoc calibration of static label offsets.

### 6.3 What Remains Unresolved
Why did joint optimization of document and label representations through scaled dot-product cross-attention fail to produce meaningful document-dependent ranking terms?
The investigation must now audit the structural and optimization properties of the forward pass, loss function, and gradient routing.
