# DR4 Frozen-Checkpoint Label-SA Counterfactual Protocol

**Diagnostic ID:** `DR4_FROZEN_SA_CF_20261012_A`  
**Date:** 2026-10-12  
**Status:** PREREGISTERED & PREPARED (EXECUTION HALTED PENDING USER AUTHORIZATION)  
**Classification:** Inference-Only Frozen-Checkpoint Diagnostic (Strictly Zero Optimizer Steps)  
**Parent Screening Run:** `RUN_0_L2_A5B_20261012_A`  
**Parent Decomposition:** `DR4_POSTTRAIN_DECOMPOSITION_20261012`  

---

## 1. Scientific Motivation and Hypothesis

In DR4 (`RUN_0_L2_A5B_20261012_A`), both `L2` (shared scalar classifier) and `L2_A5B` (unshared per-label classifier $W \in \mathbb{R}^{k \times H}$) failed the 300-step viability screen (V4: 21/54 for L2, 3/54 for L2_A5B; V5: 63/200 hits, matching the frequency prior).

Subsequent post-training logit decomposition (`DR4_POSTTRAIN_DECOMPOSITION_20261012`) established that for both trained models:
1. The static, document-invariant label self-attention term $\text{sa\_term}[i] = W_i \cdot \text{SA}[i]$ dominates the document-dependent term $\text{doc\_term}[b, i] = \alpha[b, i] \cdot (W_i \cdot h_L[i])$ by $48.03\times$ (L2) and $51.82\times$ (L2_A5B).
2. The cross-document variance of $\text{sa\_term}[i]$ is strictly $0.000000$ for every label $i$.
3. The cross-label standard deviation of $\text{sa\_term}[i]$ ($0.273$ to $0.278$) is an order of magnitude larger than the cross-document variation of $\text{doc\_term}[b, i]$ ($0.008$ to $0.010$).

### 1.1 Scientific Question
Does removing the document-invariant label-SA component from the raw logits of the frozen, trained models materially restore ranking diversity and improve Top-1 accuracy beyond the 63/200 frequency prior on the authentic 200 Dev documents?

### 1.2 Scope and Boundaries
- This intervention is **strictly inference-only**.
- It tests the immediate sensitivity of the frozen model's decision boundary to the additive static SA offset.
- It **does NOT** simulate or predict what weights would be learned if the model were trained from step 0 without self-attention (that was separately screened in DR3 / DR3b).
- It involves **zero parameter updates, zero optimizer steps, zero backpropagation, and zero modification to checkpoint tensors**.

---

## 2. Mathematical Definition of Counterfactual

For any document $b \in \{1, \dots, 200\}$ and label $i \in \{1, \dots, 54\}$:

### 2.1 Original Algebraic Logits
$$\text{original\_logits}[b, i] = \text{doc\_term}[b, i] + \text{sa\_term}[i] + b_{\text{scalar}}$$

Where:
- $\text{doc\_term}[b, i] = \alpha[b, i] \cdot (W_i \cdot h_L[i])$, with $\alpha[b, :] = \operatorname{softmax}\left(\frac{h_T[b] h_L^T}{\sqrt{H}}\right)$
- $\text{sa\_term}[i] = W_i \cdot \left(\operatorname{softmax}\left(\frac{h_L h_L^T}{\sqrt{H}}\right) h_L\right)_i$
- $b_{\text{scalar}}$ is the learned shared scalar bias.

### 2.2 Counterfactual Logits
$$\text{counterfactual\_logits}[b, i] = \text{doc\_term}[b, i] + b_{\text{scalar}} \equiv \text{original\_logits}[b, i] - \text{sa\_term}[i]$$

### 2.3 Non-Negotiable Structural Invariants
1. **Zero Parameter Mutation:** Checkpoint state dictionaries are loaded read-only and never modified.
2. **Identical Intermediate Embeddings:** $h_T[b]$ and $h_L[i]$ are extracted using the authentic trained BERT weights without modification.
3. **Cross-Document Variance Invariance:** Because $\text{sa\_term}[i]$ is strictly document-invariant:
   $$\operatorname{Var}_b(\text{counterfactual\_logits}[:, i]) \equiv \operatorname{Var}_b(\text{original\_logits}[:, i])$$
   Therefore, the per-label cross-document standard deviation and the set of labels passing V4 ($> 0.01$ std) must remain **numerically identical** within floating-point tolerance ($\le 10^{-6}$). This invariant must be formally verified by the diagnostic script.
4. **Relative Ranking Sensitivity:** While per-label variance across documents is invariant, cross-label ordering for any given document $b$ is non-linear and depends on the interaction between $\text{doc\_term}[b, i]$ and the subtracted offsets $\text{sa\_term}[i]$.

---

## 3. Strict Pre-Intervention Verification Gates

Before computing counterfactual metrics, the diagnostic must pass the following fail-closed gates:

1. **Checkpoint Identity & Hash Verification:**
   - `L2_STEP300_MODEL_ONLY.pt`: SHA-256 `99a06f1e7ba5bd72f3d855b61672b95a52f97f83e8db519e09b7992f2ca48e49`
   - `L2_A5B_STEP300_MODEL_ONLY.pt`: SHA-256 `b80852c7daafb626860dc2c981290eb2f7905e5303531c551b2fa14e0bb27022`
2. **Official BERT Checkpoint:**
   - Pinned revision: `86b5e0934494bd15c9632b12f734a8a67f723594` (`google-bert/bert-base-uncased`).
3. **Data & Split Quarantine:**
   - Test split strictly quarantined (never read, never staged).
   - Authentic AAPD `dev.csv` (SHA-256 `a85f612d4c1cb2be6d1de94cbcd9468cba97d221f826014264f4d279c82e220b`).
   - Exact frozen 200 Dev document IDs (`experiments/RUN_0/run0_val_sample_ids.json`).
   - Authentic 54-label order (SHA-256 `a3d662f3920259a94ed61d83f98f21cabe7f908d98a5662ccd9adc2413cc000b`).
4. **Forward-Pass & Algebraic Reconstruction Parity:**
   - Independent model forward-pass logits $\text{logits}_{\text{fwd}}$ must match reconstructed logits $\text{logits}_{\text{recon}}$ with maximum absolute difference $\le 10^{-5}$.
5. **Authentic Historical Predictions Agreement:**
   - Model probabilities $\sigma(\text{logits}_{\text{recon}})$ must match staged reference predictions from DR4 screening within $\le 10^{-5}$.
   - Baseline P@1 hits must reproduce exactly $63/200$.
   - Baseline dominant label must reproduce exactly (`cond-mat.dis-nn` for L2; `cmp-lg` for L2_A5B).

---

## 4. Preregistered Measurements

For both variants (`L2` and `L2_A5B`), compare **Original** vs. **Counterfactual** scores on:

| Metric | Measurement Method | Hypothesis / Expectation |
|---|---|---|
| **Top-1 Integer Hits** | Count of docs where Top-1 predicted label is in `true_labels` | Test whether hits exceed 63/200 |
| **P@1 Accuracy** | $\text{Hits} / 200$ | Baseline is 31.50% (frequency prior) |
| **Distinct Top-1 Labels** | Number of unique labels appearing at rank 1 | Test whether ranking collapses to single label |
| **Dominant Label & Fraction** | Most frequent Top-1 label and its percentage | Baseline: 100.0% (L2), 99.5% (L2_A5B) |
| **Distinct Top-3 Patterns** | Number of unique ordered tuples $(\text{rank}_1, \text{rank}_2, \text{rank}_3)$ | Baseline: 1 (L2), 2 (L2_A5B) |
| **Top-1 / Top-2 Margin** | Mean, std, min, max of $(\text{score}_1 - \text{score}_2)$ | Test margin expansion/compression |
| **V4 Passing Labels** | Count of labels with $\operatorname{std}_b(\text{logit}) > 0.01$ | **Must be identically 21 (L2) and 3 (L2_A5B)** |
| **Per-Label Logit Std Invariant** | Max difference $|\operatorname{std}_b(\text{orig}) - \operatorname{std}_b(\text{cf})|$ | **Must be $\le 10^{-6}$ for all 54 labels** |
| **Duplicate-Target Resolution** | Relative ranking of `cmp-lg` vs `cond-mat.dis-nn` | Evaluate if document term differentiates collinear targets |
| **Label Rank Flips** | Number of documents where Top-1 label changes | Quantify impact of static offset removal |

---

## 5. Scientific Interpretation Rules

1. **If Counterfactual P@1 Improves Significantly (> 63/200) and Diversity Expands:**
   - Demonstrates that the document-dependent signals encoded by BERT and scaled cross-attention contain useful discriminative ranking information, but were overwhelmed at decision time by the static self-attention offsets.
   - Does NOT prove that training without SA would learn identical document terms.
2. **If Counterfactual P@1 Does Not Improve ($\le 63/200$) or Degenerates:**
   - Demonstrates that the document-dependent terms themselves $\alpha[b, i] \cdot (W_i \cdot h_L[i])$ are either insufficiently discriminative or misaligned with true multi-label targets after 300 steps of joint optimization.
   - Refutes the conjecture that static SA offsets are the *sole* cause of ranking failure in the trained checkpoints.
3. **General Causality Guardrails:**
   - An inference-only counterfactual is an observational probe of a frozen state, NOT an inductive proof about model architectures trained under different optimizers, loss functions, or schedules.
   - It does NOT establish that dot-product attention without projection matrices is viable for general text-label ranking.

---

## 6. Execution Assets & Packaging

- Diagnostic probe script: `scripts/probe_dr4_frozen_sa_counterfactual.py`
- Test suite: `tests/test_dr4_frozen_sa_counterfactual.py`
- Generator: `scripts/gen_kaggle_notebook_dr4_counterfactual.py`
- Kaggle Notebook: `notebooks/lhrm_dr4_frozen_sa_cf_kaggle.ipynb`
- GPU Bundle: `bundles/lhrm_gpu_bundle_dr4_counterfactual_20261012.zip` (unique name; never overwrites decomposition bundle)
- Output Archive: `diagnostics/DR4_FROZEN_SA_CF_EVIDENCE.zip`
- Checksums: `diagnostics/DR4_FROZEN_SA_CF_SHA256SUMS.txt`
