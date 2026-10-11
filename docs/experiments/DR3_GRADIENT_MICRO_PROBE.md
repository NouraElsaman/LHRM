# LHRM — DR3 Controlled Gradient & Architectural Micro-Probe
## Quantitative Inspection of CSAM Forward Paths, Gradient Attenuation, and Loss Formulation

**Document Reference:** `DR3_GRADIENT_MICRO_PROBE.md`  
**Associated Audit:** `RUN_0_L2_NO_SA_ABLATION_20261011_A` Postmortem (`DR3`)  
**Diagnostic Script:** [`scripts/probe_csam_gradient_paths.py`](file:///D:/LHRM_Publication/scripts/probe_csam_gradient_paths.py)  
**Test Suite:** [`tests/test_csam_gradient_probe.py`](file:///D:/LHRM_Publication/tests/test_csam_gradient_probe.py)  
**Date:** 2026-10-11  

---

## 1. Executive Summary & Diagnostic Scope

Following the completion of the `RUN_0_L2_NO_SA_ABLATION_20261011_A` screening (where removing label-label self-attention failed to alleviate ranking collapse and reduced V4 from 20/54 to 0/54), this standalone diagnostic micro-probe was conducted to measure:
1. The exact gradient norms flowing into document representations ($h_T$), label representations ($h_L$), and classifier parameters ($W, b$).
2. The algebraic decomposition of projected logits into document-dependent ($\alpha_i W h_{L_i}$) and document-independent ($W (A h_L)_i$, $b$) components.
3. The controlled differential effect between Focal Loss ($\alpha=0.25, \gamma=2.0$) and standard Binary Cross-Entropy (BCE) under identical logits and targets.
4. Attention dispersion, entropy, and analytical gradient scaling factors.

> [!IMPORTANT]
> **Scientific Methodology & Scope:**
> This probe is executed on a single controlled mini-batch without optimizer updates, architectural changes, or GPU training.
> Because the Kaggle evidence ZIP excluded heavy weight checkpoints (`step_300.pt`), and the local environment does not have pretrained BERT weights in HuggingFace cache, this evaluation is conducted as a **clearly labeled synthetic structural probe** using approved ground-truth targets from AAPD Train. Its results characterize the architectural and gradient properties at initialization and must not be extrapolated directly as an empirical measurement of the trained state after 300 steps.

---

## 2. Input Provenance & Probe Configuration

- **Target Provenance:** First 16 samples ($B=16, k=54$) from approved AAPD Train ground truth (`Data/processed/AAPD/train_labels.npy`).
- **Target Sparsity:** Active labels per document: Mean $= 2.44$ ($4.52\%$ positive labels).
- **Representation State:** Synthetic structural probe ($H=768, k=54, B=16$, seed 42) scaled to standard BERT [CLS] hidden dimension.
- **Classifier Initialization:** Shared linear projection $W \in \mathbb{R}^{1 \times H}$ initialized with standard normal variance scaled by $1/\sqrt{H}$; bias $b = 0$.
- **Test Quarantine:** Maintained strictly. No Test data or validation splits beyond the approved diagnostic targets were accessed.
- **Checkpoint Status:** `variant_L2/checkpoints/step_300.pt` and `variant_L2_NO_SA/checkpoints/step_300.pt` were confirmed absent from local storage (excluded from evidence ZIP); probe evaluated on structural initialization.

---

## 3. Measured Numerical Results

All measurements were generated using [`scripts/probe_csam_gradient_paths.py`](file:///D:/LHRM_Publication/scripts/probe_csam_gradient_paths.py) with seed 42:

| Metric / Condition | `L2` (Focal Loss) | `L2` (BCE Loss) | `L2_NO_SA` (Focal Loss) | `L2_NO_SA` (BCE Loss) |
| :--- | :---: | :---: | :---: | :---: |
| **Loss Value** | 0.1811 | 0.7130 | 0.1259 | 0.6921 |
| **Finite Grads Verified** | **True** | **True** | **True** | **True** |
| **$\|\nabla_{h_T} \mathcal{L}\|$ (Document Norm)** | $2.3034 \times 10^{-4}$ | $3.4510 \times 10^{-4}$ | $1.6089 \times 10^{-4}$ | $3.6488 \times 10^{-4}$ |
| **$\|\nabla_{h_L} \mathcal{L}\|$ (Label Norm)** | $4.1743 \times 10^{-2}$ | $6.5442 \times 10^{-2}$ | $5.6859 \times 10^{-4}$ | $1.2631 \times 10^{-3}$ |
| **$\|\nabla_W \mathcal{L}\|$ (Classifier Weight Norm)** | $1.1903$ | $1.8317$ | $1.5248 \times 10^{-2}$ | $3.3478 \times 10^{-2}$ |
| **$\|\nabla_b \mathcal{L}\|$ (Classifier Bias Norm)** | $0.2284$ | $0.4299$ | $0.2106$ | $0.4577$ |
| **Ratio $\|\nabla_{h_T}\| / \|\nabla_W\|$** | $1.935 \times 10^{-4}$ | $1.884 \times 10^{-4}$ | $1.055 \times 10^{-2}$ | $1.090 \times 10^{-2}$ |
| **Ratio $\|\nabla_{h_T}\| / \sqrt{\|\nabla_W\|^2 + \|\nabla_b\|^2}$** | $1.899 \times 10^{-4}$ | $1.835 \times 10^{-4}$ | $7.618 \times 10^{-4}$ | $7.950 \times 10^{-4}$ |
| **Max Attention Weight ($\max \alpha$)** | 0.1148 | 0.1148 | 0.1148 | 0.1148 |
| **Attention Entropy** | 3.5288 | 3.5288 | 3.5288 | 3.5288 |
| **Uniform Entropy Benchmark ($\ln 54$)** | 3.9890 | 3.9890 | 3.9890 | 3.9890 |
| **Entropy / Uniform Ratio** | 88.46% | 88.46% | 88.46% | 88.46% |
| **Analytical Derivative Factor $\bar{\alpha}/\sqrt{H}$** | $6.694 \times 10^{-4}$ | $6.694 \times 10^{-4}$ | $6.694 \times 10^{-4}$ | $6.694 \times 10^{-4}$ |
| **Document-Dependent Term Abs Mean** | $1.2651 \times 10^{-2}$ | $1.2651 \times 10^{-2}$ | $1.2651 \times 10^{-2}$ | $1.2651 \times 10^{-2}$ |
| **Document-Dependent Term Std Across Docs** | $1.3482 \times 10^{-2}$ | $1.3482 \times 10^{-2}$ | $1.3482 \times 10^{-2}$ | $1.3482 \times 10^{-2}$ |
| **Label-Label SA Term Abs Mean** | $0.6745$ | $0.6745$ | *N/A (Excised)* | *N/A (Excised)* |
| **Ratio SA Term / Document Term** | **53.32x** | **53.32x** | *N/A* | *N/A* |

---

## 4. Analytical Deconstructions & Verified Observations

### 4.1 Algebraic Logit Decomposition
For any document $b$ and label $i$, CSAM logits decompose exactly as:
$$\text{logit}_{b, i} = \text{doc\_term}_{b, i} + \text{sa\_term}_i + b$$
where:
$$\text{doc\_term}_{b, i} = \alpha_{b, i} (W h_{L, i})$$
$$\text{sa\_term}_i = W \left( \sum_j A_{ij} h_{L, j} \right) \quad (\text{in } L2; \; 0 \text{ in } L2\_NO\_SA)$$
$$\text{bias\_term} = b$$

- In `L2` at initialization, the static label-label self-attention projected term has an average magnitude of **0.6745**, compared to **0.01265** for the document-dependent attention contribution.
- This represents an initial magnitude disparity of **$53.32\times$**, directly explaining why early ranking outputs in `L2` mirror the static label embedding space rather than document content.

### 4.2 Gradient Flow into Document Representations
The analytical derivative of attention logits with respect to $h_{T, b}$ is:
$$\frac{\partial \text{logit}_{b, i}}{\partial h_{T, b}} = \frac{W h_{L, i}}{\sqrt{H}} \alpha_{b, i} \left( h_{L, i} - \sum_k \alpha_{b, k} h_{L, k} \right)$$
- Because $\sum_k \alpha_{b, k} = 1$ across 54 labels, the average attention weight is $\bar{\alpha} = \frac{1}{54} \approx 0.0185$.
- The scaling factor $\frac{\bar{\alpha}}{\sqrt{H}} \approx \frac{0.0185}{\sqrt{768}} \approx 6.69 \times 10^{-4}$ acts as an analytical damping coefficient on document representation updates.
- In the measured probe, the gradient norm entering $h_T$ ($1.61 \times 10^{-4}$ in `L2_NO_SA`) is roughly **$1,300\times$ smaller** than the gradient updating the scalar bias $b$ ($0.2106$).
- However, comparing gradient norms across different parameter tensors (e.g. $\|\nabla_{h_T}\|$ vs $\|\nabla_W\|$ or $\|\nabla_b\|$) is **not** an absolute metric of learning speed, as learning dynamics also depend on optimizer state, parameter dimensions ($768$ vs $1$), and learning rates.

### 4.3 Focal Loss vs BCE Controlled Comparison
- Applying standard BCE to the exact same logits and targets produces roughly **$1.5\times$ to $2.2\times$ larger gradients** across all components ($\|\nabla_{h_T}\|$ increases from $2.30 \times 10^{-4}$ to $3.45 \times 10^{-4}$ in L2).
- Crucially, the **relative ratios** between parameters remain almost completely identical:
  - Ratio $\|\nabla_{h_T}\| / \|\nabla_W\|$ in L2 is $1.935 \times 10^{-4}$ under Focal Loss and $1.884 \times 10^{-4}$ under BCE.
  - Ratio $\|\nabla_{h_T}\| / \|\nabla_W\|$ in L2_NO_SA is $1.055 \times 10^{-2}$ under Focal Loss and $1.090 \times 10^{-2}$ under BCE.
- **Finding:** While Focal Loss ($\alpha=0.25, \gamma=2.0$) scales down overall gradient magnitude on sparse multi-label targets, it does **not** create a unique parameter-imbalance distortion compared to standard BCE.

---

## 5. Evaluation of Root-Cause Hypotheses

Based on the measured micro-probe evidence:

1. **Hypothesis 1 (Derivative Damping & Softmax Normalization across Labels):**
   - *Status: STRENGTHENED as an architectural property at initialization.*
   - Normalizing attention across 54 labels forces $\alpha \approx 0.0185$, introducing an analytical factor $\approx 6.7 \times 10^{-4}$ into the derivative. In our initialization probe, $\|\nabla_{h_T}\|$ is on the order of $10^{-4}$ while classifier updates are orders of magnitude larger.
   - *Limitation:* Does not prove that document gradients remained suppressed across all 300 optimizer steps of AdamW training on GPU.

2. **Hypothesis 2 (Label-Label Self-Attention Dominance in L2):**
   - *Status: STRENGTHENED at initialization, but REFUTED as the sole cause of collapse.*
   - In L2, the static self-attention term is $53.32\times$ larger than the document attention term at step 0.
   - However, excising this term entirely (`L2_NO_SA`) did not restore document sensitivity (V4 dropped from 20/54 to 0/54, and Top-1 collapsed to 100% `cond-mat.dis-nn`). Thus, self-attention dominance was a symptom, not the root cause.

3. **Hypothesis 3 (Focal Loss Parameter Distortion):**
   - *Status: WEAKENED as a primary differentiator between variants.*
   - Controlled probe confirms that switching from Focal Loss to BCE does not alter the relative gradient distribution between document embeddings and classifier weights.

---

## 6. Scientific Limitations & Missing Evidence

1. **Synthetic Structural Probe:**
   - Because local HuggingFace cache lacked pretrained BERT weights, this probe used synthetic representations with standard normal scaling. While tensor dimensions and algebraic linearity are identical to the true architecture, real BERT representations may possess clustered geometries that alter absolute gradient norms.
2. **Missing Checkpoint Tensors:**
   - The authentic Kaggle evidence ZIP contains predictions, metrics, and logs, but excludes `variant_L2/checkpoints/step_300.pt` and `variant_L2_NO_SA/checkpoints/step_300.pt`.
   - Consequently, we cannot directly measure whether the document-dependent logit contribution grew, shrank, or remained at $0.012$ after 300 steps of AdamW optimization.

---

## 7. Recommended Next Research Action

### Action Name: Step-300 Checkpoint Activation Retrieval & Diagnostic Evaluation
- **Target Artifacts Required:**
  - `variant_L2/checkpoints/step_300.pt` (SHA-256: `04fa6182c9d0da97c86230aa3ee121f9abf938e2e368ce2baefc69e6f44c79c5`)
  - `variant_L2_NO_SA/checkpoints/step_300.pt` (SHA-256: `a49c82c1b21420eae14815375361bf392a0397216f1b89bbbf72da560548ea00`)
- **Diagnostic Method:**
  Evaluate the exact same decomposition script ([`scripts/probe_csam_gradient_paths.py`](file:///D:/LHRM_Publication/scripts/probe_csam_gradient_paths.py)) directly on the loaded step-300 checkpoint tensors without training.
- **Hypothesis Tested:**
  Directly measure whether the trained classifier weights $W$ and bias $b$ evolved to suppress the document term $\alpha_i W h_{L_i}$ or whether $\alpha$ remained diffuse at step 300.
