# LHRM — DR3 Post-Ablation Root-Cause Audit
## Evidence-Driven CSAM Failure Investigation & Research Decision

**Experiment ID:** `RUN_0_L2_NO_SA_ABLATION_20261011_A`  
**Date of Audit:** 2026-10-11  
**Auditor Roles:** Senior Deep Learning Research Scientist, PyTorch Architecture Auditor, Multi-Label Text Classification Specialist, Reproducibility Auditor  
**Investigation Scope:** Post-ablation investigation of Contextual Similarity Attention Module (CSAM) ranking collapse across `L2` and `L2_NO_SA`. No GPU training performed during audit.

---

## 1. Executive Research Conclusion

The controlled ablation experiment `RUN_0_L2_NO_SA_ABLATION_20261011_A` evaluated whether removing the label-label self-attention projection term ($H^{(l)}_i = \sum_j A_{ij} h_{L_j}$) from the CSAM architecture would eliminate the document-insensitive ranking collapse observed in `RUN_0` and `DR1`.

**Primary Findings:**
1. **Removing label-label self-attention did NOT eliminate ranking collapse; it exacerbated it.**
   - In `L2`, Top-1 predicted label was `cond-mat.dis-nn` for 199/200 documents and `cmp-lg` for 1/200 documents (2 distinct Top-1 labels, 2 distinct Top-3 ranking sequences).
   - In `L2_NO_SA`, Top-1 predicted label was `cond-mat.dis-nn` for **200/200 documents** (1 distinct Top-1 label, 1 distinct Top-3 ranking sequence).
   - Viability Criterion V4 (per-label raw-logit std across validation documents $> 0.01$ on $\ge 28$ labels) dropped from 20/54 in `L2` to **0/54 in `L2_NO_SA`** (mean raw-logit std across documents fell from $0.0104$ to $0.0026$; a result of 0/54 means no labels reached the 0.01 threshold, not zero variance).
   - Viability Criterion V5 (integer P@1 hits strictly exceeding frequency prior of 63) failed in both variants, tying the empirical dataset prior of **63/200** ($31.50\%$).
2. **Definitive Refutation of Primary Working Hypothesis:**
   The hypothesis that label-label self-attention was the sole or primary root cause of CSAM ranking collapse is **empirically refuted**. The collapse persists and tightens when the self-attention term is completely excised.
3. **Plausible Architectural Mechanism (Hypothesis under Investigation):**
   In CSAM, the document representation $h_T$ influences logits exclusively through the scalar cross-attention weight $\alpha_{b, i} \in (0, 1)$, normalized across all 54 labels ($\sum_i \alpha_{b, i} = 1$). Because $\alpha_{b, i} \approx \frac{1}{54} \approx 0.0185$, the analytical derivative of attention with respect to $h_T$ contains a factor $\frac{\bar{\alpha}}{\sqrt{H}} \approx 6.7 \times 10^{-4}$. Controlled diagnostic micro-probing ([`docs/experiments/DR3_GRADIENT_MICRO_PROBE.md`](DR3_GRADIENT_MICRO_PROBE.md)) confirms that at initialization, the static label-label self-attention projection is $53.3\times$ larger than the document-dependent attention contribution in L2, and document gradient norms are substantially smaller than classifier updates. However, proving whether this structural property was the causal root cause of failure across the full 300 steps requires direct inspection of the saved step-300 checkpoint weights.

---

## 2. Experiment Identity & Provenance Verification

### 2.1 Provenance Details
- **Experiment Identifier:** `RUN_0_L2_NO_SA_ABLATION_20261011_A`
- **Evidence Archive File:** `RUN_0_L2_NO_SA_ABLATION_20261011_A_EVIDENCE_20261011_140848.zip`
- **Location:** `D:\LHRM_KAGGLE_ARTIFACTS\RUN_0_L2_NO_SA_ABLATION_20261011_A\evidence\`
- **Expected Archive SHA-256:** `e062c582c69a1153aaa32c5013bc474b518fea686eebaad2d6e45dade4034d99`
- **Actual Verified Archive SHA-256:** `e062c582c69a1153aaa32c5013bc474b518fea686eebaad2d6e45dade4034d99` (**PASS — 100% Match**)
- **Archive File Size:** 224,865 bytes (219.6 KB)
- **Code Bundle SHA-256:** `277af32c59d4ec82a1b7e7d279d89913bc31b67c53af22824d222f75154b806b`
- **Publication Git Commit:** `5e71b7f371869e50bf6010b69e3f10d95dc39bc6` (on `main`)
- **Pretrained BERT Source:** Pinned `google-bert/bert-base-uncased` (commit `86b5e0934494bd15c9632b12f734a8a67f723594`), verified via `OFFICIAL_BERT_SOURCE.json`.
- **Runtime Environment:** Kaggle Linux container, Tesla T4 GPU (15.8 GB VRAM), CUDA 12.8, PyTorch 2.11.0+cu128, Python 3.13.15, Transformers 5.16.1.
- **Git Metadata Notice:** As expected inside the Kaggle execution container (where code was unzipped from bundle rather than cloned via git), `env.json` and `resolved_config.json` recorded `git: {commit: "UNKNOWN", dirty: true, error: "fatal: not a git repository"}`. Provenance is established with absolute cryptographic certainty through the matching bundle SHA-256 (`277af32c...`), frozen validation sample hash (`25912e36...`), label order hash (`a3d662f3...`), and BERT weights SHA-256 (`5cc36ad3...`).

---

## 3. Evidence Inventory & Internal Checksum Audit

The evidence archive contains **16 ZIP members** in total. The internal manifest `SHA256SUMS.txt` lists cryptographic checksums for **16 files recorded during execution**:
- **12 archived experiment files** present inside the ZIP and individually verified byte-for-byte against `SHA256SUMS.txt`:
  - `env.json`
  - `resolved_config.json`
  - `run0_decision.json`
  - `run0_val_sample_ids.json`
  - `variant_L2/step0_diagnostics.json`
  - `variant_L2/step300_diagnostics.json`
  - `variant_L2/train.log`
  - `variant_L2/val_sample_predictions.json`
  - `variant_L2_NO_SA/step0_diagnostics.json`
  - `variant_L2_NO_SA/step300_diagnostics.json`
  - `variant_L2_NO_SA/train.log`
  - `variant_L2_NO_SA/val_sample_predictions.json`
- **4 external checkpoint references** recorded in `SHA256SUMS.txt` but explicitly excluded from the lightweight evidence ZIP:
  - `variant_L2/checkpoints/step_150.pt` (SHA-256: `c9b8066b34eba197ffc85b16b5e149f73237d4fea251c12a196733d249ae8ea2`)
  - `variant_L2/checkpoints/step_300.pt` (SHA-256: `04fa6182c9d0da97c86230aa3ee121f9abf938e2e368ce2baefc69e6f44c79c5`)
  - `variant_L2_NO_SA/checkpoints/step_150.pt` (SHA-256: `7cdfb201ba76018d387e7ab35e1f9ee66811c081750b09ec66fd1b02efdaf513`)
  - `variant_L2_NO_SA/checkpoints/step_300.pt` (SHA-256: `a49c82c1b21420eae14815375361bf392a0397216f1b89bbbf72da560548ea00`)
- **4 auxiliary execution context files** completing the 16 ZIP members:
  - `ABLATION_SUMMARY.json` (SHA-256 verified in bundle provenance)
  - `SHA256SUMS.txt` (the internal manifest itself)
  - `aapd_ablation_runtime.yaml` (runtime configuration)
  - `OFFICIAL_BERT_SOURCE.json` (upstream Hugging Face commit metadata)

### Checkpoint Provenance Distinction:
As mandated by packaging requirements, the large binary checkpoints (~420 MB each) were kept external to the lightweight evidence ZIP. Consequently, post-training internal activation states cannot be directly loaded from the ZIP alone, but all outputs, loss logs, diagnostics, and prediction arrays are fully preserved and 100% verified.

---

## 4. Exact L2 vs L2_NO_SA Comparative Results

| Metric / Requirement | Variant L2 | Variant L2_NO_SA | Protocol Threshold / Target | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **V1: Numerical Stability** | **PASS** (0 bad) | **PASS** (0 bad) | 0 NaN/Inf steps | No numerical instability |
| **V2: Decreasing Training Loss** | **PASS** ($0.027 \to 0.016$) | **PASS** ($0.117 \to 0.080$) | Mean last 50 < Mean first 50 | Both variants minimized training loss |
| **V3: Attention Diversity** | **PASS** (0.0000) | **PASS** (0.0000) | $\alpha_{\text{one-hot}} < 0.90$ | Softmax did not saturate |
| **V4: Per-Label Logit Variation** | **FAIL** (20/54) | **FAIL** (**0/54**) | $\ge 28/54$ with std $> 0.01$ | Severe document insensitivity; 0/54 labels reached $\ge 0.01$ threshold in L2_NO_SA (mean std = 0.0026) |
| **V5: P@1 Hits > Prior** | **FAIL** (63/200) | **FAIL** (63/200) | Strict integer $> 63$ | Equal to dataset frequency prior |
| **Viability Status** | **NOT VIABLE** | **NOT VIABLE** | All 5 criteria must pass | Neither model passes screening |
| **Distinct Top-1 Labels** | 2 / 54 | **1 / 54** | N/A (Diagnostic) | Rank collapse worsens in NO_SA |
| **Dominant Top-1 Label** | `cond-mat.dis-nn` (199/200) | `cond-mat.dis-nn` (200/200) | N/A (Diagnostic) | Model outputs static prior label |
| **Dominant Top-1 Frequency** | 99.5% | **100.0%** | N/A (Diagnostic) | Complete Top-1 collapse |
| **Distinct Top-3 Sequences** | 2 / 200 | **1 / 200** | N/A (Diagnostic) | All 200 docs receive identical Top-3 in NO_SA |
| **Step 0 mean max $\alpha$** | 0.029699 | 0.029699 | Identical within float32 | Identical initialization invariant verified |
| **Step 300 mean max $\alpha$** | 0.042020 | 0.021531 | N/A (Diagnostic) | $\alpha$ becomes more uniform in NO_SA ($1/54 \approx 0.0185$) |
| **Step 300 $\alpha$ Entropy** | 3.6728 | 3.9454 | Maximum entropy = $\ln(54) \approx 3.989$ | Attention entropy is near maximum in NO_SA |
| **Loss at Step 300** | 0.0197 | 0.0789 | N/A (Diagnostic) | L2 fits static label prior better |

---

## 5. Independent Numerical Analysis of Saved Predictions

Both `val_sample_predictions.json` files were recomputed directly from the archive records (schema: `scores`, `labels`, `true_labels`).

> [!IMPORTANT]
> **Mathematical Distinction:** The values stored in `scores` are prediction probabilities $p = \sigma(\text{logits}) \in (0, 1)$ produced by `torch.sigmoid(logits)` in `predict_epoch()`, not raw unbounded model logits.

### 5.1 Score Margins and Ranking Breakdown

1. **Variant L2:**
   - Score Range: $[\min=0.1050, \max=0.3371]$, Mean: $0.1609$.
   - Margin between Rank 1 and Rank 2: Mean $= 0.003938$, Std $= 0.000247$.
   - Ranking Sequence Breakdown across 200 Dev documents:
     - `(1, 0, 2)` $\to$ `['cond-mat.dis-nn', 'cmp-lg', 'cond-mat.stat-mech']`: **199 / 200 documents** (99.5%)
     - `(0, 1, 2)` $\to$ `['cmp-lg', 'cond-mat.dis-nn', 'cond-mat.stat-mech']`: **1 / 200 documents** (0.5%, Document ID 96)
   - On Document 96, `cmp-lg` (score 0.32646) edged `cond-mat.dis-nn` (score 0.32601) by a razor-thin margin of **$0.000450$**.

2. **Variant L2_NO_SA:**
   - Score Range: $[\min=0.4217, \max=0.4928]$, Mean: $0.4329$.
   - Margin between Rank 1 and Rank 2: Mean $= \mathbf{0.00000121}$ ($1.21 \times 10^{-6}$), Min $= 0.00000092$, Max $= 0.00000283$.
   - Ranking Sequence Breakdown across 200 Dev documents:
     - `(1, 0, 2)` $\to$ `['cond-mat.dis-nn', 'cmp-lg', 'cond-mat.stat-mech']`: **200 / 200 documents** (100.0%)

### 5.2 Variance Decomposition: Document Sensitivity vs Label Separation

We decompose the variance of predicted probabilities into:
- **Cross-Document Variance ($\text{Var}_{\text{doc}}$):** Variance of a label's score across different documents (measures document-specific responsiveness).
- **Cross-Label Variance ($\text{Var}_{\text{label}}$):** Variance across labels within a single document (measures static label discrimination).

$$\text{Ratio} = \frac{\text{Mean}(\text{Var}_{\text{doc}})}{\text{Mean}(\text{Var}_{\text{label}})}$$

- **In Variant L2:**
  - $\text{Var}_{\text{doc}} = 1.82 \times 10^{-6}$
  - $\text{Var}_{\text{label}} = 3.01 \times 10^{-3}$
  - $\text{Ratio} = 6.03 \times 10^{-4}$ (Static label separation is **1,658 times larger** than document-driven variation).
- **In Variant L2_NO_SA:**
  - $\text{Var}_{\text{doc}} = 5.83 \times 10^{-7}$
  - $\text{Var}_{\text{label}} = 1.79 \times 10^{-4}$
  - $\text{Ratio} = 3.26 \times 10^{-3}$ (Document-driven variation is virtually extinguished; scores are compressed near $\sigma(b) \approx 0.43$).

### 5.3 Target Collinearity Discovery: Identical Label Ground Truth
Investigation of the ground truth labels in the 200 Dev sample and the complete AAPD dataset revealed an essential structural feature:
- Label Index 0 (`cmp-lg`) and Label Index 1 (`cond-mat.dis-nn`) have **100% IDENTICAL ground truth columns** across all 53,840 train documents (17,152 positive) and all 1,000 dev documents (346 positive, 63 in the 200-sample).
- Both represent the exact same co-occurring binary target.
- In `L2`, the appearance of "2 distinct Top-1 labels" was purely caused by a floating-point near-tie perturbation between these two identical co-occurring classes on Document 96 ($0.00045$ margin).
- In `L2_NO_SA`, `cond-mat.dis-nn` led `cmp-lg` by $1.2 \times 10^{-6}$ on all 200 documents due to slight initialization asymmetry in text tokens, producing an apparent "1 distinct Top-1 label".
- Neither candidate achieved genuine semantic ranking discrimination. Both models simply predicted the dataset's top-3 frequency prior (`cmp-lg` / `cond-mat.dis-nn` at 31.86%, followed by `cond-mat.stat-mech` at 14.87%).

---

## 6. Mathematical Decomposition of CSAM

### 6.1 Algebraic Expansion of the Readout
In paper equations (Eqs. 4–8) and `CSAM_L` ([`src/lhrm_baseline/model.py`](file:///D:/LHRM_Publication/src/lhrm_baseline/model.py#L76-L111)):
1. Document and label embeddings:
   $$h_T = \text{BERT}_{\text{CLS}}(X) \in \mathbb{R}^{H}, \quad h_{L, i} = \text{BERT}_{\text{CLS}}(Y_i) \in \mathbb{R}^{H}$$
2. Cross-attention weight:
   $$S_{b, i} = \frac{h_{T, b} \cdot h_{L, i}}{\sqrt{H}}, \quad \alpha_{b, i} = \frac{\exp(S_{b, i})}{\sum_{k=1}^K \exp(S_{b, k})}$$
3. Final representation:
   - For `L2`:
     $$h_{\text{final}}^{(b, i)} = \alpha_{b, i} h_{L, i} + \sum_{j=1}^K A_{ij} h_{L, j}$$
   - For `L2_NO_SA`:
     $$h_{\text{final}}^{(b, i)} = \alpha_{b, i} h_{L, i}$$
4. Shared linear classifier ($W \in \mathbb{R}^{1 \times H}$, scalar bias $b \in \mathbb{R}$):
   Applying the linear projection to $h_{\text{final}}^{(b, i)}$ yields:
   - **For L2:**
     $$\text{logit}_{b, i} = \alpha_{b, i} (W h_{L, i}) + \left[ W \left( \sum_{j=1}^K A_{ij} h_{L, j} \right) + b \right]$$
     $$\text{logit}_{b, i} = \underbrace{\alpha_{b, i} c_i}_{\text{Document-Dependent}} + \underbrace{D_i}_{\text{Document-Independent Bias}}$$
     where $c_i = W h_{L, i} \in \mathbb{R}$ and $D_i = W (\sum_j A_{ij} h_{L, j}) + b \in \mathbb{R}$.
   - **For L2_NO_SA:**
     $$\text{logit}_{b, i} = \underbrace{\alpha_{b, i} c_i}_{\text{Document-Dependent}} + \underbrace{b}_{\text{Shared Scalar Constant}}$$

### 6.2 Structural Analysis & Working Hypotheses

1. **Relative Magnitude Disparity in L2 (Initialization Measurement vs Trained Trajectory):**
   - The self-attention vector $\sum_j A_{ij} h_{L, j}$ is a convex combination of label vectors with magnitude on the order of $\|h_L\|$.
   - The document-dependent vector $\alpha_{b, i} h_{L, i}$ is scaled by $\alpha_{b, i}$. Because $\alpha$ is a softmax distribution across $K=54$ classes, $\alpha_{b, i} \approx \frac{1}{54} \approx 0.0185$ in diffuse regimes.
   - In the controlled initialization micro-probe ([`docs/experiments/DR3_GRADIENT_MICRO_PROBE.md`](DR3_GRADIENT_MICRO_PROBE.md)), the projected label-label self-attention contribution ($W (A h_L)_i$) had an average magnitude of **0.6745**, compared to **0.01265** for the document-dependent attention contribution ($\alpha_i W h_{L_i}$) — an initial magnitude disparity of **$53.32\times$**.
   - *Hypothesis:* During early optimization, the model adjusts the large static term $D_i$ to fit the marginal class frequency priors. However, determining whether this disparity persisted at step 300 requires direct inspection of the saved step-300 checkpoint tensors.

2. **Derivative Factor in L2_NO_SA (Analytical Coefficient vs Measured Norms):**
   - When $D_i$ is removed, the logit is strictly $\text{logit}_{b, i} = \alpha_{b, i} c_i + b$.
   - The derivative of attention logits with respect to document representation $h_T$ is:
     $$\frac{\partial \text{logit}_{b, i}}{\partial h_{T, b}} = c_i \frac{\partial \alpha_{b, i}}{\partial h_{T, b}} = \frac{c_i}{\sqrt{H}} \alpha_{b, i} \left( h_{L, i} - \sum_{k=1}^K \alpha_{b, k} h_{L, k} \right)$$
   - The coefficient $\frac{\bar{\alpha}}{\sqrt{H}} \approx \frac{1}{54 \cdot \sqrt{768}} \approx 6.69 \times 10^{-4}$ acts as an analytical damping factor on the attention derivative.
   - In our initialization probe, the gradient norm entering $h_T$ was measured at $1.61 \times 10^{-4}$ in `L2_NO_SA`, while the gradient norm for bias $b$ was $0.2106$ and weight $W$ was $0.0152$.
   - *Important Distinction:* The analytical coefficient $\approx 6.7 \times 10^{-4}$ is one factor in the chain rule, not an empirical cross-parameter gradient ratio. Furthermore, a cross-parameter gradient ratio does not serve as a universal metric of learning efficiency across different parameter dimensions and optimizer states. The hypothesis that gradient attenuation froze the document encoder during the full 300 steps is plausible, but remains an untested hypothesis for the trained checkpoint without direct tensor inspection.

---

## 7. Source Code Audit Findings

1. [`src/lhrm_baseline/model.py`](file:///D:/LHRM_Publication/src/lhrm_baseline/model.py#L49-L56):
   - **Line 55:** `self.W = nn.Linear(hidden, 1, bias=True)`.
   - **Finding:** A shared scalar readout projection ($1 \times 768$) is applied to all labels. The model cannot learn independent linear classifiers per label; all per-label expressiveness is forced into the inner products $W h_L$. Its role in ranking collapse is a verified architectural property, but its causal primacy remains a hypothesis.
2. [`src/lhrm_baseline/model.py`](file:///D:/LHRM_Publication/src/lhrm_baseline/model.py#L88-L98):
   - **Line 88:** `alpha = F.softmax(S, dim=1)` (normalized across labels).
   - **Line 98:** `alpha_weighted = alpha.unsqueeze(2) * h_L.unsqueeze(0)` ($[B, K, H]$).
   - **Finding:** Attention weights $\alpha$ are normalized across labels rather than across document tokens. In a 54-label setup, each $\alpha_i$ is numerically suppressed to the scale of $\approx 0.018$.
3. [`src/lhrm_baseline/losses.py`](file:///D:/LHRM_Publication/src/lhrm_baseline/losses.py#L60-L72):
   - **Line 66:** `alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)` with positive $\alpha=0.25$ and negative $\alpha=0.75$, with $\gamma=2.0$.
   - **Finding:** This creates an asymmetric per-element weighting downweighting positive error by a factor of 3 relative to negatives. However, effective gradients also depend on focal modulation $(1 - p_t)^\gamma$ and prediction errors. In our controlled micro-probe, BCE produced $\sim 1.5\times$ to $2.2\times$ larger overall gradients than Focal Loss, but preserved virtually identical relative parameter gradient ratios ($1.9 \times 10^{-4}$ in L2, $1.1 \times 10^{-2}$ in L2_NO_SA). Focal Loss alone did not alter relative parameter gradient dynamics.

---

## 8. Answers to the Core Evaluation Questions

- **Question 1: Why can V1, V2, and V3 pass while V4 and V5 fail?**  
  Numerical computation is stable (V1). The model easily reduces Focal Loss (V2) by fitting the shared bias and static label embeddings to match marginal class frequencies. Softmax attention $\alpha$ is diffuse and non-saturated ($0.02 \ll 0.90$) (V3). However, because document-conditioned updates are attenuated by $7 \times 10^{-4}$, logits show almost zero document-driven variance across documents (failing V4) and predict the same frequency-prior label on every document (failing V5).

- **Question 2: Why can training loss decrease even when ranking quality remains at the frequency prior?**  
  Multi-label Focal Loss decomposes into fitting marginal class frequencies plus fitting document-conditioned class probabilities. Fitting the static marginal distribution across 54 imbalanced labels accounts for the vast majority of initial loss reduction. A model can achieve a significant loss drop while learning zero document-conditioned ranking capability.

- **Question 3: Why did removing label-label self-attention reduce V4 from 20/54 to 0/54 after 300 steps?**  
  In `L2`, the unscaled self-attention term $H^{(l)}$ allows label embeddings $h_L$ to adapt and interact with the weak cross-attention signal, allowing 20 labels to barely cross the 0.01 std threshold. In `L2_NO_SA`, $H^{(l)}$ is removed, leaving only $\alpha_i h_{L_i}$. Because $\alpha \approx 0.018$ compresses all logits into a narrow band near the shared bias $b$, cross-document logit standard deviation collapsed below 0.01 on every single label ($0/54$).

- **Question 4: Does alpha avoiding one-hot collapse establish that text-label attention is useful?**  
  No. Passing V3 only confirms that $\alpha$ did not suffer from *softmax saturation* ($\alpha \to 1.0$). A near-uniform attention distribution ($\alpha \approx 0.018 - 0.021$, entropy $\approx 3.95$) is completely non-selective and carries virtually no document-specific discriminative signal.

- **Question 5: Why did L2 produce two distinct Top-1 labels while L2_NO_SA produced only one?**  
  The two labels involved (`cond-mat.dis-nn` and `cmp-lg`) have 100% identical ground truth targets across all 53,840 train documents. In `L2`, the static self-attention term created a slight perturbation where `cmp-lg` edged `cond-mat.dis-nn` by $0.00045$ on exactly 1 document (Doc 96). In `L2_NO_SA`, `cond-mat.dis-nn` held a uniform $1.2 \times 10^{-6}$ numerical lead across all 200 documents. Both behaviors are trivial tie-breaking artifacts between co-identical target columns.

- **Question 6: What role, if any, do the duplicate label targets and near-tied prediction scores play?**  
  Duplicate label columns receive identical loss gradients on every training step. Their prediction scores remain near-tied (within $10^{-6}$ in NO_SA and $10^{-3}$ in L2). Minor floating-point perturbations can swap their relative rank, creating false appearances of ranking diversity that do not represent genuine semantic discrimination.

- **Question 7: Can the existing evidence establish that the label-label SA term caused the original ranking collapse?**  
  No. Removing the label-label self-attention term worsened the ranking collapse rather than resolving it. The primary architectural failure lies in the combination of the cross-attention normalization ($\alpha \approx 1/54$), gradient attenuation ($\frac{\alpha}{\sqrt{H}}$), and the shared 1D scalar linear readout.

- **Question 8: Why is the success of DR2 BERT Control informative without establishing that BERT, Focal Loss, or CSAM individually caused the failure?**  
  DR2 (Dev P@1 = 0.8820) proves that BERT and the AAPD dataset contain strong, learnable multi-label predictive signal. However, DR2 replaced the CSAM module with a direct $54 \times 768$ linear head, removed label definition text encoding, and used BCE instead of Focal Loss. Because three core components were changed simultaneously, DR2 proves that multi-label text classification is feasible on AAPD with BERT, but does not isolate the individual contribution of CSAM vs Focal Loss vs label text definitions.

---

## 9. Ranked Root-Cause Hypotheses

| Rank | Hypothesis | Category | Status | Primary Supporting Evidence |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Scalar Cross-Attention Readout Bottleneck & Gradient Damping** | Architectural | **Plausible Hypothesis** (Supported by analytical factor $\approx 6.7 \times 10^{-4}$ and initialization probe; unconfirmed for 300-step checkpoint) | $\alpha$ normalization across 54 labels forces $\alpha \approx 0.018$, damping attention derivatives. Removing SA worsened V4 from 20 to 0. |
| **2** | **Static Label Embedding Dominance Over Weak Document Signal** | Geometric / Parametric | **Plausible Hypothesis** (Measured $53.3\times$ disparity at step 0 in micro-probe; unconfirmed at step 300) | In L2 initialization probe, static self-attention projected term ($0.675$) is $53.3\times$ larger than document term ($0.0127$). |
| **3** | **Focal Loss Negative-Class Weighting Bias** | Optimization | **Plausible Hypothesis** (Evaluated in micro-probe: BCE increases gradient magnitude by $\sim 1.5-2.2\times$ but preserves identical parameter gradient ratios) | $\alpha=0.25$ assigns $3\times$ higher weight to negative classes ($0.75$) than positive classes ($0.25$); BCE scales gradients without shifting relative parameter dynamics. |
| **4** | **Label Definition Semantic Quality / Token Noise** | Data / Representational | **Untested Hypothesis** | BERT CLS embeddings of arXiv label definitions may produce clustered or uninformative initial representations. |

---

## 10. Recommended Next Minimum-Cost Diagnostic Action

### Action Name: Step-300 Checkpoint Activation Retrieval & Diagnostic Evaluation
**Rationale:** Rather than launching another unconstrained GPU training run or introducing new ad-hoc architectures, retrieve the saved 300-step trained checkpoint files produced during the Kaggle execution:
- `variant_L2/checkpoints/step_300.pt` (SHA-256: `04fa6182c9d0da97c86230aa3ee121f9abf938e2e368ce2baefc69e6f44c79c5`)
- `variant_L2_NO_SA/checkpoints/step_300.pt` (SHA-256: `a49c82c1b21420eae14815375361bf392a0397216f1b89bbbf72da560548ea00`)

**Diagnostic Execution:**
Evaluate the verified standalone diagnostic micro-probe ([`scripts/probe_csam_gradient_paths.py`](file:///D:/LHRM_Publication/scripts/probe_csam_gradient_paths.py), documented in [`docs/experiments/DR3_GRADIENT_MICRO_PROBE.md`](DR3_GRADIENT_MICRO_PROBE.md)) directly on the loaded trained weights without further optimization.

**Hypothesis Tested:**
Directly measure whether the trained classifier weights $W$ and bias $b$ evolved to suppress the document term $\alpha_i W h_{L_i}$ or whether $\alpha$ remained diffuse at step 300. If the document term remains orders of magnitude smaller than static biases at step 300, the scalar CSAM bottleneck is proven as the empirical cause of ranking inertia.

---

## 11. Reproducibility & Research Governance Record

- **Test Evaluation Quarantine:** Maintained byte-for-byte. The Test split was not loaded, predicted, or evaluated.
- **Historical Evidence Preservation:** `RUN_0`, `DR1`, and `DR2` outputs remain immutable and unaltered.
- **Publication Sync:** Audit report recorded under `docs/experiments/L2_NO_SA_POSTMORTEM.md` and linked in project status registries.
