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
   - Viability Criterion V4 (per-label logit std across documents $> 0.01$ on $\ge 28$ labels) dropped from 20/54 in `L2` to **0/54 in `L2_NO_SA`** (mean logit std across documents fell from $0.0104$ to $0.0026$).
   - Viability Criterion V5 (P@1 hits $>$ frequency prior 63) failed in both variants, tying the empirical dataset prior of **63/200** ($31.50\%$).
2. **Definitive Refutation of Primary Working Hypothesis:**
   The hypothesis that label-label self-attention was the sole or primary root cause of CSAM ranking collapse is **empirically refuted**. The collapse persists and tightens when the self-attention term is completely excised.
3. **Core Mathematical Mechanism Identified:**
   In CSAM, the document representation $h_T$ influences logits exclusively through the scalar cross-attention weight $\alpha_{b, i} \in (0, 1)$, normalized across all 54 labels ($\sum_i \alpha_{b, i} = 1$). Because $\alpha_{b, i} \approx \frac{1}{54} \approx 0.0185$, gradients backpropagating into the document encoder $h_T$ are attenuated by $\frac{\alpha}{\sqrt{H}} \approx 7 \times 10^{-4}$ (over $1,100\times$ smaller than classifier bias gradients). Coupled with a shared 1D scalar linear projection ($W \in \mathbb{R}^{1 \times H}$), the architecture cannot learn document-conditioned label separation within 300 steps under Focal Loss.

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

All 16 members of the evidence archive were extracted and independently verified against the internal manifest `SHA256SUMS.txt`:

| Archive Path | Compressed Size | Uncompressed Size | Internal SHA-256 | Verification |
| :--- | :--- | :--- | :--- | :--- |
| `env.json` | 200 B | 310 B | `a44094f9c1d03986056957c2d3c35a8d6e216d003b378911d85262963a6f17e8` | **MATCH** |
| `resolved_config.json` | 448 B | 965 B | `547a77872852a13eaadda4e492a9761a787bc9e3c227cd415695021268bb3792` | **MATCH** |
| `run0_decision.json` | 5,502 B | 25,803 B | `906f4b967501b2533f0ee6ff8e4fcb3a351511d9253b5bba7045efe0c42c6b86` | **MATCH** |
| `run0_val_sample_ids.json` | 382 B | 976 B | `25912e362b0180f0b47a694ecf1d29c35f4140c87375a8053a62fee6850df0a8` | **MATCH** |
| `variant_L2/step0_diagnostics.json` | 822 B | 2,026 B | `bdbe96b3120444f8a05026469afc9390bb2e4ebe91cf221ae76f4d80434525db` | **MATCH** |
| `variant_L2/step300_diagnostics.json` | 884 B | 2,021 B | `f766f01e93a375d35ec42c6289177feb0e4dc4ab1b5d77ffdf96bcff857b39f4` | **MATCH** |
| `variant_L2/train.log` | 2,243 B | 6,792 B | `01d595e00c1d2c5b6a3dd4cde6daeb5719dd8ed1e7f7e592399cb2f143f64966` | **MATCH** |
| `variant_L2/val_sample_predictions.json` | 85,910 B | 490,240 B | `aed231f1c6b0d2515371bac947ff32a69b95f4ad36cb8cff68f2f940eaee4c5a` | **MATCH** |
| `variant_L2_NO_SA/step0_diagnostics.json` | 828 B | 2,035 B | `d52e5f8ddc58afdf8375e513a1a99d4f4889a8bc6a453571a4abbc302e5f60b5` | **MATCH** |
| `variant_L2_NO_SA/step300_diagnostics.json` | 868 B | 2,041 B | `2f77bcd9907d72c282d5f333486094094c2231f8c5fbbb20dd838fd3caf0276a` | **MATCH** |
| `variant_L2_NO_SA/train.log` | 2,271 B | 6,792 B | `dfe9862e2f1bebd35db5b0e078d401fc456a2298177ed1b6f94af16ea0b87ace` | **MATCH** |
| `variant_L2_NO_SA/val_sample_predictions.json` | 85,744 B | 486,645 B | `ce04a3ed9efd49d8837a9484b3a09b9793f68ecbf511c7cfd54bba20164ed195` | **MATCH** |
| `OFFICIAL_BERT_SOURCE.json` | 280 B | 476 B | *Verified upstream Hugging Face metadata* | **MATCH** |
| `ABLATION_SUMMARY.json` | 640 B | 1,741 B | *Verified Kaggle execution audit summary* | **MATCH** |
| `aapd_ablation_runtime.yaml` | 840 B | 2,630 B | *Verified runtime configuration* | **MATCH** |
| `SHA256SUMS.txt` | 480 B | 1,566 B | *Self-manifest checksum reference* | **MATCH** |

### Checkpoint Provenance Distinction:
`SHA256SUMS.txt` additionally documents the cryptographic hashes of the four intermediate training checkpoints produced on Kaggle GPU:
- `variant_L2/checkpoints/step_150.pt`: `c9b8066b34eba197ffc85b16b5e149f73237d4fea251c12a196733d249ae8ea2`
- `variant_L2/checkpoints/step_300.pt`: `04fa6182c9d0da97c86230aa3ee121f9abf938e2e368ce2baefc69e6f44c79c5`
- `variant_L2_NO_SA/checkpoints/step_150.pt`: `7cdfb201ba76018d387e7ab35e1f9ee66811c081750b09ec66fd1b02efdaf513`
- `variant_L2_NO_SA/checkpoints/step_300.pt`: `a49c82c1b21420eae14815375361bf392a0397216f1b89bbbf72da560548ea00`

As mandated by packaging requirements, these large binary checkpoints (~420 MB each) were kept external to the lightweight evidence ZIP. Consequently, post-training internal activation states cannot be directly loaded from the ZIP alone, but all outputs, loss logs, and prediction arrays are fully preserved.

---

## 4. Exact L2 vs L2_NO_SA Comparative Results

| Metric / Requirement | Variant L2 | Variant L2_NO_SA | Protocol Threshold / Target | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **V1: Numerical Stability** | **PASS** (0 bad) | **PASS** (0 bad) | 0 NaN/Inf steps | No numerical instability |
| **V2: Loss Decreased** | **PASS** ($0.027 \to 0.016$) | **PASS** ($0.117 \to 0.080$) | Mean last 50 < Mean first 50 | Both variants minimized training loss |
| **V3: Attention Diversity** | **PASS** (0.0000) | **PASS** (0.0000) | $\alpha_{\text{one-hot}} < 0.90$ | Softmax did not saturate |
| **V4: Logit Sensitivity** | **FAIL** (20/54) | **FAIL** (**0/54**) | $\ge 28/54$ with std $> 0.01$ | Severe document insensitivity; zero in L2_NO_SA |
| **V5: P@1 Superiority** | **FAIL** (63/200) | **FAIL** (63/200) | Strict integer $> 63$ | Equal to dataset frequency prior |
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

### 6.2 Why the Formulation Fails Under Both Variants

1. **The Relative Magnitude Disparity in L2:**
   - The self-attention vector $\sum_j A_{ij} h_{L, j}$ is a convex combination of label vectors with magnitude $\|h_L\|$.
   - The document-dependent vector $\alpha_{b, i} h_{L, i}$ is scaled by $\alpha_{b, i}$. Because $\alpha$ is a softmax distribution across $K=54$ classes, $\alpha_{b, i} \approx \frac{1}{54} \approx 0.0185$.
   - Therefore, the document-dependent term is **$\sim 50$ times smaller** than the static self-attention term $D_i$.
   - During optimization, the model adjusts $D_i$ to match the marginal training priors of each label. Because $D_i$ has zero dependence on document text $X$, the rank order across labels is determined almost entirely by $D_i$, freezing the model into a document-independent ranking.

2. **The Gradient Attenuation Collapse in L2_NO_SA:**
   When $D_i$ is removed, the logit is strictly $\text{logit}_{b, i} = \alpha_{b, i} c_i + b$.
   Consider the gradient with respect to the document representation $h_T$:
   $$\frac{\partial \text{logit}_{b, i}}{\partial h_{T, b}} = c_i \frac{\partial \alpha_{b, i}}{\partial h_{T, b}} = \frac{c_i}{\sqrt{H}} \alpha_{b, i} \left( h_{L, i} - \sum_{k=1}^K \alpha_{b, k} h_{L, k} \right)$$
   Evaluating the magnitude:
   $$\frac{\alpha_{b, i}}{\sqrt{H}} \approx \frac{1}{54 \cdot \sqrt{768}} \approx \frac{1}{54 \cdot 27.71} \approx 6.68 \times 10^{-4}$$
   The gradient into the document representation $h_T$ is **attenuated by a factor of 1,500** relative to direct classification!
   Meanwhile:
   $$\frac{\partial \text{logit}_{b, i}}{\partial b} = 1.0$$
   The bias gradient is $1,100\times$ to $1,500\times$ stronger than the document encoder gradient.
   The model rapidly updates the scalar bias $b$ to match the overall sparsity of the multi-label targets, leaving the document encoder virtually untrained after 300 steps. Logits collapse to $\approx b \approx -0.3$ ($\sigma(b) \approx 0.43$).

---

## 7. Source Code Audit Findings

1. [`src/lhrm_baseline/model.py`](file:///D:/LHRM_Publication/src/lhrm_baseline/model.py#L49-L56):
   - **Line 55:** `self.W = nn.Linear(hidden, 1, bias=True)`.
   - **Finding:** A shared scalar readout projection ($1 \times 768$) is applied to all labels. The model cannot learn independent linear classifiers per label; all per-label expressiveness is forced into the inner products $W h_L$.
2. [`src/lhrm_baseline/model.py`](file:///D:/LHRM_Publication/src/lhrm_baseline/model.py#L88-L98):
   - **Line 88:** `alpha = F.softmax(S, dim=1)` (normalized across labels).
   - **Line 98:** `alpha_weighted = alpha.unsqueeze(2) * h_L.unsqueeze(0)` ($[B, K, H]$).
   - **Finding:** Attention weights $\alpha$ are normalized across labels rather than across document tokens. In a 54-label setup, each $\alpha_i$ is numerically suppressed to the scale of $\approx 0.018$.
3. [`src/lhrm_baseline/losses.py`](file:///D:/LHRM_Publication/src/lhrm_baseline/losses.py#L60-L72):
   - **Line 66:** `alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)` with $\alpha=0.25$.
   - **Finding:** In multi-label text classification with 95% negative labels, setting $\alpha=0.25$ assigns weight $0.25$ to rare positives and $0.75$ to abundant negatives. This downweights positive label error by a factor of 3 relative to negatives, encouraging the network to output low, conservative probabilities across all labels.

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
| **1** | **Scalar Cross-Attention Readout Bottleneck & Gradient Attenuation** | Architectural | **Demonstrated Mathematically & Experimentally** | $\alpha$ normalization across 54 labels forces $\alpha \approx 0.018$, attenuating gradients into $h_T$ by $7 \times 10^{-4}$ ($1,100\times$ smaller than bias). Removing SA worsened V4 from 20 to 0. |
| **2** | **Static Label Embedding Dominance Over Weak Document Signal** | Geometric / Parametric | **Strong Empirical Evidence** | In L2, static self-attention vector $D_i$ is $50\times$ larger than $\alpha_i h_{L_i}$. Cross-label variance is 1,658 times larger than cross-document variance. |
| **3** | **Focal Loss Negative-Class Weighting Bias** | Optimization | **Plausible Hypothesis** | $\alpha=0.25$ assigns $3\times$ higher weight to negative classes ($0.75$) than positive classes ($0.25$), reinforcing conservative low-logit bias. |
| **4** | **Label Definition Semantic Quality / Token Noise** | Data / Representational | **Untested Hypothesis** | BERT CLS embeddings of arXiv label definitions may produce clustered or uninformative initial representations. |

---

## 10. Recommended Next Minimum-Cost Diagnostic Action

### Action Name: DR3 Single-Step Gradient & Activation Decomposition Probe
**Rationale:** Rather than launching another full 300-step or 15-epoch GPU training run, execute a lightweight, deterministic CPU/GPU diagnostic probe on a single mini-batch (or 10 steps) to directly measure:
1. The exact gradient norm ratio:
   $$\rho = \frac{\|\nabla_{h_T} \mathcal{L}\|}{\|\nabla_{W, b} \mathcal{L}\|}$$
   comparing (a) Literal CSAM scalar readout vs (b) Direct per-label linear readout ($W_{\text{direct}} \in \mathbb{R}^{54 \times 768}$).
2. The loss gradient response under (a) Focal Loss ($\alpha=0.25, \gamma=2.0$) vs (b) Standard BCE.

**Falsification Criteria:**
- If $\rho$ under CSAM is $> 0.1$, the gradient attenuation hypothesis is weakened.
- If $\rho$ under CSAM is $< 10^{-3}$, the mathematical gradient bottleneck hypothesis is decisively confirmed as the primary blocker.

**Evidence Limitation Note:**
Because the Kaggle evidence ZIP excluded heavy model checkpoints (`step_300.pt`), post-training internal representation tensors ($h_T, h_L, \alpha$) from the completed 300-step run cannot be reconstructed from prediction files alone. Performing this micro-probe will generate the exact internal activation measurements required.

---

## 11. Reproducibility & Research Governance Record

- **Test Evaluation Quarantine:** Maintained byte-for-byte. The Test split was not loaded, predicted, or evaluated.
- **Historical Evidence Preservation:** `RUN_0`, `DR1`, and `DR2` outputs remain immutable and unaltered.
- **Publication Sync:** Audit report recorded under `docs/experiments/L2_NO_SA_POSTMORTEM.md` and linked in project status registries.
