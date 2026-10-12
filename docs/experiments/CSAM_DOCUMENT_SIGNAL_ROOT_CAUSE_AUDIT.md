# Complete CSAM Implementation Forensic Audit & Quantitative Root-Cause Matrix

**Document ID:** `CSAM_DOCUMENT_SIGNAL_ROOT_CAUSE_AUDIT_20261012`  
**Date:** 2026-10-12  
**Status:** FORENSIC AUDIT COMPLETE  
**Audited Artifacts:** Source Code (`src/lhrm_baseline/`), Author Codebase (`Model/lhrm.py`), Checkpoints & Evidence (`DR4_A5B`, `DR4_POSTTRAIN_DECOMPOSITION`, `DR4_FROZEN_SA_CF`)

---

## 1. Executive Summary

Across four consecutive experimental stages (`RUN_0` Variant L, `DR1` Variant L2, `DR3/DR3b` L2_NO_SA replay, and `DR4` L2_A5B unshared classifier), every literal CSAM variant has failed the registered viability screen at Step 300 with an identical failure signature:
- **V4 Failure:** Only 0 to 21 of 54 labels exhibit cross-document logit standard deviation $> 0.01$.
- **V5 Failure:** Exactly 63 of 200 documents achieve correct Top-1 predictions, identically matching the frequency prior (31.50%).
- **Collapse to Single Prior Label:** Predictions concentrate on `cond-mat.dis-nn` or its identical ground-truth duplicate `cmp-lg` for 199 to 200 of 200 documents.
- **Counterfactual Failure:** Subtracting the static label-SA offsets at inference time (`DR4_FROZEN_SA_CF_20261012_A`) resulted in **0 recovered documents** ($\text{False} \to \text{True} = 0$).

This forensic audit traced the entire pipeline from input text to loss gradients to answer the fundamental question:  
**Why did standard BERT (DR2 / Variant B) learn strong discriminative rankings on AAPD (P@1 = 88.20%, F1 = 75.57%), while CSAM completely failed to exploit document-dependent information?**

### The Core Forensic Findings:
1. **Structural Information Discard (The "Alpha-Only" Bottleneck):** In the paper-derived literal implementation (Variant L/L2), the 768-dimensional document embedding $h_T$ is **completely discarded** after computing dot-product similarities. The document influences the final representation $h_{\text{final}}$ exclusively through a single scalar attention weight $\alpha_i$. In contrast, the authors' original repository (`Model/lhrm.py`) explicitly preserved the document embedding by concatenating it: $\text{emb} = \operatorname{concat}([h_T, \text{agg}]) \in \mathbb{R}^{2H}$ before linear classification into 54 labels.
2. **Attention Dilution & Gradient Attenuation:** Normalizing $\alpha = \operatorname{softmax}(h_T \cdot h_L / \sqrt{H})$ over 54 classes forces the average attention weight to $1/54 \approx 0.0185$. Scaling by $1/\sqrt{768}$ flattens the distribution to near-uniform entropy ($3.95 / 3.989$). The gradient backpropagating into document BERT is multiplied by $\alpha_i / \sqrt{H} \approx 6.68 \times 10^{-4}$ ($1496\times$ attenuation), virtually freezing document feature learning during training.
3. **Severe Multi-Label Class Imbalance Inversion in Focal Loss:** The registered Focal Loss implementation applied $\alpha = 0.25$ to positives and $1 - \alpha = 0.75$ to negatives. In AAPD, where $95.5\%$ of labels are negative (zeros), this configuration downweights the already rare positive class by $3\times$ ($0.25 / 0.75$). When combined with the 20:1 negative-to-positive class ratio, negative loss gradients dominate positive gradients by **$64.5\times$**, heavily penalizing any document-dependent positive predictions and strongly rewarding a static negative frequency-prior bias. In contrast, the authors' repository and our successful DR2 control used standard unweighted `BCEWithLogitsLoss`.

---

## 2. Equation-to-Code Implementation Forensic Audit

We audited the real source implementation in `src/lhrm_baseline/` against the paper equations (Ayash et al., IEEE Access 2025, Eqs. 1–8 & 17) and cross-referenced with the authors' repository (`Model/lhrm.py`).

| Mathematical Operation / Paper Equation | Implementing Code Location | Expected Shape | Actual Shape | Status & Forensic Finding |
|---|---|---|---|---|
| **Eq. 1: Document Representation**<br>$h_T = \operatorname{BERT}_{\text{CLS}}(X)$ | `model.py:get_cls_embedding()`<br>`model.py:321` | $[B, H]$ ($H=768$) | $[B, 768]$ | **VERIFIED.** Uses `last_hidden_state[:, 0]` (Decision A1), not `pooler_output`. |
| **Eq. 2: Label Representation**<br>$h_{Li} = \operatorname{BERT}_{\text{CLS}}(Y_i)$ | `model.py:get_cls_embedding()`<br>`model.py:330` | $[k, H]$ ($k=54$) | $[54, 768]$ | **VERIFIED.** Shared BERT backbone (Decision A2). Evaluated with autograd enabled (A3). |
| **Eq. 4: Document-Label Similarity**<br>$S_i = h_T \cdot h_{Li}$ | `model.py:84` (L)<br>`model.py:88` (L2/A5B) | $[B, k]$ | $[B, 54]$ | **ARCHITECTURAL DIVERGENCE.** Paper Eq. 4 is unscaled dot-product. L2 added $1/\sqrt{H}$ scaling (Decision A6) to prevent one-hot attention collapse. |
| **Eq. 5: Cross-Attention Weights**<br>$\alpha_i = \operatorname{softmax}_i(S_i)$ | `model.py:91` (CSAM_L)<br>`model.py:188` (A5B) | $[B, k]$ | $[B, 54]$ | **VERIFIED CODE, CRITICAL BOTTLENECK.** Normalizes across 54 labels ($\sum_i \alpha_i = 1$). Mean is $1/54 \approx 0.0185$. |
| **Eq. 6: Label Self-Attention**<br>$A_{ij} = \operatorname{softmax}_j\left(\frac{h_{Li} h_{Lj}^T}{\sqrt{d}}\right)$ | `model.py:95-96`<br>`model.py:191-192` | $[k, k]$ | $[54, 54]$ | **VERIFIED.** Rows sum to 1. Document-invariant by construction ($\operatorname{Var}_b(A) \equiv 0$). |
| **Eq. 7: Label Refinement**<br>$h_{\text{final}}^{(i)} = \alpha_i h_{Li} + \sum_j A_{ij} h_{Lj}$ | `model.py:98-107`<br>`model.py:194-200` | $[B, k, H]$ | $[B, 54, 768]$ | **CRITICAL ARCHITECTURAL BOTTLENECK.** Direct document representation $h_T$ is **discarded**. Document enters only via scalar $\alpha_i$. Self-attention term is at full norm $\approx 27.7$, whereas $\alpha_i h_{Li}$ has norm $\approx 0.51$ ($50\times$ smaller). |
| **Eq. 8: Classifier Head (Variant L/L2)**<br>$\hat{y}_i = W \cdot h_{\text{final}}^{(i)} + b$ | `model.py:58, 114`<br>`self.W = nn.Linear(H, 1)` | $[B, k]$ | $[B, 54]$ | **VERIFIED LITERAL READING (A5).** Uses single shared weight vector $W \in \mathbb{R}^{1 \times H}$ across all labels. Prevents label-specific readout projection. |
| **Eq. 8: Alternative Classifier (L2_A5B)**<br>$\hat{y}_i = W_i \cdot h_{\text{final}}^{(i)} + b$ | `model.py:162, 205`<br>`self.W = Parameter(k, H)` | $[B, k]$ | $[B, 54]$ | **VERIFIED ALTERNATIVE (A5b).** Unshares weight row per label. Preserves shared scalar bias $b \in \mathbb{R}$. Tested in DR4 (failed). |
| **Author Repo Architecture (Variant R)**<br>$\operatorname{concat}([h_T, \operatorname{agg}]) \to \operatorname{Linear}(2H, k)$ | `Model/lhrm.py:189-192`<br>`model.py:248-253` | $[B, 2H] \to [B, k]$ | $[B, 1536] \to [B, 54]$ | **CRITICAL STRUCTURAL DIVERGENCE.** Author's own repository retained $h_T$ directly in the classifier! Rejected as "Variant R fallback" in Phase 1 without testing. |
| **Eq. 17: Multi-Label Objective**<br>$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$ | `losses.py:66-72`<br>`focal_alpha=0.25`, $\gamma=2$ | Scalar loss | Scalar loss | **CRITICAL OPTIMIZATION BOTTLENECK.** Implementation downweights positive classes by $3\times$ ($0.25$ vs $0.75$). In 95.5% sparse AAPD, negative gradients overwhelm positives by $64.5\times$. Author repo used standard unweighted BCE! |

---

## 3. Deep-Dive: Document-Dependent Signal Audit

### 3.1 Trace of Document Information Through CSAM
Consider raw document text $X$:
1. $X$ is encoded by BERT into contextual sequence embeddings. The $[CLS]$ token is extracted as $h_T \in \mathbb{R}^{768}$.
2. $h_T$ interacts with 54 label embeddings $h_L \in \mathbb{R}^{54 \times 768}$ solely through the dot-product:
   $$S[b, i] = \frac{h_T[b] \cdot h_L[i]}{\sqrt{768}}$$
3. Softmax normalizes $S$ into attention distribution $\alpha[b, :] \in \Delta^{53}$:
   $$\sum_{i=1}^{54} \alpha[b, i] = 1, \quad \alpha[b, i] \ge 0$$
4. In Eq. 7, the refined representation for label $i$ is:
   $$h_{\text{final}}[b, i] = \alpha[b, i] h_L[i] + \text{SA}[i]$$
   where $\text{SA}[i] = \sum_j A_{ij} h_L[j]$ is **independent of document $b$**.
5. The raw logit is computed by projecting $h_{\text{final}}$ with classifier weight $W_i$:
   $$\text{logits}[b, i] = W_i \cdot h_{\text{final}}[b, i] + b = \underbrace{\alpha[b, i] (W_i \cdot h_L[i])}_{\text{doc\_term}[b, i]} + \underbrace{W_i \cdot \text{SA}[i]}_{\text{sa\_term}[i]} + b$$

### 3.2 The Fundamental Signal Vanishing Mechanism
Notice the profound structural implications of this equation:
- **The document embedding $h_T$ never enters the logit computation as a feature vector.** All 768 semantic dimensions of the document are collapsed into a single probability scalar $\alpha[b, i]$.
- **Zero-Sum Multi-Label Competition:** Because $\sum_{i=1}^{54} \alpha[b, i] = 1$, the expected value of $\alpha[b, i]$ across labels is strictly $1/54 \approx 0.0185$.
- **Attention Flattening Under $1/\sqrt{H}$:** Scaling dot products by $\sqrt{768} \approx 27.71$ compresses raw dot-product spreads from $[-20, 20]$ down to $[-0.7, 0.7]$. Taking softmax over 54 items with exponent range $\approx 1.4$ produces an almost completely uniform distribution:
  - Measured attention entropy in DR4: **$3.69$ (L2)** and **$3.95$ (L2_A5B)** against theoretical maximum $\ln(54) \approx 3.989$ ($98.9\%$ uniform).
  - Cross-document standard deviation of $\alpha[b, i]$: only $\sigma_\alpha \approx 0.003$ to $0.005$.
- **The Double Static Bias Trap:**
  Decomposing $\alpha[b, i] = \bar{\alpha}_i + \epsilon[b, i]$ (where $\bar{\alpha}_i = \mathbb{E}_b[\alpha[b, i]] \approx 0.0185$ and $\mathbb{E}_b[\epsilon] = 0$):
  $$\text{logits}[b, i] = \underbrace{\epsilon[b, i] (W_i \cdot h_L[i])}_{\text{dynamic document perturbation}} + \underbrace{\bar{\alpha}_i (W_i \cdot h_L[i])}_{\text{static mean doc offset}} + \underbrace{W_i \cdot \text{SA}[i]}_{\text{static SA offset}} + b$$
  Look at the empirical magnitudes measured on the trained checkpoints:
  - Dynamic document perturbation ($\operatorname{std}_b$ across documents): **$0.0003$ to $0.005$ logits**.
  - Static mean document offset spread ($\bar{\alpha}_i (W_i \cdot h_L[i])$): **$0.086$ logits** ($17\times$ to $280\times$ larger).
  - Static self-attention offset spread ($W_i \cdot \text{SA}[i]$): **$1.62$ to $1.67$ logits** ($300\times$ to $5000\times$ larger).

**Conclusion:** Even when $\text{sa\_term}[i]$ is subtracted, the static mean document offset $\bar{\alpha}_i (W_i \cdot h_L[i])$ is still an order of magnitude larger than the dynamic perturbation $\epsilon[b, i] (W_i \cdot h_L[i])$. Therefore, static label ordering dominates the ranking completely before and after SA subtraction!

---

## 4. Deep-Dive: Gradient Flow & Trainability Audit

We derived the analytical backward pass to determine how supervisory signals flow from the loss back into the document BERT encoder:

### 4.1 Analytical Gradient to Document Encoder
The loss gradient with respect to document representation $h_T$ is:
$$\frac{\partial \mathcal{L}}{\partial h_T} = \sum_{i=1}^{54} \frac{\partial \mathcal{L}}{\partial \text{logit}_i} \frac{\partial \text{logit}_i}{\partial h_T}$$
Using $\text{logit}_i = \alpha_i (W_i \cdot h_L[i]) + \text{sa\_term}_i + b$:
$$\frac{\partial \text{logit}_i}{\partial \alpha_i} = W_i \cdot h_L[i]$$
From the Jacobian of the scaled softmax $\alpha_i = \operatorname{softmax}\left(\frac{h_T \cdot h_L}{\sqrt{H}}\right)$:
$$\frac{\partial \alpha_i}{\partial h_T} = \frac{\alpha_i}{\sqrt{H}} \left( h_L[i] - \sum_{j=1}^{54} \alpha_j h_L[j] \right)$$
Combining terms:
$$\frac{\partial \text{logit}_i}{\partial h_T} = \frac{\alpha_i (W_i \cdot h_L[i])}{\sqrt{H}} \left( h_L[i] - \sum_{j=1}^{54} \alpha_j h_L[j] \right)$$

### 4.2 The $1496\times$ Gradient Attenuation Bottleneck
Look at the pre-factor scaling document gradients:
$$\text{Factor} = \frac{\alpha_i}{\sqrt{H}} \approx \frac{1/54}{\sqrt{768}} = \frac{1}{54 \times 27.7128} \approx \frac{1}{1496.5} \approx 6.68 \times 10^{-4}$$
- When a misclassification occurs on label $i$, the gradient flowing into classifier weight $W_i$ is proportional to $h_{\text{final}} \approx \text{SA}[i]$, with vector norm $\approx 27.7$.
- In contrast, the gradient flowing into document BERT $h_T$ is scaled by $\frac{\alpha_i}{\sqrt{H}} \approx 6.68 \times 10^{-4}$.
- **Gradient Magnitude Ratio:**
  $$\frac{\|\nabla_{W_i} \mathcal{L}\|}{\|\nabla_{h_T} \mathcal{L}\|} \sim \frac{27.7}{6.68 \times 10^{-4} \times 27.7} \approx 1496\times$$
- During AdamW optimization (learning rate $2 \times 10^{-5}$):
  The classifier weights $W$ and static offsets update quickly to minimize the global loss, while document BERT receives virtually negligible gradient feedback to adjust its representation space to discriminate labels.

---

## 5. Deep-Dive: Loss Function & Class Imbalance Audit

### 5.1 The Positive-Class Penalty in Focal Loss
In `src/lhrm_baseline/losses.py`:
```python
alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
focal_weight = (1 - p_t).pow(self.gamma)
loss = alpha_t * focal_weight * ce
```
With registered hyperparameters $\alpha = 0.25, \gamma = 2.0$:
- For positive ground-truth targets ($y = 1$): $\alpha_t = 0.25$.
- For negative ground-truth targets ($y = 0$): $\alpha_t = 1 - 0.25 = 0.75$.

### 5.2 Mathematical Analysis on AAPD Class Sparsity
In AAPD (54 labels, mean $\sim 2.4$ positive labels per document):
- Positive fraction: $2.4 / 54 \approx 4.44\%$.
- Negative fraction: $51.6 / 54 \approx 95.56\%$.
- Relative weighting of individual examples: Positives are weighted at **$0.25$**, while negatives are weighted at **$0.75$** ($3\times$ higher).
- Aggregate batch loss weight ratio:
  $$\text{Negative Gradient Mass} = 51.6 \times 0.75 = 38.70$$
  $$\text{Positive Gradient Mass} = 2.4 \times 0.25 = 0.60$$
  $$\text{Ratio} = \frac{38.70}{0.60} = \mathbf{64.5\times}$$

### 5.3 The Consequence on Optimization Strategy
An optimizer minimizing this loss discovers an immediate, trivial local optimum:
- Predict negative logits ($\text{logit}_i < 0 \implies p_i \to 0$) for all labels.
- For negative examples ($y=0$), $p_t = 1 - p \to 1 \implies (1 - p_t)^2 \to 0$. The massive $64.5\times$ negative loss vanishes!
- For the rare positive labels ($y=1$), the penalty is small because $\alpha_t$ is only $0.25$.
- This explains why training loss decreased steadily in all runs (passing gate V2), while the model converged to predicting negative logits with static frequency-prior ordering!

In contrast:
- **DR2 Control Baseline:** Used standard `BCEWithLogitsLoss()`. In BCE, positive and negative errors receive equal unit weight ($1.0$ vs $1.0$).
- **Author's Codebase (`Model/lhrm.py`):** Explicitly used `nn.BCEWithLogitsLoss()`, not Focal Loss!

---

## 6. Quantitative Root-Cause Hypothesis Matrix

We systematically evaluated seven candidate root-cause hypotheses against all empirical evidence from `RUN_0`, `DR1`, `DR2`, `DR3/DR3b`, `DR4`, and `DR4_FROZEN_SA_CF`:

| Hypothesis | Description | Supporting Evidence | Contradicting Evidence | Category | Confidence | Decisive Test |
|---|---|---|---|---|---|---|
| **H1: Static label-SA offsets dominate score scale** | Additive SA offset $W \cdot \text{SA}[i]$ drowns out document-dependent variations. | Decomposition showed SA magnitude is $50\times$ larger than document terms ($1.65$ vs $0.033$). | **Refuted by Counterfactual:** Subtracting SA at inference produced 0 recovered hits (63/200 unchanged). Document term itself is collapsed. | Symptom / Amplifier | **LOW** (as root cause) | Completed in `DR4_FROZEN_SA_CF`. |
| **H2: Cross-attention normalization dilutes document signal** | Softmax over 54 labels combined with $1/\sqrt{H}$ scaling attenuates document dynamic range to $\pm 0.003$ and gradients by $1496\times$. | Attention entropy $3.95/3.989$ ($98.9\%$ uniform). Derived gradient factor $\alpha_i/\sqrt{H} \approx 6.7 \times 10^{-4}$ matches DR3 micro-probe measurement. | None. Pure mathematical and empirical consistency across all runs. | Architectural Defect | **HIGH** | Compare gradients with/without unprojected softmax. |
| **H3: Shared/aligned classifier directions prevent discrimination** | Shared vector $W \in \mathbb{R}^{1 \times H}$ (L2) or collinear rows (A5b) enforce identical projection across labels. | L2 shared head mathematically forces identical readout. A5b classifier rows had $0.9985$ cosine similarity. | Unsharing $W$ in A5b did not improve results (V4 dropped from 21 to 3; P@1 remained 63/200). Collinearity is a symptom of diffuse input. | Architectural Constraint | **MEDIUM** | Evaluate classifier with non-diffuse inputs. |
| **H4: Focal loss configuration forces frequency prior** | $\alpha=0.25$ downweights rare positives by $3\times$, giving $64.5\times$ gradient dominance to negatives in 95.5% sparse AAPD. | Negative logits learned ($b = -0.023$). Steady loss decrease (V2 PASS) despite ranking collapse. DR2 succeeded with standard BCE ($88.2\%$). Author code used BCE. | DR3 probe showed BCE also has gradient attenuation if attention is diffuse, but BCE provides $3\times$ higher positive gradient. | Optimization Impediment | **HIGH** | Retrain/probe with standard BCE vs Focal Loss. |
| **H5: Document representation discarded (Missing Direct Feature Path)** | Literal Variant L/L2 discards $h_T$, relying entirely on scalar $\alpha_i$. Author's repo concatenated $h_T$: $\operatorname{concat}([h_T, \operatorname{agg}])$. | Literal CSAM discards $h_T$. DR2 (direct $h_T \to \text{Linear}$) succeeded (88.2%). Author repo (`Model/lhrm.py:190`) explicitly concatenated $[h_T, \operatorname{agg}]$. | None. The literal reading removed the primary discriminative feature path present in standard multi-label models. | Structural / Implementation Divergence | **HIGH** | Evaluate representation with direct $h_T$ concatenation. |
| **H6: Multi-label objective vs single-label attention mismatch** | Softmax attention enforces zero-sum competition across labels ($\sum \alpha_i = 1$), whereas multi-label classification requires independent binary detection. | Documents have multiple positive labels (mean 2.4). Softmax cannot activate multiple labels simultaneously without reducing per-label magnitude. | Standard attention can sometimes act as a soft selector, but zero-sum property penalizes documents with many active labels. | Formulation Mismatch | **HIGH** | Compare softmax vs sigmoid cross-attention. |
| **H7: Duplicate-target label collinearity** | `cmp-lg` and `cond-mat.dis-nn` have identical columns in AAPD, causing ranking instability. | Explains why L2 preferred `cond-mat` and A5b preferred `cmp-lg`, and why counterfactual flipped 199/200 docs. | Only explains the choice between indices 0 and 1; does not explain why the other 52 labels never rank #1. | Dataset Artifact | **LOW** (as system-wide cause) | Remove duplicate targets from metric computation. |

---

## 7. Synthesis: The Triad of Failure in Literal CSAM

The failure of literal CSAM is not due to any single isolated bug, but an unfortunate compounding **triad of structural and optimization bottlenecks**:

```
[Document Text X] 
       │
       ▼ (BERT)
    [ h_T ] ────┐ (768-dim semantic document representation)
       │        │
       │        ▼ (DISCARDED in Variant L/L2! Never reaches classifier!)
       │
       ▼ (Dot product with h_L / sqrt(768))
       S  (values compressed into [-0.7, 0.7])
       │
       ▼ (Softmax over 54 classes: Bottlenecks H2 & H6)
     alpha (near-uniform ~ 0.0185; dynamic range +-0.003; gradients attenuated 1496x)
       │
       ▼ (Weighted label embedding + static SA: Bottlenecks H1 & H5)
    h_final = alpha * h_L + SA  (SA is 50x larger than alpha * h_L; static doc term 17x larger)
       │
       ▼ (Linear projection)
     logits
       │
       ▼ (Focal Loss alpha=0.25: Bottleneck H4)
   Loss Gradients (95.5% negative labels dominate gradients by 64.5x -> predicts negative prior)
```

1. **Structural:** The literal implementation discarded the direct document representation $h_T$, reducing the document to a scalar attention weight $\alpha_i$ (unlike the author's working repository which concatenated $[h_T, \operatorname{agg}]$).
2. **Dynamical:** The $1/\sqrt{H}$ scaling flattened cross-attention across 54 labels to near-uniform entropy, reducing document-dependent logit variance to $\sim 0.005$ and attenuating backward gradients to document BERT by $1496\times$.
3. **Loss:** Focal Loss ($\alpha=0.25$) on sparse multi-label data downweighted positive classes by $3\times$, directing $64.5\times$ of optimization mass toward predicting static negative logits matching the frequency prior.
