# LHRM — DR4 Preregistration: Controlled Screening of Variant L2_A5B
## Investigation of Unshared Classifier Head Parameterization under Locked Experimental Protocol

**Document Reference:** `DR4_A5B_PREREGISTRATION.md`  
**Experiment ID:** `RUN_0_L2_A5B_20261012_A`  
**Baseline Variant:** `L2` (Scaled dot-product cross-attention, label-label self-attention enabled, shared classifier $W \in \mathbb{R}^{1 \times H}$, shared scalar bias $b \in \mathbb{R}$)  
**Candidate Variant:** `L2_A5B` (Identical to L2 except unshared classifier weight matrix $W \in \mathbb{R}^{k \times H}$, shared scalar bias $b \in \mathbb{R}$)  
**Date of Preregistration:** 2026-10-12  
**Auditor Roles:** Senior ML Research Scientist, PyTorch Architecture Engineer, Reproducibility Auditor, Experimental Design Specialist  
**Execution Status:** **PREPARED ONLY — GPU TRAINING NOT LAUNCHED (REQUIRES USER AUTHORIZATION)**  

---

## 1. Context, Motivation & Scientific Status

### 1.1 Context
In `RUN_0_L2_NO_SA_ABLATION_20261011_A` and its confirmed replay `RUN_0_L2_NO_SA_REPLAY_20261011_B`:
- Both `L2` and `L2_NO_SA` failed viability criteria V4 and V5, yielding exactly 63/200 Top-1 hits matching the dataset frequency prior.
- Removing label-label self-attention in `L2_NO_SA` failed to recover document-sensitive ranking; attention remained diffuse (entropy ratio 98.9%), logit standard deviation collapsed to 0/54 passing labels, and all 200 validation documents predicted `cond-mat.dis-nn`.
- Post-training gradient probes on authentic trained checkpoints confirmed severe document gradient attenuation ($\|\nabla_{h_T}\| \approx 10^{-5}$ vs $\|\nabla_W\| \approx 0.04 - 1.92$).
- Under diffuse attention $\alpha_{b, i} \approx \bar{\alpha}$, the shared readout vector $W \in \mathbb{R}^{1 \times H}$ projects representations into document-invariant scalar rankings:
  $$\text{logit}_{b, i} \approx \bar{\alpha} (W h_{L_i}) + W (A h_L)_i + b$$
  Because $W$ is shared across all labels, $(W h_{L_i})$ and $W (A h_L)_i$ are static label constants, permanently locking the relative ordering.

### 1.2 Scientific Status of Candidate `L2_A5B`
- `L2_A5B` is an **alternative architectural interpretation** of Figure 2 (Decision A5b), **NOT** a confirmed paper-faithful reproduction.
- Paper Eq. 8 explicitly writes $y_i = W h_{\text{final}}^{(i)} + b$ with scalar output, implying shared $W \in \mathbb{R}^{1 \times H}$. Figure 2 can be alternatively interpreted as an unshared linear classification layer where each label has its own dedicated projection vector $w_i \in \mathbb{R}^H$.
- Preregistering `L2_A5B` isolates whether parameter unsharing alone resolves the ranking collapse under the 300-step screening protocol.

---

## 2. Strict Single-Component Isolation Rules

To maintain strict scientific control, `L2_A5B` differs from `L2` by **EXACTLY ONE** component: the classifier weight parameterization.

### 2.1 The Single Modification
- **Classifier Weight Parameterization:**
  - `L2`: Shared vector $W \in \mathbb{R}^{1 \times H}$ applied across all labels via `nn.Linear(H, 1)`.
  - `L2_A5B`: Unshared matrix $W \in \mathbb{R}^{k \times H}$ where row $i$ ($w_i \in \mathbb{R}^H$) classifies label $i$.
  - Shared scalar bias $b \in \mathbb{R}$ is **strictly retained** in both variants. A per-label bias vector is an additional degrees-of-freedom change and is explicitly forbidden for this test.

### 2.2 Prohibited Confounding Changes
The following modifications are strictly **FORBIDDEN** in this experiment:
1. **DO NOT** remove label-label self-attention (Eq. 6). Label self-attention remains fully **ENABLED**.
2. **DO NOT** change cross-attention temperature scaling ($1/\sqrt{H}$ retained).
3. **DO NOT** introduce query/key projection layers ($W_Q, W_K$).
4. **DO NOT** alter encoder architecture (shared `bert-base-uncased`, revision `86b5e0934494bd15c9632b12f734a8a67f723594`).
5. **DO NOT** replace Focal Loss ($\alpha=0.25, \gamma=2.0$).
6. **DO NOT** change training duration (strictly 300 optimizer steps).
7. **DO NOT** change optimizer hyperparameters (AdamW, weight decay 0.01, seed 42, batch size 8).
8. **DO NOT** change learning rate: **The approved AAPD baseline learning rate is strictly $2\times 10^{-5}$ (NOT $3\times 10^{-5}$)**.
9. **DO NOT** modify dataset splits, label order, label definitions, or tokenization (max length 512 for docs, 64 for labels).
10. **DO NOT** touch the Test split (quarantined under Decision A1).

---

## 3. Mathematical Specification of A5b Head

Given:
- Document CLS embedding: $h_T \in \mathbb{R}^{B \times H}$
- Label CLS embeddings: $h_L \in \mathbb{R}^{k \times H}$ ($k=54, H=768$)
- Scaled similarity: $S_{b, i} = \frac{h_{T, b} \cdot h_{L_i}}{\sqrt{H}}$
- Document attention: $\alpha_{b, i} = \frac{\exp(S_{b, i})}{\sum_{j=1}^k \exp(S_{b, j})}$
- Label self-attention: $A_{i, j} = \frac{\exp(h_{L_i} \cdot h_{L_j} / \sqrt{H})}{\sum_{m=1}^k \exp(h_{L_i} \cdot h_{L_m} / \sqrt{H})}$
- Refined label representation (Eq. 7):
  $$h_{\text{final}, b, i} = \alpha_{b, i} h_{L_i} + \sum_{j=1}^k A_{i, j} h_{L_j} \in \mathbb{R}^H$$
  Tensor shape: $[B, k, H]$.

### 3.1 Unshared Readout Logit Equation
For document $b \in \{1, \dots, B\}$ and label $i \in \{1, \dots, k\}$:
$$\text{logits}[b, i] = w_i \cdot h_{\text{final}, b, i} + b$$
where $w_i = W[i, :] \in \mathbb{R}^H$ is the dedicated weight vector for label $i$, and $b \in \mathbb{R}$ is the shared scalar bias.

### 3.2 PyTorch Tensor Implementation
```python
# h_final: [B, k, H]
# self.W: [k, H] -> unsqueeze(0) -> [1, k, H]
# self.bias: [1]
logits = (h_final * self.W.unsqueeze(0)).sum(dim=-1) + self.bias  # [B, k]
```
- **CRITICAL ANTI-PATTERN AVOIDED:** Applying `nn.Linear(H, k)` to `h_final` $[B, k, H]$ produces $[B, k, k]$. That is NOT the intended architecture. The explicit elementwise broadcast-and-sum computes exactly the per-label dot product $[B, 54]$.

---

## 4. Controlled Initialization Protocol & Step-0 Parity

To isolate the parameter-sharing hypothesis from random initialization variance:

1. **Backbone Cloning:** Both `L2` and `L2_A5B` instantiate the exact same BERT backbone weights.
2. **Classifier Weight Cloning:**
   The single shared weight vector of `L2` ($W_{\text{shared}} \in \mathbb{R}^{1 \times H}$) is broadcast into all $k=54$ rows of `L2_A5B`:
   $$W_{\text{A5B}}[i, :] = W_{\text{shared}}[0, :], \quad \forall i \in \{1, \dots, 54\}$$
3. **Classifier Bias Cloning:**
   The shared scalar bias of `L2` ($b_{\text{shared}} \in \mathbb{R}$) is copied to `L2_A5B`:
   $$b_{\text{A5B}} = b_{\text{shared}}$$
4. **Step-0 Numerical Verification Gates:**
   - **Logit Agreement:** $\max_{b, i} |\text{logits}_{L2}[b, i] - \text{logits}_{L2\_A5B}[b, i]| < 10^{-6}$.
   - **Attention Agreement:** $\max_{b, i} |\alpha_{L2}[b, i] - \alpha_{L2\_A5B}[b, i]| < 10^{-6}$.
   - **Loss Agreement:** $|\mathcal{L}_{L2} - \mathcal{L}_{L2\_A5B}| < 10^{-6}$.
   - **Independent Gradient Evolution:** For $t > 0$, because each row $W[i, :]$ receives gradients only from label $i$'s loss contribution:
     $$\nabla_{w_i} \mathcal{L} = \sum_{b=1}^B \frac{\partial \mathcal{L}}{\partial \text{logits}[b, i]} h_{\text{final}, b, i}$$
     The rows $w_i$ can freely diverge during optimization, breaking the shared projection constraint.

---

## 5. Viability Evaluation & Decision Gates

Evaluation retains the established V1–V5 criteria applied to the frozen 200-document validation sample:

### 5.1 Primary Viability Gates

| Criterion | Metric Definition | Success Threshold |
| :--- | :--- | :---: |
| **V1 (Numerical Stability)** | Non-finite losses or gradients | 0 NaN / Inf over 300 steps |
| **V2 (Loss Minimization)** | Mean loss trajectory | Mean of last 50 steps < mean of first 50 steps |
| **V3 (Attention Dispersion)** | Softmax saturation fraction | $\alpha_{\text{one-hot}} < 0.90$ |
| **V4 (Logit Variation)** | Cross-document logit standard deviation | $\ge 28 / 54$ labels with std $> 0.01$ |
| **V5 (Ranking Superiority)** | Top-1 correct prediction hits on 200 Dev sample | Strictly $> 63$ hits ($> 31.50\%$) |

### 5.2 Exploratory Ranking & Diagnostic Measurements
In addition to V1–V5, the screening will record:
1. **Distinct Top-1 Predictions:** Number of unique labels assigned rank 1 across the 200 documents (baseline was 1–2).
2. **Distinct Top-3 Sequences:** Number of unique ordered triplets in Top-3 predictions.
3. **Dominant Label Frequency:** Percentage of documents assigned to the most frequent Top-1 label.
4. **Attention Entropy & Mean Max Attention:** Quantifying cross-attention sharpess vs uniformity.
5. **Per-Label Classifier Gradient Norms:** Auditing whether all 54 rows of $W$ receive active gradient updates.
6. **Collinear Label Prior Analysis:** Tracking predictions for `cmp-lg` vs `cond-mat.dis-nn`.

---

## 6. Scientific Interpretation Boundaries & Epistemological Limits

1. **Ranking Diversity Is Not Reproduction:**
   Observing multiple distinct Top-1 predictions or passing V4 does not prove the model is learning correct semantic ranking. Correctness is measured strictly by P@1, nDCG, and F1.
2. **Passing V4 & V5 Is Preliminary:**
   If `L2_A5B` passes V1–V5, this justifies proceeding to full training and Dev evaluation. It does not establish successful reproduction of paper metrics.
3. **Failure Bounds:**
   If `L2_A5B` fails V4 or V5 (collapsing to $\le 63$ hits), this demonstrates that **unsharing the classifier head alone is insufficient** to resolve the ranking collapse under 300-step screening. It **does NOT prove** that document-label dot-product cross-attention is the sole remaining failure source, as other unablated factors (e.g. attention temperature scaling, lack of query/key projections, optimization timescale, learning rate) could also contribute.

---

## 7. Audit & Provenance Manifest

- **Pretrained BERT Source:** Pinned official revision `google-bert/bert-base-uncased` (`86b5e0934494bd15c9632b12f734a8a67f723594`).
- **Dataset:** AAPD Train (53,840), Dev (1,000), 54 labels.
- **Frozen Val Sample SHA-256:** Derived from `experiments/RUN_0/run0_val_sample_ids.json`.
- **Target Experiment ID:** `RUN_0_L2_A5B_20261012_A`.
- **Output Safety Guard:** Fail-closed if target directory exists and is non-empty. Historical run IDs (`RUN_0`, `DR1`, `DR2`, `RUN_0_P1F_EXT1`, `RUN_0_L2_NO_SA_ABLATION_20261011_A`, `RUN_0_L2_NO_SA_REPLAY_20261011_B`) are permanently write-protected.
- **Quarantine:** Test split is strictly excluded from all training, validation, and screening commands.
