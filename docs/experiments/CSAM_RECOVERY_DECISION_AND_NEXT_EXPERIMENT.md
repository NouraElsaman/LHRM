# CSAM Recovery Decision and Next Scientific Experiment

**Document ID:** `CSAM_RECOVERY_DECISION_AND_NEXT_EXPERIMENT_20261012`  
**Date:** 2026-10-12  
**Status:** PREREGISTERED PROPOSAL (ZERO GPU EXECUTION WITHOUT USER APPROVAL)  
**Parent Forensic Audit:** `docs/experiments/CSAM_DOCUMENT_SIGNAL_ROOT_CAUSE_AUDIT.md`  

---

## 1. Diagnosis: Why Literal CSAM Failed

Based on the forensic audit of the source implementation (`src/lhrm_baseline/`), comparison with the authors' repository (`Model/lhrm.py`), and empirical evidence across four screening runs and two post-training diagnostics, the root failure of literal CSAM is attributed to the compounding effect of three specific mechanisms:

1. **The "Alpha-Only" Structural Discard (H5 & H2):** In literal Variant L/L2 (Equations 7–8), the 768-dimensional document embedding $h_T$ is completely discarded from the classification input. The document is compressed into a single scalar attention weight $\alpha_i$ per label. In contrast, standard multi-label models (including our successful DR2 control baseline, Dev P@1 = 88.20%) and the authors' original repository (`Model/lhrm.py`) directly feed the document embedding $h_T$ into the classification head via concatenation ($\operatorname{concat}([h_T, \text{agg}]) \in \mathbb{R}^{2H}$).
2. **Attention Dilution & Gradient Starvation (H2):** Under $1/\sqrt{H}$ scaling, the cross-attention distribution $\alpha$ across 54 labels collapses into near-uniform noise ($98.9\%$ entropy, dynamic range across documents $\pm 0.003$). This attenuates backward gradients to the document BERT encoder by a factor of $\frac{\alpha_i}{\sqrt{H}} \approx 6.68 \times 10^{-4}$ ($1496\times$ suppression), leaving document BERT effectively untrained.
3. **Class-Imbalance Inversion in Focal Loss (H4):** Focal Loss with $\alpha=0.25$ downweights rare positive labels by $3\times$ ($0.25$ vs $0.75$). In AAPD, where $95.5\%$ of labels are zeros, the background negative class exerts $64.5\times$ more gradient mass than positive targets. The optimizer easily minimizes loss by learning static negative logits that predict the dataset frequency prior (63/200 hits).

---

## 2. Smallest Proposed Next Intervention

We recommend **EXACTLY ONE** immediate next research action:

### Recommendation: Category 2 — Decisive Frozen-Model Diagnostic Before Editing Architecture
**Experiment Name:** DR4 Frozen-Backbone Document Feature Alignment Probe  
**Diagnostic ID:** `DR4_FROZEN_BACKBONE_PROBE_20261012_A`  

### Why This Action is Decisive and More Informative Than Another Training Ablation
Before authorizing any new GPU training run (such as testing BCE or Variant R), we must empirically resolve the critical open question identified in the forensic audit:
> **Did the BERT backbone in `L2_STEP300` retain its document-discriminative capacity, or did 300 steps of CSAM training degrade its representations?**

- If document representations $h_T$ extracted from `L2_STEP300_MODEL_ONLY.pt` maintain high separability for AAPD labels (comparable to or better than pre-trained BERT), it proves conclusively that BERT was not damaged, and the failure is **100% localized to the downstream CSAM aggregation and classification head**.
- If, conversely, $h_T$ from `L2_STEP300` has lower separability than step-0 BERT, it proves that the $64.5\times$ negative gradient pressure from Focal Loss actively corrupted the backbone weights.

This diagnostic is:
- **Inference-only on already existing checkpoints** (`L2_STEP300_MODEL_ONLY.pt`, `L2_A5B_STEP300_MODEL_ONLY.pt`, and initial BERT).
- **Fast and deterministic.**
- **A strict falsification gate:** Resolves whether a recovery architecture can reuse the trained backbone or must start from clean pre-trained BERT.

---

## 3. Preregistered Protocol for Next Experiment (`DR4_FROZEN_BACKBONE_PROBE_20261012_A`)

### 3.1 Experimental Procedure
1. Load three frozen backbones on the 200 Dev screening documents:
   - Backbone 0: Initial clean pre-trained `bert-base-uncased` (revision `86b5e0934494bd15c9632b12f734a8a67f723594`).
   - Backbone 1: Trained Step-300 BERT from `L2_STEP300_MODEL_ONLY.pt`.
   - Backbone 2: Trained Step-300 BERT from `L2_A5B_STEP300_MODEL_ONLY.pt`.
2. Extract document $[CLS]$ embeddings $h_T \in \mathbb{R}^{200 \times 768}$ for all three backbones.
3. Compute document feature geometry:
   - Mean pairwise cosine similarity across documents (measuring document representation collapse/spread).
   - Document-to-true-label semantic alignment: Cosine similarity between $h_T[b]$ and label embeddings $h_L[i]$ for positive vs negative labels.
4. Evaluate a zero-training nearest-centroid classifier:
   - Compute label centroids $C_i = \frac{1}{|D_i|} \sum_{b \in D_i} h_T[b]$ using the Train split documents.
   - Predict Top-1 label as $\operatorname{argmax}_i \cos(h_T[b], C_i)$ on the 200 Dev documents.
   - Measure P@1 against the 63/200 frequency prior.

### 3.2 Expected Observable Effect
- If document features were preserved:
  - $\cos(h_T[b], h_L[i])$ for true positive labels will be significantly higher than for negative labels ($p < 0.01$).
  - Centroid classification P@1 will exceed the frequency prior ($> 63/200$).
- If document features were collapsed/corrupted:
  - True-positive vs negative label alignment margin will be near zero ($< 0.005$).
  - Nearest-centroid P@1 will collapse to the majority label (63/200).

### 3.3 Falsification Criteria
- **Falsification of "Corrupted Backbone" Hypothesis:** If Backbone 1 matches or exceeds Backbone 0 in semantic alignment margin, the hypothesis that BERT was damaged during CSAM training is falsified. Recovery efforts must focus exclusively on the downstream classifier architecture.
- **Falsification of "Untapped Document Signal in Checkpoint" Hypothesis:** If Backbone 1 exhibits collapsed document spread (pairwise cosine similarity $> 0.999$), the hypothesis that existing trained checkpoints contain usable document signals is falsified.

---

## 4. Downstream Recovery Architecture Proposal (Contingent on Diagnostic)

Once the backbone probe is completed, the project will evaluate the prioritized recovery intervention:

### Prioritized Recovery Architecture: Variant R (The Author's Genuine Repository Architecture)
- **Model Topology (`Model/lhrm.py:189-192`):**
  $$\text{agg} = \operatorname{softmax}\left(\frac{h_T \cdot h_L^T}{\sqrt{H}}\right) h_L \in \mathbb{R}^{B \times H}$$
  $$\text{emb} = \operatorname{concat}([h_T, \text{agg}]) \in \mathbb{R}^{B \times 2H}$$
  $$\text{logits} = \operatorname{Linear}(2H, k)(\text{emb}) \in \mathbb{R}^{B \times 54}$$
- **Loss Objective:** Standard `nn.BCEWithLogitsLoss()` (replacing Focal Loss with $\alpha=0.25$).
- **Rationale:**
  1. Restores the direct 768-dimensional document feature path to the classifier.
  2. Eliminates the multi-label class imbalance inversion penalty by weighting positive and negative errors equally.
  3. Reconnects our reproduction with the code the original authors actually implemented and evaluated in their repository.

---

## 5. Registered Viability Gates for Any Future Training Screen

If a training run is subsequently authorized by the user, it will be strictly governed by the registered V1–V5 viability gates:

| Gate | Criterion | Threshold | Scientific Rationale |
|---|---|---|---|
| **V1** | Numerical Stability | 0 NaN/Inf batches, $< 5$ AMP skips | Ensures training stability |
| **V2** | Decreasing Training Loss | $\text{Loss}_{\text{last50}} < \text{Loss}_{\text{first50}}$ | Verifies gradient optimization |
| **V3** | Non-Collapsed Attention | One-hot attention fraction $< 0.90$ | Prevents point saturation |
| **V4** | Cross-Document Signal | $\ge 28/54$ labels with $\operatorname{std}_b(\text{logit}) > 0.01$ | Confirms document-conditioned variation |
| **V5** | Ranking Superiority | Correct Top-1 hits $> 63/200$ (strict inequality) | **Must strictly beat frequency prior** |

All five gates are required for viability. Achieving 63/200 on V5 is a **FAIL**.

---

## 6. Resource Estimate & Scope

| Task Phase | Compute Platform | Estimated Runtime | Checkpoints / Data Needed |
|---|---|---|---|
| **Phase A: Frozen Backbone Probe** (`DR4_FROZEN_BACKBONE_PROBE`) | CPU (local) or lightweight Kaggle CPU | $< 3$ minutes on CPU | Existing Step-300 checkpoints + 200 Dev texts |
| **Phase B: Controlled Recovery Training Screen** (Variant R, 300 steps) | Kaggle GPU (T4 / P100) | $\approx 12$ minutes on GPU | AAPD Train/Dev + official BERT |

---

## 7. Decision Boundary

**No GPU execution or retraining will commence without explicit user authorization.**  
The current task stops at preparation and audit. The user may authorize executing the CPU backbone probe or proceeding directly to the recovery implementation.
