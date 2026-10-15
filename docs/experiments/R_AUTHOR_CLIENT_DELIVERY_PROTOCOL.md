# LHRM Author-Aligned Model Recovery Protocol (R_AUTHOR) & Client Delivery Plan

**Protocol Identifier**: `AAPD_R_AUTHOR_BCE_RECOVERY_20261015_A`  
**Target Variant**: `R_AUTHOR` (Author-Repository-Aligned CSAM)  
**Primary Objective**: Standard `torch.nn.BCEWithLogitsLoss()`  
**Authoritative Publication Repository**: `D:\LHRM_Publication` (remote: `https://github.com/NouraElsaman/LHRM.git`)  
**Target Execution Environment**: Kaggle GPU (T4 x2 or P100)

---

## 1. Executive Summary & Purpose

The client priority requires delivering a genuinely trained, testable LHRM-style model with real empirical measurements as expeditiously as possible. Following the completion of DR4 and the frozen counterfactual diagnostic, the forensic audit established that literal CSAM (paper Eqs. 7–8) discarded the document representation $h_T$ and suffered massive gradient starvation ($1496\times$) and focal-loss gradient imbalance ($64.2\times$). 

In contrast, the original paper authors' reference repository (`Model/lhrm.py`) avoided this failure by explicitly concatenating the document representation $[h_T, \text{agg}] \in \mathbb{R}^{2H}$ and optimizing standard BCE loss.

This document specifies the exact protocol for **Variant R_AUTHOR**, an author-code-aligned reference implementation. 

> [!IMPORTANT]
> **Scientific Classification**: `R_AUTHOR + BCE` is an **authors-code-aligned reference recovery**, NOT a strict mathematical reproduction of published paper Eqs. 3–8 or paper focal loss. Changing the architecture and loss together is justified here as an urgent reference-code recovery, not as a single-factor scientific ablation. No claim is made that R_AUTHOR has succeeded prior to authenticated GPU execution.

---

## 2. Comprehensive Architectural Comparison Matrix

| Component / Dimension | Paper Equations (Eqs. 3–8, 17) | Author Reference Code (`Model/lhrm.py`) | Our Tested Literal CSAM (`L` / `L2`) | Our Recovery Model (`R_AUTHOR`) |
|---|---|---|---|---|
| **Document Representation** | $h_T = \text{BERT}_\text{CLS}(X)$ (last hidden state) | `bert(...).pooler_output` (`Model/lhrm.py:185`) | `last_hidden_state[:, 0, :]` (CLS token) | `bert(...).pooler_output` (matching author repo) |
| **Label Embedding Initialization** | $h_{L_i} = \text{BERT}_\text{CLS}(Y_i)$ per batch with gradients | Precomputed once: mean over all token embeddings for `"label: definition"` (`Model/lhrm.py:222-243`) | Dynamic BERT CLS encoding every batch with gradients | Precomputed once: mean over all token embeddings for `"label: definition"` (matching author repo) |
| **Label Representation Dynamics** | Dynamic BERT encoding | Trainable parameter `nn.Parameter(label_emb)` (`Model/lhrm.py:147`) | Dynamic BERT encoding | Trainable parameter `nn.Parameter(label_emb)` (matching author repo) |
| **Label Self-Attention** | Scaled dot-product self-attention $A_{ij} = \text{softmax}(h_i h_j^T / \sqrt{d})$ | `nn.MultiheadAttention(768, heads=4)` + Residual + LayerNorm (`Model/lhrm.py:150-158`) | $A_{ij} = \text{softmax}(h_i h_j^T / \sqrt{d})$ explicit matrix | `nn.MultiheadAttention(768, heads=4)` + Residual + LayerNorm (matching author repo) |
| **Document-Label Similarity** | Unscaled dot product $S_i = h_T \cdot h_{L_i}$ (Eq. 4) | Unscaled dot product $w = \text{softmax}(h_\text{doc} \cdot L^T)$ (`Model/lhrm.py:159`) | Scaled dot product $S_i / \sqrt{H}$ in L2; unscaled in L | Unscaled dot product $w = \text{softmax}(h_\text{doc} \cdot L^T)$ (matching author repo) |
| **Document Signal Path** | **Discarded**: document enters solely as scalar attention weight $\alpha_i$ | **Preserved**: direct concatenation $[h_\text{doc}, \text{agg}] \in \mathbb{R}^{2H}$ (`Model/lhrm.py:190`) | **Discarded**: document enters solely as scalar weight $\alpha_i$ | **Preserved**: direct concatenation $[h_\text{doc}, \text{agg}] \in \mathbb{R}^{2H}$ (matching author repo) |
| **Classifier Head** | Shared scalar linear $W \in \mathbb{R}^{1 \times H}$ (Eq. 8) | Full linear projection `nn.Linear(2H, 54)` (`Model/lhrm.py:182`) | Shared $W \in \mathbb{R}^{1 \times H}$ (L/L2); unshared $W \in \mathbb{R}^{54 \times H}$ (L2_A5B) | Full linear projection `nn.Linear(2H, 54)` (matching author repo) |
| **Training Objective** | Focal Loss (Eq. 17) | `torch.nn.BCEWithLogitsLoss()` (`Model/lhrm.py:339`) | Focal Loss ($\alpha=0.25, \gamma=2.0$) | `torch.nn.BCEWithLogitsLoss()` (matching author repo) |
| **Training Epochs & Batch Size** | 15 epochs, batch size 8 (Section IV-D) | Default script: 3 epochs, batch size 16 (`Model/lhrm.py:19-20`) | 300 steps screening, batch size 8 | **15 epochs, batch size 8** (Paper training schedule) |

---

## 3. Training & Optimization Schedule

While the authors' quickstart script demonstrates execution with 3 epochs and batch size 16, this client delivery run executes the **official registered paper schedule**:
- **Dataset**: AAPD (53,840 train, 1,000 dev, 54 labels)
- **BERT Backbone**: `google-bert/bert-base-uncased` (pinned commit `86b5e0934494bd15c9632b12f734a8a67f723594`)
- **Max Sequence Length**: 512 tokens (head truncation)
- **Batch Size**: 8
- **Optimizer**: AdamW (`lr=2e-5`, `weight_decay=0.01`)
- **Number of Epochs**: 15 (total $\approx 100,950$ global optimizer steps)
- **Precision**: Mixed Precision (AMP enabled)
- **Seed**: 42 (deterministic CuDNN and RNG initialization)

### Informational Step-300 Checkpoint
An informational checkpoint is evaluated at global step 300:
- Evaluates training loss and Dev P@1 on the fixed 200 diagnostic documents.
- Provides immediate early visibility and comparisons against the historic 63/200 frequency prior.
- **Does NOT abort training**: Training continues smoothly through all 15 epochs regardless of step-300 score.

---

## 4. Non-Leaky HRRM Pipeline Protocol

To avoid the data leakage in the authors' monolithic script (fitting HRRM on test data and evaluating on the same rows), we enforce a **strictly non-overlapping 4-way Dev partition** generated deterministically from seed 42:
- **`diagnostic_ids` (200 documents)**: Matches the exact historical screening sample (frequency prior reference: 63/200). Used solely for continuity verification.
- **`model_selection_ids` (400 documents)**: Dedicated to tracking neural model validation performance and selecting `best_dev_model_only.pt`.
- **`hrrm_fit_ids` (200 documents)**: Used **exclusively to fit** the Gradient Boosting regressor on neural prediction outputs `(score, rank)`.
- **`hrrm_eval_ids` (200 documents)**: Held-out subset used **exclusively to evaluate** and compare base neural metrics against HRRM re-ranked metrics.

This guarantees:
1. HRRM is never evaluated on rows it was fitted on.
2. The comparison between base neural and HRRM re-ranked outputs is evaluated on identical held-out rows.
3. Test split remains completely quarantined.

---

## 5. Dataset Mapping Disclosures & Provenance

The client-approved original AAPD dataset is used without modification:
1. **Target Collinearities**:
   - `cmp-lg` and `cond-mat.dis-nn` have identical ground-truth column vectors.
   - `math.co` and `math.it` have identical ground-truth column vectors.
2. **Category Semantic Verification**:
   - The mapping of raw AAPD column names to actual scientific categories remains unverified (`UNRESOLVED_LABEL_MAPPING_ACCEPTED_BY_CLIENT_DECISION`).
   - Predictions report probabilities against the supplied raw target columns.
3. **Quarantine Policy**:
   - `test.csv` is strictly quarantined and is never read, copied, or evaluated during training or HRRM fitting.

---

## 6. Client Inference Demonstration

The workflow provides an interactive inference function:
`predict_abstract(text, model, tokenizer, label_order, device, hrrm_model=None, top_k=5)`
- Inputs any arbitrary unlabelled text string (e.g. abstract).
- Outputs Top-5 predicted labels with sigmoid probability scores.
- Outputs optional re-ranked Top-5 predictions if HRRM is attached.
- Displays a clean client-ready Markdown summary table with clear disclosure of unverified category mappings.

---

## 7. Critical Kaggle Notebook Release Patches (2026-10-15)

Following independent audit of the release deliverables, 8 critical patches were applied to canonical notebook `notebooks/lhrm_r_author_aapd_train_evaluate_kaggle.ipynb` and its generator `scripts/gen_kaggle_notebook_r_author.py`:

1. **Patch 1 — Bundle Manifest Dual-Path Verification**:
   - Resolved manifest hash distinction: sidecar manifest (`7a5f5fa0...`) contains `bundle_sha256` for local verification, while internal manifest (`8a6ebf14...`) defines constituent file hashes.
   - Dual-path support: verifies raw ZIP SHA (`e25c583e...`) when attached directly, or verifies internal manifest SHA and all 16 constituent files against expected hashes when auto-extracted by Kaggle.
2. **Patch 2 — Authentic AAPD CSV Loading**:
   - Replaced incorrect `'text'` column assumption with authentic `'abstract'` document text column.
   - Verified 55 total CSV columns (`abstract` + 54 binary target columns), 53,840 train rows, 1,000 dev rows, and exact label-order hash `a3d662f3920259a94ed61d83f98f21cabe7f908d98a5662ccd9adc2413cc000b`.
3. **Patch 3 — Preflight Variable Initialization & RUN_ID**:
   - Defined `RUN_ID`, `SEED`, `MODEL_VARIANT`, `LOSS_TYPE`, `LR`, `BATCH_SIZE`, etc. canonically in Cell 1 preflight.
4. **Patch 4 — Reproducible Seed Initialization**:
   - Set Python `random`, NumPy, PyTorch CPU, and PyTorch CUDA seeds to 42 before label embedding initialization and training.
5. **Patch 5 — Real Epoch-Boundary Checkpoint Resume**:
   - Implemented full state restoration (model, optimizer, AMP scaler, epoch, global step, history, best selection score, RNG states).
   - Atomic saving (`latest.pt.tmp` -> `latest.pt`) preventing checkpoint corruption on session interruption.
6. **Patch 6 — Leakage-Safe Model Selection**:
   - Restricted checkpoint selection to the 400 fixed `model_selection_ids`. Full Dev metrics are logged descriptively but do not drive checkpoint selection.
7. **Patch 7 — Checkpoint Consistency Across HRRM & Client Demo**:
   - Selected neural checkpoint (`best_dev_model_only.pt`) is loaded explicitly and used consistently for recomputing Dev predictions, fitting HRRM (on 200 fit docs), evaluating HRRM (on 200 held-out docs), and running client abstract inference.
8. **Patch 8 — Honest Complete Evidence Export**:
   - Produces all documented deliverables: `RUN_MANIFEST.json`, `CONFIG_RESOLVED.json`, `dev_partition.json`, `epoch_metrics.csv`, `train_history.json`, `training_summary.json`, `per_label_metrics.csv`, `step300_info.json`, `hrrm_heldout_comparison.json`, `inference_demo.json`, `neural_dev_predictions.json`, `SHA256SUMS.txt`, and `EVIDENCE.zip`.
   - Exposes valid `FileLink` download links for all generated outputs.
