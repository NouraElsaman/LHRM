# Label-Aware Hierarchical Ranking Model (LHRM)

## Research Objective
Multi-label text classification requires predicting multiple labels for a document. Existing approaches often treat labels independently. The LHRM architecture aims to map document text embeddings and label description embeddings into the same representation space using a Contextual Similarity Attention Module (CSAM), establishing dependencies before final ranking.

## Original Paper and Attribution
- **Authors:** Lama Ayash, Abdulmohsen Algarni, and Omar Alqahtani
- **Journal:** IEEE Access (2025)
- **DOI:** 10.1109/ACCESS.2025.11129041
- **Original Source Repository:** [LamaAy/Label-Aware-Hierarchical-Ranking-Model-for-Multi-Label-Text-Classification](https://github.com/LamaAy/Label-Aware-Hierarchical-Ranking-Model-for-Multi-Label-Text-Classification)

## Subsequent Engineering and Experimental Contributions
This repository builds upon the original client implementation by adding significant rigor and reproducibility pipelines:
- Complete end-to-end reproducibility auditing.
- Isolation of test sets to prevent data leakage.
- Independent standard Multi-Label BERT baseline implementation.
- GPU / Kaggle-ready environment packaging.
- Deterministic evaluation protocols (P@k, nDCG@k, F1).
- Controlled structural screening and architectural ablations.

## Current Architecture Variants
The source tree supports several architecture options:
- **Variant L**: The literal, paper-faithful CSAM implementation explicitly mapping Eqs. 4-8.
- **Variant L2**: A controlled stabilization of `L` incorporating $\frac{1}{\sqrt{H}}$ scaling to the document-label similarity computation.
- **Variant L2_A5B**: An alternative architectural interpretation parameterizing an unshared classifier weight $W \in \mathbb{R}^{k \times H}$ with shared scalar bias $b \in \mathbb{R}$, retaining label-label self-attention.
- **Variant B**: A standard independent BERT control (`[CLS] -> Linear(54)`) utilizing BCEWithLogitsLoss.
- **Variant L2_NO_SA**: A prepared isolated mathematical ablation removing label-label self-attention.
- **Variant R**: A repository-style fallback CSAM using concat aggregation.

## Dataset and Environment
- **Dataset**: AAPD (53,840 train, 1,000 dev, 54 labels) based on the original Zenodo source. 
- **Environment**: Python 3.12+, PyTorch 2.x, Transformers. *Note: You must independently procure the AAPD dataset; it is not bundled.*
- **Pretrained Checkpoint**: `google-bert/bert-base-uncased`.

## Repository Structure
- `run_baseline.py`: Central CLI for dataset preparation, screening, training, and artifact management.
- `src/lhrm_baseline/`: Core modular pipeline (data, model, losses, diagnostics, train, metrics).
- `legacy/` and `Model/`: Retained original client implementations.
- `configs/`: Immutable yaml configuration files for deterministic execution.
- `notebooks/`: Executable Kaggle environment definitions.
- `docs/`: Phase analysis and decision registries.

## Working CLI Commands
Dataset Preparation:
```bash
python run_baseline.py prepare --config configs/final/aapd_baseline.yaml
```

Architecture Viability Screening (RUN_0 - Requires GPU):
```bash
python run_baseline.py run0 --config configs/final/aapd_baseline.yaml --run_id RUN_0_L2_NO_SA_20261010 --steps 300 --candidates L2,L2_NO_SA
```

Full Baseline Training (Requires GPU):
```bash
python run_baseline.py train --config configs/final/aapd_baseline_dr2.yaml --run_id RUN_DR2_B --variant B
```

## Experimental Chronology
### RUN_0 & DR1 Diagnostic Findings
- **Variant L (300 steps)**: Failed early viability screening due to unrecoverable attention saturation (rank collapse).
- **Variant L2 (1000 steps, DR1)**: Restored numerical stability but decisively failed ranking superiority (V5). The ordering was document-insensitive, producing the exact same top-1 prediction for every document in the 200-document diagnostic sample.

### Completed DR2 Control (Variant B)
An independent non-CSAM standard BERT Multi-Label Baseline was trained for 15 epochs.
- Reached P@1 of 0.8820 on Dev at Epoch 5.
- Validates the data split and establishes a ceiling reference for CSAM components.

## Current Scientific Limitations
The paper-derived CSAM architecture (L/L2) has demonstrated a structural bottleneck where it learns a static, document-independent ranking. This indicates that explicitly encoding label descriptions into the same embedding space as document texts without further non-linear projections fails to distinguish document semantics on AAPD.

### Completed Controlled Ablation: L2 vs L2_NO_SA (DR3)
Executed 300 optimizer steps per candidate on Kaggle (`RUN_0_L2_NO_SA_ABLATION_20261011_A`) to evaluate whether removing the label-label self-attention projection term allows document-sensitive ranking to emerge.
- **Outcome**: Both variants decisively failed V5 (Top-1 hits 63/200 == dataset frequency prior 63/200).
- Removing self-attention tightened the collapse: V4 logit sensitivity fell from 20/54 to 0/54, and 200/200 documents received identical Top-1 predictions (`cond-mat.dis-nn`).
- Empirically refutes the hypothesis that label-label self-attention was the sole cause of ranking collapse.
- Detailed postmortem documented in [`docs/experiments/L2_NO_SA_POSTMORTEM.md`](docs/experiments/L2_NO_SA_POSTMORTEM.md) and controlled gradient micro-probe in [`docs/experiments/DR3_GRADIENT_MICRO_PROBE.md`](docs/experiments/DR3_GRADIENT_MICRO_PROBE.md).

### Completed DR3 Replay & Authentic Trained Gradient Diagnostic (`RUN_0_L2_NO_SA_REPLAY_20261011_B`)
Executed controlled 300-step replay on Kaggle GPU, recovered complete step-300 trained checkpoints (`L2_STEP300_MODEL_ONLY.pt`, `L2_NO_SA_STEP300_MODEL_ONLY.pt`), and evaluated an authentic post-training gradient diagnostic on real representations.
- **Outcome**: Replicated V1–V5 pass/fail profile and P@1 hit count (63/200), not bitwise-identical predictions. Prediction differences between collinear labels are not attributed solely to GPU nondeterminism without direct trajectory evidence.
- **Trained Diagnostic Measurements**:
  - In `L2`, static label self-attention ($1.651$) dominates the document term ($0.034$) by $48.42\times$.
  - In `L2_NO_SA`, cross-attention remains diffuse (entropy 3.945 / 3.989, 98.9% uniform), with cross-document document term variance of only $0.00207$ (vs. $0.05447$ across labels).
  - Paired with the shared scalar classifier ($W \in \mathbb{R}^{1 \times H}$), this locks label rankings into identical order across all documents.
  - Earlier synthetic probe was an illustrative baseline, not the true initialization state of trained models.
  - Trained L2 gradient ratios are not identical across losses (document-to-classifier ratio was $2.695 \times 10^{-5}$ under BCE vs $3.769 \times 10^{-4}$ under Focal); BCE as a training objective remains untested in CSAM.
  - Comprehensive empirical analysis documented in [`docs/experiments/DR3_TRAINED_REPLAY_FINAL_ANALYSIS.md`](docs/experiments/DR3_TRAINED_REPLAY_FINAL_ANALYSIS.md).

### Completed DR4 Screening & Trained Logit Decomposition (`RUN_0_L2_A5B_20261012_A`)
Executed 300 optimizer steps comparing `L2` (shared classifier) versus `L2_A5B` (unshared per-label classifier $W \in \mathbb{R}^{k \times H}$, Decision A5b), followed by post-training logit decomposition (`DR4_POSTTRAIN_DECOMPOSITION_20261012`) on real trained checkpoints.
- **Outcome**: Both variants failed viability screening (V4: 21/54 for L2, 3/54 for L2_A5B; V5: 63/200 Top-1 hits, matching the frequency prior).
- **Predictions**: L2 predicted `cond-mat.dis-nn` for 200/200 documents. L2_A5B predicted `cmp-lg` for 199/200 documents and `cond-mat.dis-nn` for 1/200 documents (both labels share identical target columns in AAPD).
- **Trained Logit Decomposition**:
  - Confirmed exact algebraic logit decomposition ($\text{logits}_{b, i} = \text{doc\_term}_{b, i} + \text{sa\_term}_i + b$) with reconstruction error $< 7.15 \times 10^{-7}$.
  - The static document-invariant label self-attention term dominated the document-dependent term by $48.03\times$ in L2 and $51.82\times$ in L2_A5B.
  - The 54 unshared classifier weight rows in `L2_A5B` remained highly collinear (mean off-diagonal cosine similarity $0.9985$).
  - Full postmortem documented in [`docs/experiments/DR4_FINAL_POSTMORTEM.md`](docs/experiments/DR4_FINAL_POSTMORTEM.md).
- **Causality Limit**: Unsharing the classifier head alone did not resolve ranking collapse under the 300-step screening protocol, but this observational finding does not establish an inductive proof regarding all training regimes or initializations.
- **Standard Baseline Reference**: The standard BERT multi-label control baseline (DR2 / Variant B, Dev P@1 = 0.8820) remains strictly distinct from CSAM reproduction claims.

### Completed DR4 Frozen Label-SA Counterfactual (`DR4_FROZEN_SA_CF_20261012_A`)
Evaluated inference-only removal of static label self-attention offsets ($\text{original\_logits} - \text{sa\_term}$) on trained step-300 checkpoints.
- **Outcome**: 0 recovered hits. Both models remained at 63/200 Top-1 hits (31.50%), matching the frequency prior.
- V4 cross-document logit standard deviation was strictly invariant under the constant offset subtraction ($\Delta \sigma \le 4.28 \times 10^{-8}$).
- Refuted the hypothesis that static label-SA offsets were the exclusive cause of ranking collapse. Document-dependent terms themselves fail to generate useful discrimination.
- Full analysis documented in [`docs/experiments/DR4_FROZEN_SA_CF_FINAL_ANALYSIS.md`](docs/experiments/DR4_FROZEN_SA_CF_FINAL_ANALYSIS.md).

### CSAM Implementation Forensics & Root-Cause Audit
A systematic equation-to-code audit explained why CSAM failed to learn despite the successful DR2 control:
1. **Structural Information Discard**: Literal paper equations (Eqs. 7–8) discard the document CLS embedding $h_T$, routing document signal purely through attention weights $\alpha_i$. In contrast, the original authors' reference repository concatenated $[h_T, \text{agg}] \in \mathbb{R}^{2H}$.
2. **Attention Dilution & Gradient Starvation**: Softmax normalization over 54 labels forces mean $\alpha \approx 1/54$. Combined with $1/\sqrt{H}$ scaling, the analytical backpropagation factor to document BERT is $\approx 6.68 \times 10^{-4}$ ($1496\times$ suppression), leaving the document encoder effectively frozen at initialization.
3. **Focal Loss Gradient Inbalance**: Under $\alpha=0.25$, positive errors receive $0.25$ weight while negative errors receive $0.75$ weight ($3\times$ downweighting). Under AAPD sparsity (95.5% zeros), negative gradient mass dominates by $64.5\times$, driving the model into a static negative-bias frequency-prior state.
- Full investigation and hypothesis matrix in [`docs/experiments/CSAM_DOCUMENT_SIGNAL_ROOT_CAUSE_AUDIT.md`](docs/experiments/CSAM_DOCUMENT_SIGNAL_ROOT_CAUSE_AUDIT.md).
- Controlled recovery proposal and preregistration in [`docs/experiments/CSAM_RECOVERY_DECISION_AND_NEXT_EXPERIMENT.md`](docs/experiments/CSAM_RECOVERY_DECISION_AND_NEXT_EXPERIMENT.md).

---

## Latest Experimental Evidence — RUN_0 / DR1

### Variant L (300-step screening) - Attention Concentration Failure
![Variant L Screening](snapshots/run0_variant_l_screening.png)

### Variant L2 (1000-step DR1) - Passed numerical checks but failed ranking (V5)
![Variant L2 DR1](snapshots/dr1_variant_l2_1000_steps.png)

### DR1 Prediction Concentration Audit
*Note: This is a read-only local audit script confirming the structural top-1 identical prediction across 200 validation documents, derived from the original Kaggle evidence.*
![DR1 Audit](snapshots/dr1_prediction_concentration_audit.png)
