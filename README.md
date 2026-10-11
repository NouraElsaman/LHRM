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

---

## Latest Experimental Evidence — RUN_0 / DR1

### Variant L (300-step screening) - Attention Concentration Failure
![Variant L Screening](snapshots/run0_variant_l_screening.png)

### Variant L2 (1000-step DR1) - Passed numerical checks but failed ranking (V5)
![Variant L2 DR1](snapshots/dr1_variant_l2_1000_steps.png)

### DR1 Prediction Concentration Audit
*Note: This is a read-only local audit script confirming the structural top-1 identical prediction across 200 validation documents, derived from the original Kaggle evidence.*
![DR1 Audit](snapshots/dr1_prediction_concentration_audit.png)
