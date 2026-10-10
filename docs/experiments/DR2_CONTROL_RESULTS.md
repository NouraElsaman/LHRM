# DR2 Control Baseline Results (Variant B)

## Overview
This document records the results of Deviation Request 2 (DR2), an independent BERT Multi-Label Classification control baseline evaluated on the AAPD dataset.

**Important Note:** This is an independent control baseline (`[CLS] -> Linear(54)`) utilizing BCEWithLogitsLoss. It is **not** a paper-faithful reproduction of LHRM, as it does not explicitly encode label descriptions or utilize CSAM.

## Experiment Configuration
- **Dataset:** AAPD (Zenodo source)
- **Train Documents:** 53,840
- **Dev Documents:** 1,000
- **Labels:** 54
- **Architecture:** Variant B (Standard BERT Multi-Label Control)
- **Loss:** `bce` (BCEWithLogitsLoss)
- **Epochs Completed:** 15
- **Optimizer Steps:** 100,950
- **Seed:** 42

## Execution Outcomes
- **Training process exit code:** 0
- **Final TEST evaluation:** NOT EXECUTED
- **HRRM fitting:** NOT EXECUTED

## Recorded DEV Results (Threshold 0.5)

| Metric | Best Observed | Final (Epoch 15) |
|---|---|---|
| **P@1** | 0.8820 (Epoch 5) | 0.8680 |
| **Micro-F1** | 0.7557 (Epoch 5) | 0.7358 |
| **Macro-F1** | 0.6055 (Epoch 8) | 0.5757 |
| **P@3** | 0.6147 | - |
| **P@5** | 0.4120 | - |
| **nDCG@5** | 0.8522 | - |
| **Training Loss** | (Not for selection) | 0.0062 |

*Note: Decreasing training loss alongside weaker late-epoch Dev metrics is consistent with possible overfitting, but no definitive cause is claimed.*

## Checkpoint Policy
Checkpoints were saved on a rolling basis (`latest.pt` and `previous.pt`) with one explicitly saved at `step_100950.pt` representing the absolute final 15th epoch state for potential unbiased downstream HRRM evaluation. The downloaded Kaggle evidence ZIP (if any) omits these heavy `.pt` files to prevent large multi-GB transfers.

## External Artifact References
- The downloaded DR2 artifacts from Kaggle may be found at `D:\LHRM_KAGGLE_ARTIFACTS\RUN_DR2_B_20261010\` locally on the executing machine. They are excluded from this repository.
