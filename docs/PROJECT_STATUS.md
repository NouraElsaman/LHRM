# Project Status

## Original LHRM Implementation
- Introduced the scientific CSAM architecture mapping document text embeddings and label description embeddings into the same representation space.
- Historically involved multiple variants, some diverging from explicit paper equations (e.g. historical `csam_nosa` and `csam_r`).

## Variant L
- A literal, paper-faithful CSAM implementation explicitly mapping Eqs. 4-8.
- **Status**: Tested in 300-step `RUN_0` screening.
- **Result**: Demonstrated catastrophic rank collapse. The attention tensor `alpha` saturated unrecoverably, causing all documents to be assigned the same top-1 prediction (which did not exceed the raw dataset frequency prior of 63/200). 

## Variant L2
- A controlled revision to Variant L incorporating a $\frac{1}{\sqrt{H}}$ scaling factor to document-label similarity (Eq. 4), matching the stabilization found in Eq. 6.
- **Status**: Tested in `RUN_0` and subsequently extended to 1,000 steps (`DR1`).
- **Result (DR1)**: Successfully passed numerical stability, loss reduction, attention concentration (V3), and logit variance (V4). However, it decisively failed V5 (Ranking Superiority). Despite varying logits, the ordering remained document-insensitive, resulting in the exact same top-1 prediction for every document (63/200 hits).

## Completed DR2 BERT Control (Variant B)
- An independent BERT Multi-Label Control baseline (`[CLS] -> Linear(54)`) utilizing standard `BCEWithLogitsLoss`. It does not explicitly encode label descriptions or utilize CSAM.
- **Status**: Full 15-epoch training completed on Kaggle.
- **Result**: Achieved P@1 of 0.8820 (Epoch 5) and Micro-F1 of 0.7557. Demonstrates standard baseline learning capability on the AAPD dataset. Test split and HRRM fitting were intentionally excluded.

## Prepared Ablation: L2_NO_SA
- A mathematically isolated ablation of Variant `L2` which removes the label-label self-attention projection term ($A_{ij} h_{L_j}$). 
- **Status**: Architecture and test coverage prepared. 
- **Next Step**: Pending GPU execution and diagnostic screening on Kaggle. Not yet evaluated.

## Future Work
- **HRRM Fitting**: To be executed on validated Dev predictions downstream after a final baseline architecture is cemented.
- **Test Split Evaluation**: Completely locked. No evaluation is performed on the Test split until the final protocol is frozen.

---

## Addendum: RUN_0 V1-V5 Criteria Documentation Discrepancy
Historical project documents contained inconsistent descriptions of the `RUN_0` viability rules, particularly suggesting that **V5** (Ranking Superiority) was an informational sanity check rather than a strict gating requirement.

However, the actual executed code (`src/lhrm_baseline/diagnostics.py`) and contemporaneous preregistration (`docs/phase1/run0_preregistration.md`) explicitly mandate passing ALL five criteria (`viable = V1 and V2 and V3 and V4 and V5`). 

**Reconciliation**: The recorded code execution and active preregistration are treated as primary evidence. V5 is a mandatory gate for architecture viability. While V5 (comparing P@1 against the static frequency prior) may not be a universal measure of overall model quality, it is the strict mathematical threshold required for this specific screening protocol. The historical phrasing describing V5 as optional is explicitly superseded by the enforced execution logic.
