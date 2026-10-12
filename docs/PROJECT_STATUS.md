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

## Completed Controlled Ablation: L2 vs L2_NO_SA (`RUN_0_L2_NO_SA_ABLATION_20261011_A`)
- A mathematically isolated ablation of Variant `L2` which removes the label-label self-attention projection term ($A_{ij} h_{L_j}$).
- **Status**: Executed on Kaggle GPU (300 steps per candidate) and independently audited (`DR3`).
- **Result**: Decisively **NOT VIABLE**. Both variants failed V4 and V5 (Top-1 hits 63/200 == frequency prior 63/200). Removing label-label self-attention tightened ranking collapse from 2 distinct Top-1 labels to 1 (200/200 assigned `cond-mat.dis-nn`), and reduced V4 passing labels from 20/54 to 0/54. Empirically refutes the hypothesis that label-label self-attention was the sole cause of ranking collapse. Detailed investigation documented in [`docs/experiments/L2_NO_SA_POSTMORTEM.md`](experiments/L2_NO_SA_POSTMORTEM.md) and controlled micro-probe in [`docs/experiments/DR3_GRADIENT_MICRO_PROBE.md`](experiments/DR3_GRADIENT_MICRO_PROBE.md).

## Completed DR3 Checkpoint Replay & Trained Gradient Diagnostic (`RUN_0_L2_NO_SA_REPLAY_20261011_B`)
- **Status**: Executed on Kaggle GPU, deliverables downloaded to `D:\LHRM_KAGGLE_ARTIFACTS\RUN_0_L2_NO_SA_REPLAY_20261011_B\`, verified with SHA-256 manifest, and integrated.
- **Result**: Replicated the V1–V5 pass/fail profile and exact P@1 hit count (63/200, 31.50%) of `RUN_0_L2_NO_SA_ABLATION_20261011_A`, not bitwise-identical predictions. Observed prediction differences between collinear labels (`cmp-lg` vs `cond-mat.dis-nn`) are not attributed solely to GPU nondeterminism without direct trajectory evidence.
- **Checkpoints Preserved**: Successfully recovered and verified lightweight step-300 model-only checkpoints for both `L2` (`d703579c...`, 417.7 MB) and `L2_NO_SA` (`c9dd8ebc...`, 417.7 MB).
- **Trained Diagnostic Findings**:
  - In `L2`, the static label-label self-attention term ($1.651$) dominates the document term ($0.034$) by $48.42\times$.
  - In `L2_NO_SA`, cross-attention remains diffuse (entropy 3.945 / 3.989, 98.9% uniform), causing document term variation across documents to be only $0.00207$ (vs. $0.05447$ across labels).
  - Paired with the shared scalar classifier ($W \in \mathbb{R}^{1 \times H}$), this locks label rankings into identical order across all documents.
  - The earlier synthetic probe was an illustrative baseline, not the true initialization state of the trained models.
  - Trained L2 gradient ratios are not identical across losses (document-to-classifier ratio was $2.695 \times 10^{-5}$ under BCE vs $3.769 \times 10^{-4}$ under Focal); BCE as a training objective remains untested in CSAM.
  - Full analysis documented in [`docs/experiments/DR3_TRAINED_REPLAY_FINAL_ANALYSIS.md`](experiments/DR3_TRAINED_REPLAY_FINAL_ANALYSIS.md).

## Completed DR4 Controlled Screening & Post-Training Decomposition (`RUN_0_L2_A5B_20261012_A`)
- **Status**: Executed on Kaggle GPU, verified, and post-training logit decomposition diagnostic completed (`DR4_POSTTRAIN_DECOMPOSITION_20261012`).
- **Result**: Decisively **NOT VIABLE**. Both `L2` and `L2_A5B` failed V4 and V5 (P@1 correct hits: 63/200, matching frequency prior 63/200).
- **Key Findings**:
  - Unsharing the classifier head parameters ($W \in \mathbb{R}^{54 \times H}$) did not alleviate ranking collapse. Instead, passing labels under V4 dropped from 21/54 in `L2` to 3/54 in `L2_A5B`.
  - The static label self-attention term ($1.624$) dominates the document term ($0.031$) by $51.82\times$ in `L2_A5B` (vs $48.03\times$ in `L2`).
  - Cross-attention became more diffuse under `L2_A5B` (attention entropy rose from 3.6907 to 3.9537; mean max $\alpha$ dropped from 0.0406 to 0.0278).
  - The 54 unshared weight vectors of $W$ remained highly collinear (mean off-diagonal cosine similarity $0.9985 \pm 0.0022$).
  - Full postmortem documented in [`docs/experiments/DR4_FINAL_POSTMORTEM.md`](experiments/DR4_FINAL_POSTMORTEM.md).

## Completed DR4 Frozen-Checkpoint Label-SA Counterfactual (`DR4_FROZEN_SA_CF_20261012_A`)
- **Status**: Executed on Kaggle GPU, authentic evidence verified (`DR4_FROZEN_SA_CF_EVIDENCE.zip`, SHA-256: `a32eab0a33a39bb079259e175b61dd442d33386cb8412f59f69cda93bd92a610`), and finalized.
- **Intervention**: Frozen-model counterfactual removing static label self-attention offsets at inference time: $\text{counterfactual\_logits}[b, i] = \text{original\_logits}[b, i] - \text{sa\_term}[i]$.
- **Results**:
  - Top-1 correct count strictly unchanged: exactly 63/200 (31.50%) for both L2 and L2_A5B (0 net improvement over frequency prior 63/200).
  - V4 variance invariant under constant shift ($\Delta \sigma \le 4.28 \times 10^{-8}$): L2 remains 21/54; L2_A5B remains 3/54.
  - Distinct Top-1 labels: L2 shifted from 1 (`cond-mat.dis-nn`: 200) to 2 (`cond-mat.dis-nn`: 199, `cs.se`: 1); L2_A5B shifted from 2 (`cmp-lg`: 199, `cond-mat.dis-nn`: 1) to 1 (`cond-mat.dis-nn`: 200). Note `cmp-lg` and `cond-mat.dis-nn` have identical ground truth columns.
  - Mean counterfactual probability margin between Top-1 and Top-2 is $3.42 \times 10^{-6}$, indicating extreme numerical proximity.
  - Refutes hypothesis that static label-SA offsets were the exclusive cause of ranking failure. Document-dependent terms themselves fail to generate useful label discrimination.
  - Full report in [`docs/experiments/DR4_FROZEN_SA_CF_FINAL_ANALYSIS.md`](experiments/DR4_FROZEN_SA_CF_FINAL_ANALYSIS.md).

## Completed Forensic Audit & Root-Cause Matrix
- **Status**: Comprehensive audit of CSAM signal paths, gradient flow, and loss formulation documented in [`docs/experiments/CSAM_DOCUMENT_SIGNAL_ROOT_CAUSE_AUDIT.md`](experiments/CSAM_DOCUMENT_SIGNAL_ROOT_CAUSE_AUDIT.md).
- **Key Findings**:
  - Identifies three compounding failure mechanisms in literal CSAM (L/L2):
    1. *Structural Information Bottleneck*: Literal CSAM (Eqs. 7–8) drops document representation $h_T$, routing document signal purely through attention weights $\alpha_i$ (unlike author's reference repo which concatenates $[h_T, \text{agg}]$).
    2. *Gradient Starvation*: Dilute cross-attention over 54 labels combined with $1/\sqrt{H}$ scaling attenuates backpropagated gradients to document BERT by $\approx 1496\times$.
    3. *Loss Gradient Imbalance*: Focal loss with $\alpha=0.25$ downweights positives $3\times$ relative to negatives, resulting in $64.5\times$ greater negative gradient mass under AAPD sparsity ($95.5\%$ zeros), causing early collapse to a document-invariant negative frequency-prior state.
- **Recovery Decision**: Preregistered Category 2 frozen-backbone diagnostic probe followed by Category 3 recovery screen (Variant R + BCE) documented in [`docs/experiments/CSAM_RECOVERY_DECISION_AND_NEXT_EXPERIMENT.md`](experiments/CSAM_RECOVERY_DECISION_AND_NEXT_EXPERIMENT.md).

## Future Work
- **HRRM Fitting**: To be executed on validated Dev predictions downstream after a final baseline architecture is cemented.
- **Test Split Evaluation**: Completely locked. No evaluation is performed on the Test split until the final protocol is frozen.

---

## Addendum: RUN_0 V1-V5 Criteria Documentation Discrepancy
Historical project documents contained inconsistent descriptions of the `RUN_0` viability rules, particularly suggesting that **V5** (Ranking Superiority) was an informational sanity check rather than a strict gating requirement.

However, the actual executed code (`src/lhrm_baseline/diagnostics.py`) and contemporaneous preregistration (`docs/phase1/run0_preregistration.md`) explicitly mandate passing ALL five criteria (`viable = V1 and V2 and V3 and V4 and V5`). 

**Reconciliation**: The recorded code execution and active preregistration are treated as primary evidence. V5 is a mandatory gate for architecture viability. While V5 (comparing P@1 against the static frequency prior) may not be a universal measure of overall model quality, it is the strict mathematical threshold required for this specific screening protocol. The historical phrasing describing V5 as optional is explicitly superseded by the enforced execution logic.
