# RUN_0 Pre-Registration: Architecture Viability Screening (AAPD)

**Date pre-registered:** 2026-10-07
**Supersedes:** Reuters-focused run0 pre-registration (Phase 1D; that file now SECONDARY_PATH)
**Status:** PRE_REGISTERED (no run has occurred)

---

## 1. Purpose

RUN_0 is a DIAGNOSTIC_ONLY / SMOKE_ONLY screening run.
Its sole purpose is to determine whether architecture variant L or L2 is viable
on the AAPD dataset. All outputs are labelled SMOKE_ONLY or DIAGNOSTIC_ONLY.
No output from RUN_0 appears in any baseline results table.
The test split (test.csv, 1,000 docs) is NOT loaded or predicted during RUN_0.

---

## 2. Candidates

Candidate L:  Variant L  -- literal Eqs. 4-8 (dot-product similarity, no scaling)
Candidate L2: Variant L2 -- Eq. 4 with S_i/sqrt(768) scaling

Same seed=42, same data order, both trained in the same notebook session.
Screening order: L first, then L2. The first viable candidate is selected.
Selection is NEVER by performance.

---

## 3. Data

Dataset: AAPD (Zenodo 10.5281/zenodo.6344750, USE_ZENODO_SOURCE_AS_PROVIDED)
Train split: train.csv (53,840 docs), first 300 optimisation steps, batch=8
Dev sample: fixed 200-document sample from dev.csv (1,000 docs), seed=42
  - 200 doc IDs drawn at session start, saved to run0_val_sample_ids.json
  - Used for V3/V4/V5 diagnostics only; NOT tuned or inspected before the run
Test split: NOT LOADED during RUN_0

Note: AAPD has two pairs of byte-identical label columns:
  cmp-lg == cond-mat.dis-nn (N active training docs: see aapd_data_decision.md)
  math.co == math.it
The frequency-prior ranker used in V5 must use seeded tie-breaking on these ties.

---

## 4. Viability Criteria (V1-V5, pre-registered numeric)

| ID | Criterion | Pass condition |
|---|---|---|
| V1 | All losses finite for all 300 steps | nan_inf_count == 0 (record AMP overflow/skip counts) |
| V2 | Loss decreases overall | mean(last 50 steps) < mean(first 50 steps) |
| V3 | Alpha not collapsed to one-hot | fraction of 200 dev docs with max_i(alpha_i) > 0.9 is < 0.9 |
| V4 | Logit variance | for > 50% of 54 labels, std of label logit across 200 docs is > 0.01 |
| V5 | P@1 sanity check | candidate P@1 > label-frequency-prior P@1 on 200 dev docs (seeded tie-break) |

All five must pass for a candidate to be declared viable.

---

## 5. Decision Rule (scripted; no human eye-picking)

1. Evaluate L by V1-V5.
2. If L passes all five: VIABLE. Select L. Write FREEZE-B = L. STOP.
3. If L fails: evaluate L2 by V1-V5.
4. If L2 passes: VIABLE. Select L2. Write FREEZE-B = L2. STOP.
5. If neither passes all five: STOP_AND_ASK (report results; wait for user instruction).

Pre-registered safeguards:
  (a) V3-only failure in L: if argmax-alpha is the true label for > 50% of dev docs,
      do NOT silently fall through to L2. Report both V3 result and the argmax diagnostic;
      ask the user. This is an escalation condition, not a failure.
  (b) V2/V5-only failure at 300 steps: a single pre-declared 1,000-step extension may
      be run for BOTH candidates. Nothing further without user approval.

The decision function writes run0_decision.json with:
  chosen_variant, V1..V5 boolean values, reason, status
  (DECIDED / ESCALATE / EXTEND / STOP_AND_ASK / NONE_VIABLE)

---

## 6. Informational Diagnostics (not criteria)

Reported at steps 0 and 300 for each candidate:
  - Distribution of similarity scores S_i (Eq. 4) across 200 dev docs
  - Mean and std of max-alpha; normalised entropy of alpha distribution
  - Number of distinct argmax-alpha labels across 200 dev docs
  - Loss curve (all 300 steps logged to train.log)
  - Gradient norms for document-path BERT parameters, label-path BERT parameters,
    and W (shared linear head); checked for non-zero flow
  - AMP overflow/skip count
  - Checkpoint-resume equivalence: checkpoint at step 150; resume to 300;
    compare loss at step 150-300 vs uninterrupted fp32 20-step reference (tolerance stated)
  - Throughput (steps/s) and peak GPU memory (GB)
  - HRRM code-path check: fit on one half of the 200-doc dev predictions,
    apply to the other half. Code check ONLY; not a performance measure.

---

## 7. Infrastructure Checks (DIAGNOSTIC_ONLY)

All checked within the notebook before the 300-step loop:
  - GPU/driver/CUDA/PyTorch/transformers versions (env.json)
  - Parameter counts for BERT encoder, CSAM module, linear head W
  - Forward pass shapes: h_T [B, 768]; h_L [54, 768]; alpha [B, 54]; logits [B, 54]
  - Backward pass: gradient to h_T, h_L, W all non-zero
  - Gradient flow from document BERT and label BERT paths separately

---

## 8. Artifacts (named by experiment ID)

Saved to experiments/{run_id}/:
  - resolved_config.json (with _config_freeze_sha256, _label_order_sha256, _git)
  - variant_L/step0_diagnostics.json
  - variant_L/step300_diagnostics.json
  - variant_L/checkpoints/step_150.pt
  - variant_L/checkpoints/step_300.pt
  - variant_L/train.log
  - variant_L/val_sample_predictions.json
  - (same for variant_L2 if run)
  - run0_val_sample_ids.json
  - run0_decision.json
  - env.json
  - results_summary.json
  - SHA256SUMS.txt

Import: python run_baseline.py import_artifacts --path experiments/{run_id}/

---

## 9. Boundaries

After FREEZE-B is written:
  - The architecture variant is fixed. It is NOT changed after seeing any test result.
  - Nothing else in CONFIG_FREEZE.json changes without a new registry entry.
  - The AAPD test split is predicted exactly ONCE, during the full baseline run (RUN A).

CONFIG_FREEZE.json stage A SHA-256: d40edf3959792d56d0a63014cbed6661f1fb0d63463be8215fa385404e222205
