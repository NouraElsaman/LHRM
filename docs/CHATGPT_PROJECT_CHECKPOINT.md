# LHRM Baseline — Project Checkpoint

**Updated:** 2026-10-09T23:23:00Z
**Phase:** 1 (RUN A Readiness — implementation complete)
**Git HEAD:** `7939d86b0b6bfba7097178f1e3f07ee37b682f12` (dirty)

---

## Current status

| Gate | Status | Evidence |
|---|---|---|
| G0 Data integrity | PASS | AAPD.zip + all 3 CSVs hash-verified |
| G1 Label order | PASS | SHA256 = `a3d662f392...` |
| G2 Unit tests | **PASS (86/88)** | 86 passed, 2 skipped — includes 13 new RUN A readiness tests |
| G3 Metrics | PASS | |
| G4 HRRM | PASS | |
| **RUN_0** | **NONE_VIABLE** | Both L and L2 failed preregistered V1–V5 |
| FREEZE-B | **NOT SET** | DR2 (Variant B) approved as control baseline |
| DR1 (1000-step ext) | **NOT VIABLE** | Executed, V5 failed (tied prior 63/200) |
| DR2 (Control Baseline)| **APPROVED** | In progress (Variant B) |
| **RUN A readiness** | **✅ IMPLEMENTATION COMPLETE** | All A1/A2/A4/A7/A8/A9 defects fixed |
| RUN A | NOT STARTED | Blocked on FREEZE-B |

---

## RUN_0_P1F — verified results

**Full archive SHA256:** `74930ca14074fe1123f13fdd10d4f5043a23a775571e06ab20dc4508306802c4`
**Location:** `D:\LHRM_KAGGLE_ARTIFACTS\RUN_0_P1F_20261009\`
**Pointer:** `experiments/RUN_0_P1F/ARTIFACT_POINTER.json`

### V1-V5 matrix

| | V1 | V2 | V3 | V4 | V5 | Decision |
|---|---|---|---|---|---|---|
| **L** | ✅ | ✅ | ❌ 0.98 | ❌ 5/54 | ❌ 63=63 | NONE_VIABLE |
| **L2** | ✅ | ✅ | ✅ 0.00 | ❌ 18/54 | ❌ 63=63 | NONE_VIABLE |

### Key observations (from actual diagnostic JSON, not console)

- L alpha collapses: `one_hot_frac` 0.41→0.98; `n_distinct_argmax` 19→1 (all docs route to 1 label)
- L2 alpha healthy: `one_hot_frac` 0.00 maintained; `mean_max_alpha` 0.030→0.043
- V4 threshold = 28/54 labels with per-label logit std > 0.01 (fp32). L: 5/54. L2: 18/54.
- V5 tie is confirmed real (not float artefact). Both produce exactly 63/200 = prior.
- Extension NOT eligible: V4 fails both, V3 fails L. Preregistration requires V2/V5-only failure.

---

## AAPD source check summary

```
ZIP SHA:      PASS  e1b418427bcd...
train.csv:    PASS  e7e73064385f...
dev.csv:      PASS  a85f612d4c1c...
test.csv:     PASS  a0672754aead... (read-only integrity; not used in experiment)
label order:  PASS  a3d662f39202...
Project provenance: PASS
Kaggle raw-byte CSV equality: NOT_INDEPENDENTLY_VERIFIED
```

---

## Section H — ONE recommended next action

### Option: 1,000-step Variant L2 extension (preregistration required first)

**Rationale:** L2 is the only candidate that passes V3. Its failures are V4 (18/54 → needs 27/54)
and V5 (tie with prior). These are both consistent with a model that has begun to learn but hasn't
converged. 300 steps is approximately 4.46% of one epoch over 53,840 training documents with batch
size 8 (≈6,730 steps/epoch). The signal is extremely early.

**V4 gap:** needs 9 more labels to develop std > 0.01. At step 300, median std = 0.0085 (below
0.01 threshold). This is a soft failure — small additional training may push several labels over.

**V5 gap:** any improvement in P@1 beyond 63/200 would pass. The model is currently at random-init-
quality top-1 predictions after only 300 steps on a 54-label problem.

**This option requires a new preregistration deviation because the preregistration only permits
automatic extension for V2/V5-only failure.** A deviation must document:
1. Why V4-failing-by-margin qualifies for extension (explicit argument)
2. The exact step count (1,000)
3. That L is NOT extended (V3 collapse is architectural, not a training-length issue)
4. That extension outcome does NOT change the scientific conclusion if V4/V5 still fail —
   the result remains NONE_VIABLE unless BOTH pass

**Estimated GPU runtime:** L2 only, 1,000 steps, T4 GPU, 3.36 steps/s measured throughput.
Additional 700 steps = ~208 s ≈ **3.5 minutes** of GPU time.
Full fresh 1,000-step run = ~298 s ≈ **5 minutes** of GPU time.
Well within one Kaggle session.

**If extension passes V4+V5 for L2:** architecture = L2; proceed to FREEZE-B and RUN A.
**If extension still fails:** NONE_VIABLE confirmed at 1,000 steps; escalate to client.

### Preregistration deviation draft (for user approval)

```
DEVIATION_REQUEST_DR1
Date: 2026-10-09
Requested by: user
Status: EXECUTED (NOT VIABLE)

Deviation: Extend RUN_0 to 1,000 steps for Variant L2 only.

Justification:
  - V3 PASS for L2 (alpha does not collapse; architecture is structurally sound)
  - V4 failure is by margin: 18/54 labels have logit std >0.01; threshold is 28/54
  - V5 failure: 63/200 = prior; model has learned 0 improvement above baseline after 4.46% of epoch 1
  - 300 steps is insufficient training signal for 54-label BERT fine-tuning
  - Variant L is NOT extended (V3 collapse is structural; alpha is not recoverable by more steps)

Constraints (unchanged):
  - Seed = 42, same val IDs, same label order, same BERT checkpoint
  - Decision criteria V1-V5 unchanged; thresholds unchanged
  - Test split is NOT loaded or predicted during extension
  - Extension run is identified as RUN_0_P1F_EXT1; does not overwrite RUN_0_P1F
  - If V4+V5 still fail at 1,000 steps: result = NONE_VIABLE; no further extension

Approval required: YES (APPROVED by user on 2026-10-10)
Outcome: V5 FAIL (Top-1 hits 63/200 == prior 63/200). NOT VIABLE. Data anomaly discovered (duplicate columns cmp-lg == cond-mat.dis-nn).
```

### H. Decision Registry (Pending / Recent)

```text
DEVIATION_REQUEST_DR2
Date: 2026-10-10
Requested by: user
Status: APPROVED

Deviation: Run a standard BERT multi-label control baseline (Variant B: [CLS] -> Linear(54)) using BCEWithLogitsLoss.

Constraints:
- 15 epochs, same official BERT weights, AAPD splits, seed 42.
- No test split access.
- Does not claim to be a literal LHRM reproduction.
- Sigmoid + threshold 0.5 for metrics.
```

### Next executable action (approved)
- **Action**: Implement Variant B in `model.py` and `run_baseline.py`.
- **Target**: Prepare DR2 Kaggle notebook and bundle.
- **Constraints**: No local GPU training. No changes to historic artifacts.

---

## RUN A Readiness Fixes (2026-10-09)

All defects identified in the Phase 1F audit were implemented and verified.

| Defect | Description | Fix | Test |
|---|---|---|---|
| **A1** | Test split loaded during training | `cmd_train` never loads test; `cmd_test_eval` is a separate authorized command | `test_a1_cmd_train_does_not_load_test_split` |
| **A2** | No per-step progress logging | `train_epoch_stepped` emits `[TRAIN]` every 100 steps to stdout + JSONL | `test_a2_progress_jsonl_written` |
| **A4** | Incomplete RNG restore on resume | `save/load_checkpoint` now saves/restores Python random RNG; atomic write via tmp+rename | `test_a4_checkpoint_save_load_epoch_boundary` |
| **A7** | Reuters hardcoded label fallback | `_resolve_label_order()` helper: pred-dir → resolved_config → AAPD canonical | — (structural) |
| **A8** | No F1 metrics | `compute_f1_metrics`, `compute_per_label_metrics`, `compute_all_metrics` with `include_f1=True` | `test_a8_*` |
| **A9** | 15 per-epoch checkpoints = 15×1.3 GB | Rolling: `latest.pt` + `previous.pt` only; `best_dev_model_only.pt` (model state only) | `test_a9_rolling_checkpoint_rotation` |

### New files
- `tests/test_run_a_readiness.py` — 13 CPU-only tests for A1/A2/A4/A8/A9
- `notebooks/lhrm_run_a_kaggle.ipynb` — 6-cell production Kaggle notebook with gates, live progress, separate test eval cell

### Training outputs (per epoch)
- `train.log` — JSONL epoch-complete records
- `train_progress.jsonl` — step-level JSONL + epoch-complete (screenshot evidence)
- `epoch_metrics.csv` — CSV with P@1/P@3/P@5/nDCG@3/nDCG@5/micro_f1/macro_f1 per epoch
- `resolved_config.json` — provenance (includes `_test_split_loaded=False`)
- `checkpoints/latest.pt`, `checkpoints/previous.pt`, `checkpoints/final_epoch.pt`
- `predictions_val.json` — dev predictions from final epoch
- `val_per_label_metrics.json` — per-label F1/TP/FP/FN on dev

---

## Files changed in Phase 1F / RUN A Readiness

| File | Change |
|---|---|
| `run_baseline.py` | Phase 1F: AAPD routing, preflight, resolved_config |
| `run_baseline.py` | **RUN A: A1** `cmd_train` test isolation; `cmd_test_eval` new gate command |
| `run_baseline.py` | **RUN A: A5** data_dir case-variant resolution + absolute path logging |
| `run_baseline.py` | **RUN A: A7** `_resolve_label_order()` helper; dataset-aware for all eval/HRRM cmds |
| `run_baseline.py` | **RUN A: A9** rolling checkpoints (latest.pt + previous.pt); no per-epoch files |
| `run_baseline.py` | **RUN A** `--variant` arg on `train`; variant=TBD guard; full provenance banner |
| `src/lhrm_baseline/train.py` | **RUN A: A2** `train_epoch_stepped` periodic JSONL + stdout progress every 100 steps |
| `src/lhrm_baseline/train.py` | **RUN A: A4** `save_checkpoint_v2` saves python random RNG; atomic tmp+rename write |
| `src/lhrm_baseline/train.py` | **RUN A: A4** `load_checkpoint` restores CUDA RNG + python random RNG; `weights_only=False` |
| `src/lhrm_baseline/metrics.py` | **RUN A: A8** `compute_f1_metrics`, `compute_per_label_metrics`; `compute_all_metrics` with F1 |
| `src/lhrm_baseline/diagnostics.py` | Phase 1F: Source of truth for V1-V5 |
| `src/lhrm_baseline/model.py` | Phase 1F: CSAM_L returns alpha in eval mode |
| `tests/test_run_a_readiness.py` | **NEW** — 13 CPU tests: A1/A2/A4/A8/A9 |
| `notebooks/lhrm_run_a_kaggle.ipynb` | **NEW** — 6-cell Kaggle notebook with gates + separate test_eval cell |
| `notebooks/lhrm_run0_kaggle_cli.ipynb` | Phase 1F: CLI-driven RUN_0 notebook |
| `docs/phase1/run0_analysis.md` | Phase 1F: Verified RUN_0_P1F results |
| `docs/CHATGPT_PROJECT_CHECKPOINT.md` | Updated: gate table, RUN A readiness section |
| `experiments/RUN_0_P1F/ARTIFACT_POINTER.json` | Phase 1F: Pointer to external archive |
