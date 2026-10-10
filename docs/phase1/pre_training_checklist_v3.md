# Pre-Training Checklist v3

**Date:** 2026-10-07
**Replaces:** v2
**Status:** IN_PROGRESS (blocking items: CF3, CF7, CF6, CF8; C11)

---

## Section 1: Code Integrity

| ID | Item | Status | Evidence |
|---|---|---|---|
| C1 | src/ files syntax-valid | PASS | ast.parse |
| C2 | CLI subcommands work | PASS | --help exits 0 for all 8 subcommands |
| C3 | run0/freeze/import_artifacts present | PASS | 2026-10-07 |
| C4 | train --run_id/--max_steps/--resume_from | PASS | Phase 1D |
| C5 | Model/lhrm.py unmodified | PASS | git diff clean |
| C6 | legacy/lhrm_original.py unmodified | PASS | git diff clean |
| C7 | All unit tests pass | PASS | 53 pass, 1 skip, 0 fail (2026-10-07) |
| C8 | criterion=friedman_mse asserted | PASS | test_hrrm_explicit_gb_config |
| C9 | Tie-breaking seeded-random (A16/A18) | PASS | N6 fix confirmed |
| C10 | Dynamic padding collate_fn | PASS | N7 fix confirmed |
| C11 | train_epoch_stepped exported from train.py | OPEN | needed for new train cmd |

---

## Section 2: AAPD Data Integrity (client-approved source)

| ID | Item | Status | Evidence |
|---|---|---|---|
| D1 | train=53840, dev=1000, test=1000 | True | aapd_data_decision.md |
| D2 | 54 labels; header order as supplied, unmodified | True | aapd_data_decision.md |
| D3 | Label mapping | ACCEPTED_BY_CLIENT_DECISION | A_D1; aapd_data_decision.md |
| D4 | All 54 label defs present (case-insensitive) | True | 0 missing |
| D5 | Label texts <= 64 tokens | True | aapd_data_decision.md |
| D6 | Zero-label docs = 0 in all splits | True | aapd_data_decision.md |
| D7 | Label-order hash stable | PASS | a3d662f3920259a9... |
| D8 | Cross-split duplicates reported | PASS | train/dev=4 train/test=5 |
| D9 | No pipeline step reorders/renames columns | PASS | load_aapd_split preserves CSV header order |

---

## Section 3: Configuration

| ID | Item | Status | Evidence |
|---|---|---|---|
| CF1 | AAPD max_len=512; dynamic padding | PASS | A10; token stats in aapd_data_decision.md |
| CF2 | Reuters max_len=512 | PASS (SECONDARY_PATH) | reuters_baseline.yaml |
| CF3 | configs/final/aapd_baseline.yaml | OPEN | Step E pending |
| CF4 | reuters_baseline.yaml | PASS (SECONDARY_PATH) | existing |
| CF5 | bert-base-uncased in local cache | PASS | confirmed |
| CF6 | Architecture variant chosen | OPEN | pending RUN_0 |
| CF7 | CONFIG_FREEZE.json stage A | OPEN | Step E pending |
| CF8 | FREEZE_B | OPEN | pending RUN_0 |

---

## Section 4: Protocol Controls

| ID | Item | Status | Evidence |
|---|---|---|---|
| P1 | No test-set text used in decisions | PASS | all decisions pre-registered |
| P2 | HRRM fits on dev predictions only (A17) | PASS | test_hrrm PASS |
| P3 | best.pt diagnostic only | PASS | preregistration.md |
| P4 | Focal params frozen; not tuned by val | PASS | A9: gamma=2, alpha=0.25 |
| P5 | Architecture by V1-V5 (non-performance) | OPEN | RUN_0 pending |
| P6 | BCE control planned (A20) | PASS | registered |

---

## Blocking Items Before G5 (RUN_0)

1. **C11**: train_epoch_stepped in train.py -- verify/add export
2. **CF3/CF7**: configs/final/aapd_baseline.yaml + CONFIG_FREEZE.json (Step E)
3. **CF6/CF8**: Architecture after RUN_0

**D3 is no longer blocking. AAPD: READY_FOR_CLIENT_APPROVED_REPRODUCTION**
