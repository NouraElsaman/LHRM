# State Reconciliation (Phase 1E)

**Date:** 2026-10-07
**Git commit:** 2a56bdb71c4847cb20d0a843813abaf99a2d9308
**Dirty:** True
**Purpose:** Reconcile current repo state after Phase 1C/1D partial work; set baseline for Phase 1E.

---

## 1. Git State

Latest commits:
  2a56bdb S1 large join COMPLETE: AAPD labels definitively UNRESOLVED
  0799b40 Phase 1C: close G1-G4, FREEZE-A, all N-findings resolved
  bb8dad8 G4: pre-training checklist, A_R1/A_R2, 37 tests pass
  a949248 G3: prepare_reuters.py, reuters_baseline.yaml, processed Reuters data
  aff27e3 G0/G1/G2/G3: Fix F1-F13, decision registry v2

Unstaged changes (dirty=True):
  M  Data/processed/Reuters/ (P4/P5 Reuters join and resplit from Phase 1D)
  ??  docs/phase1/decision_registry_v4.md (Phase 1D; not committed)
  ??  docs/phase1/reuters_data_verification_v2.md
  ??  docs/phase1/run0_preregistration.md (now overwritten for AAPD by Phase 1E)
  ??  docs/phase1/s1_purity_of_mapping.md
  ??  docs/phase2_candidates.md

Action: commit after Phase 1E preflight items complete; tree must be clean for FREEZE-B.

---

## 2. Test Suite (2026-10-07, pytest tests/)

53 passed, 1 skipped, 0 failed.

Per-file counts:
  test_data.py     : confirmed passing (data loading, label consistency)
  test_metrics.py  : confirmed passing (precision_at_k, ndcg, seeded tie-break)
  test_hrrm.py     : confirmed passing (F8 friedman_mse, save/load)
  test_model.py    : confirmed passing (CSAM_L, CSAM_L2 shapes)
  test_losses.py   : confirmed passing (focal, BCE)
  test_evaluate.py : confirmed passing (bootstrap CIs, evaluate_predictions)

Note: test_registry_quotes.py not yet written (P2 item; open).

---

## 3. Phase 1D Preflight Items (P1-P8) -- Status

| Item | Description | Status | Note |
|---|---|---|---|
| P1 | CLI: run0, import_artifacts, freeze subcommands; --run_id/--max_steps | DONE | written 2026-10-07 |
| P2 | Registry v4 integrity + quote tests | PARTIAL | registry_v4.md written; test_registry_quotes.py OPEN |
| P3 | CONFIG_FREEZE.json machine-checkable | DONE | configs/final/CONFIG_FREEZE.json stage A written |
| P4 | Reuters NLTK document-level join (full-text hash) | DONE (SECONDARY_PATH) | reuters_data_verification_v2.md written |
| P5 | Reuters revised dedup rule (A_R1b/P5) | DONE (SECONDARY_PATH) | Reuters data resplit |
| P6 | Reuters truncation rate at 512 | DONE (SECONDARY_PATH) | 5.88% recorded |
| P7 | Housekeeping: checklist v3, screening doc v3, clean tree | PARTIAL | checklist_v3 written; tree still dirty |
| P8 | S1 purity-of-mapping metric | DONE | s1_purity_of_mapping.md written |

---

## 4. Items Superseded by Phase 1E

| Superseded item | Reason |
|---|---|
| "Screen on Reuters first" (Phase 1D ordering) | Phase 1E: AAPD is the primary benchmark |
| arXiv recovery workflow (Route A/B) | Out of scope per client decision (A_D1) |
| Reuters-first ordering for G5/G6 | AAPD-first ordering; Reuters is SECONDARY_PATH |
| lhrm_s1_arxiv_join_cpu.ipynb | Not created; if it exists, mark DISABLED_OUT_OF_SCOPE |
| "AAPD_BASELINE = BLOCKED" | Lifted by A_D1; status is now READY_FOR_CLIENT_APPROVED_REPRODUCTION |
| reuters_data_verification.md (visual-inspection claims) | Replaced by v2 (SECONDARY_PATH) |

---

## 5. What Exists and Passes

Code:
  src/lhrm_baseline/ -- all modules present; 53/54 tests pass
  run_baseline.py -- 8 subcommands (prepare, run0, train, evaluate, hrrm_fit, hrrm_apply, import_artifacts, freeze)
  src/lhrm_baseline/train.py -- forward_pass, train_epoch_stepped, predict_epoch_loader, save_checkpoint_v2 added (C11 fix)

Data:
  Data/AAPD.zip -- present; SHA-256 recorded in CONFIG_FREEZE.json
  data/processed/AAPD/ -- present (train 53840, dev 1000, test 1000; 54 labels)
  data/processed/Reuters/ -- present (SECONDARY_PATH; P5 resplit applied but not committed)

Configs:
  configs/aapd_paper.yaml -- exists but max_len=256 (OLD; superseded by configs/final/aapd_baseline.yaml)
  configs/final/aapd_baseline.yaml -- WRITTEN NOW (max_len=512, 15 epochs, all Phase 1E values)
  configs/final/CONFIG_FREEZE.json -- WRITTEN NOW (stage A)
  configs/reuters_baseline.yaml -- exists (SECONDARY_PATH; max_len=512 already correct)

Docs:
  docs/phase1/decision_registry_v4.md -- written (A_D1 appended)
  docs/phase1/aapd_data_decision.md -- written
  docs/phase1/aapd_label_mapping_verification.md -- updated (ACCEPTED_BY_CLIENT_DECISION)
  docs/phase1/pre_training_checklist_v3.md -- written
  docs/phase1/run0_preregistration.md -- written (AAPD version)
  docs/phase1/reuters_data_verification_v2.md -- written (SECONDARY_PATH)
  docs/phase2_candidates.md -- written

---

## 6. Open Items Before HANDOFF 1

| Priority | Item | Who |
|---|---|---|
| HIGH | C7: confirm 53 tests still pass after train.py edits | agent |
| HIGH | C11: verify forward_pass/train_epoch_stepped imports cleanly | agent |
| HIGH | P2: test_registry_quotes.py (verbatim quote checks) | agent |
| HIGH | Step G: lhrm_run0_aapd_gpu.ipynb | agent |
| HIGH | Step G: scripts/make_gpu_bundle.py | agent |
| MED  | Commit all Phase 1E changes; verify dirty=False | agent |

ENGINEERING_PHASE1: IN_PROGRESS (Steps D-G remain).
