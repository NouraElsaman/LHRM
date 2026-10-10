# Phase 1F — Final Status

**Updated:** 2026-10-08

## STATUS BLOCK

```
ENGINEERING_PHASE1           = IN_PROGRESS
  F1 (alpha tensor returned)  = CLOSED (model.py + diagnostics.py + 3 tests)
  F2 (integer prior, tie-break) = CLOSED (diagnostics.py + 2 tests)
  F3 (fp32 no autocast)       = CLOSED (diagnostics.py + 1 test)
  F6 (deduplicated logic)     = CLOSED (single diagnostics.py; notebook cells 9/16 patched)
  F7 (registry corrections)   = CLOSED (decision_registry_v5.md)
  test_diagnostics.py         = 16/16 PASSED
  Full test suite              = 73 passed, 2 skipped (network tests)

RUN_0_STATUS                 = AWAITING_CORRECTED_RERUN
  Bundle (Phase 1F)           = BUILT
    Path: dist/lhrm_run0_bundle_p1f.zip/lhrm_gpu_bundle_20261008_135414.zip
    SHA-256: 4b3898c8096a20985ad6c0cb357ef671dfa4b8a43d53c42de058eb255f3b3753
    Files: 20 | Size: 66.9 KB
  Local step-0 cross-check    = RUNNING (CPU, BERT forward, ~10 min)
```

## Gate Summary (evidence-based)

| Gate | Status | Evidence |
|---|---|---|
| G0 (code faithful to paper) | PASS | Eq.4-8 verified, A-assumptions documented |
| G1 (data provenance) | PASS | Zenodo URL, SHA256, label hash in freeze |
| G2 (metric implementation) | PASS | P@k seeded, integer prior, test_metrics 100% |
| G3 (infrastructure) | PASS | AMP, collator, seeds, checkpoints |
| G4 (pre-run checklist) | PASS | freeze written, val IDs saved, test split locked |
| RUN_0 diagnostics | AWAITING | F1-F3 bugs closed; corrected rerun needed |

## Key files changed in Phase 1F

| File | Change |
|---|---|
| `src/lhrm_baseline/model.py` | `CSAM_L.forward` returns `alpha` + `S` tensors in eval mode |
| `src/lhrm_baseline/diagnostics.py` | NEW — single source of truth for all RUN_0 diagnostics |
| `run_baseline.py` | `_val_diagnostics_v2` → `diagnostics.compute_diagnostics`; integer prior |
| `notebooks/lhrm_run0_aapd_gpu.ipynb` | Cells 9 + 16 = thin wrappers (no duplicate diagnostic code) |
| `tests/test_diagnostics.py` | NEW — 16 regression tests for F1-F3 + decision function + artifacts |
| `docs/phase1/run0_audit.md` | NEW — full evidence-based audit (F1-F7 with file/line references) |
| `scripts/run0_local_crosscheck.py` | NEW — CPU step-0 cross-check for GPU agreement |
| `scripts/patch_notebook.py` | NEW — notebook patch utility |

## Next action (user)

Upload `lhrm_gpu_bundle_20261008_135414.zip` to Kaggle and run the notebook.
Return the artifact JSON files as specified in `handoff2_status.md`.

The agent will then:
1. Import artifacts and validate SHA256SUMS.txt
2. Read `run0_decision.json` (status: DECIDED / EXTEND / ESCALATE / NONE_VIABLE)
3. Write `run0_analysis.md`
4. If DECIDED → proceed to Freeze-B + Run A (full AAPD baseline)
