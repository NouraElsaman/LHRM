# Latest AG Status

**Updated:** 2026-10-06

---

```
ENGINEERING_PHASE1    = IN_PROGRESS
EXPERIMENTAL_BASELINE = NOT_STARTED
AAPD_BASELINE         = BLOCKED (UNRESOLVED_LABEL_MAPPING)
REUTERS_BASELINE      = READY_FOR_RUN_0 (pending architecture screening)

GATE SUMMARY:
  G0: PASS-with-open-items (AAPD label mapping blocked)
  G1: PASS (Reuters NLTK join verified; A_R1b implemented)
  G2: PASS (AAPD token lengths measured; Reuters max_len=512)
  G3: PASS (prepare_reuters.py A_R1b; D8 resolved; 53 tests pass)
  G4: PASS (all pre-training checklist blocking items resolved except D3)
  G5: UNSTARTED (awaiting FREEZE-B after RUN 0 architecture screening)

CURRENT_GATE          = G5 (Rotterdam; start RUN 0 when GPU available)
BLOCKING_ITEMS        = D3 (AAPD label mapping); CF6 (architecture -- post RUN 0)
```

---

## Changes Since Last Report (Phase 1C)

1. **N6 fixed:** Tie-breaking now seeded-random in both `metrics.py` and `hrrm.py`.
   Tests: `test_seeded_rank_order_*`, `test_metrics_label_permutation_invariance` — PASS.

2. **N7 fixed:** Dynamic padding implemented in `data.py` (`MultiLabelDataset` + `collate_fn`).
   Tests: `test_collate_fn_pads_to_batch_max`, `test_collate_fn_single_item` — PASS.

3. **D8 fixed:** Reuters Zenodo data had 31 train/dev + 4 train/test + 2 dev/test overlaps.
   Fixed by A_R1b: merge+dedup+remove_test_contamination+seeded_90/10_resplit.
   Final: train=6,910 / val=767 / test=3,019, all disjoint.
   Test: `test_reuters_splits_text_disjoint` — PASS.

4. **CF2/CF4 closed:** Reuters max_len=512 (A10-Reuters; 5.88% truncated at BERT limit).
   Updated: `reuters_baseline.yaml`, `a10_reuters_token_length.md`.

5. **CF7 closed:** FREEZE-A written at `docs/phase1/FREEZE_A.md`.

6. **S1 Route 4:** arXiv API join confirms systematic label mis-assignment in AAPD CSV.
   5/5 test docs matched; in 4/5 cases the column labels do NOT match true arXiv categories.
   Large join (300 docs) running in background — results will update S1 status.
   AAPD training remains BLOCKED.

7. **S3 verified:** Reuters NLTK join confirms same corpus as LHRM paper.
   earn=3,964 (matches paper), 90 labels exact-match, test=3,019 identical.

8. **N4/N5 corrected:** decision_registry.md v3; architecture_interpretation_screening.md v2.
   A7 (Eq 6 is label-label self-attention, not bare dot product) — CORRECTED.
   A8 (doc dependence via alpha only) — CORRECTED.
   A9 (focal-loss rationale) — CORRECTED.

9. **New tests:** test_s4_engineering.py (17 tests: N6, N7, HRRM guards, hash, disjoint,
   --help, smoke) — 16 PASS, 1 SKIP.

10. **Docs written:** decision_registry.md v3, architecture_interpretation_screening.md v2,
    aapd_label_mapping_verification.md (S1 final), reuters_data_verification.md (S3 verified),
    pre_training_checklist.md v2, preregistration.md, FREEZE_A.md.

---

## Next Steps (when GPU is available)

1. Run `run_baseline.py --config configs/reuters_baseline.yaml --run_id RUN_0 --dry_run`
   to check GPU availability and confirm forward pass.
2. RUN 0: Reuters 2-epoch smoke run; collect V1-V5 viability diagnostics.
3. Choose architecture variant (L or L2) by V1-V5 (non-performance).
4. Write FREEZE-B with architecture filled in.
5. RUN A: Full 10-epoch Reuters baseline.

## Open Items

- D3: AAPD label mapping (BLOCKED). S1 large join in progress.
- CF6: Architecture variant (pending RUN 0).
- CF8: FREEZE-B (pending CF6).
