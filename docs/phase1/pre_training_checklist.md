# Pre-Training Checklist v2

**Date:** 2026-10-06
**Status:** NOT_PASSED (blocking items open; G5 may not start)

---

## Grading: PASS | FAIL/OPEN | PENDING (blocked)

---

## Section 1: Code Integrity

| ID | Item | Status | Evidence |
|---|---|---|---|
| C1 | All src/ files syntax-valid | PASS | ast.parse all *.py |
| C2 | prepare_aapd.py --help exits 0 | PASS | test_script_help_exits_0 |
| C3 | prepare_reuters.py --help exits 0 | PASS | test_script_help_exits_0 |
| C4 | run_baseline.py --help exits 0 | PASS | test_script_help_exits_0 |
| C5 | Model/lhrm.py unmodified | PASS | git diff confirms no changes |
| C6 | legacy/lhrm_original.py unmodified | PASS | git diff confirms no changes |
| C7 | All unit tests pass | PASS (pending S4 rerun post-D8) | see below |
| C8 | F8: criterion='friedman_mse' asserted | PASS | test_hrrm_explicit_gb_config |
| C9 | Tie-breaking (N6/A18): seeded-random in hrrm.py + metrics.py | PASS | N6 fix committed |
| C10 | Dynamic padding (N7): collate_fn pads to batch max | PASS | N7 fix committed |

---

## Section 2: Data Integrity

| ID | Item | Status | Evidence |
|---|---|---|---|
| D1 | AAPD train=53840, dev=1000, test=1000 | PASS | Verified from Zenodo |
| D2 | Reuters test=3019; earn total=3964 | PASS | S3 NLTK join |
| D3 | AAPD label mapping | OPEN/BLOCKED | UNRESOLVED_LABEL_MAPPING; arXiv evidence of mis-assignment |
| D4 | Reuters 90 labels match defs (exact case) | PASS | S3 verified |
| D5 | AAPD processed arrays loadable | PASS | DATA_MANIFEST.json |
| D6 | Reuters D8 fix: A_R1b implemented | PASS | prepare_reuters.py A_R1b; train=6910,val=767,test=3019 |
| D7 | Zero-label docs: AAPD=0, Reuters=0 | PASS | Both prepare scripts |
| D8 | Reuters train/dev/test text disjoint | PASS | test_reuters_splits_text_disjoint (after A_R1b) |
| D9 | AAPD label texts <= 64 tokens | PASS | test_label_texts_token_count_aapd |
| D10 | Reuters label texts built (90) | PASS | prepare_reuters.py output |
| D11 | Label-order hash stable (AAPD) | PASS | test_aapd_label_order_hash_stable |
| D12 | Label-order hash stable (Reuters) | PASS | test_reuters_label_order_hash_stable |

---

## Section 3: Configuration

| ID | Item | Status | Evidence |
|---|---|---|---|
| CF1 | AAPD max_len = 512 (p99 = 385 tokens) | PASS | token_length_stats.json |
| CF2 | Reuters max_len | OPEN | Token-length measurement not yet run |
| CF3 | aapd_baseline.yaml consistent | PASS | Checked manually |
| CF4 | reuters_baseline.yaml max_len | OPEN | Blocked on CF2 |
| CF5 | bert-base-uncased in cache | PASS | Confirmed |
| CF6 | Architecture variant chosen | OPEN | Blocked on smoke run (FREEZE-A first) |
| CF7 | FREEZE-A written | OPEN | Blocked on CF2 |
| CF8 | FREEZE-B written | PENDING | Blocked on CF6/CF7 |

---

## Section 4: Decision Registry

| ID | Item | Status | Evidence |
|---|---|---|---|
| DR1 | All A1-A20+A_R1b in registry v3 | PASS | decision_registry.md v3 |
| DR2 | No PBC items open | PASS | None in v3 |
| DR3 | A_R1b registered (Reuters resplit) | PASS | Registry v3 |
| DR4 | A18 tie-breaking implemented + tested | PASS | N6 fix + tests |
| DR5 | A9 focal-loss rationale corrected (N5) | PASS | Registry v3 A9 |
| DR6 | A7/A8 equation errors corrected (N4) | PASS | Registry v3; arch screening v2 |

---

## Section 5: Protocol Controls

| ID | Item | Status | Evidence |
|---|---|---|---|
| P1 | No test-set text used in any decision | PASS | All decisions pre-registered |
| P2 | HRRM fits on val predictions only (A17) | PASS | HRRM API test |
| P3 | best.pt diagnostic only, not reported | PASS | preregistration.md |
| P4 | Focal params not tuned by validation | PASS | A9: gamma=2,alpha=0.25 from Lin et al. |
| P5 | Architecture by V1-V5 (non-performance) | OPEN | Smoke run pending |
| P6 | BCE control planned | PASS | A20 registered |

---

## BLOCKING ITEMS (must be PASS before G5):

1. OPEN D3: AAPD label mapping (UNRESOLVED -- AAPD training blocked)
2. OPEN CF2: Reuters token-length measurement
3. OPEN CF4: reuters_baseline.yaml max_len
4. OPEN CF7: FREEZE-A

**G4 status: NOT_PASSED. G5 may not start until all blocking items are PASS.**
**Reuters-only path: G5 is unblocked for Reuters if CF2/CF4/CF7 are closed.**
