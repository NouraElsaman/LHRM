# RUN_0 Audit Report — Phase 1F

**Date:** 2026-10-08
**Classification:** CASE B (diagnostic implementation bug) + CASE C (numerical evaluation bug)
**Conclusion:** The NONE_VIABLE / STOP_AND_ASK result is not accepted or rejected. It is invalid and requires a clean rerun after the fixes below.

---

## Status at time of audit

```
RUN_0_STATUS              = INVALID_DIAGNOSTICS
DIAGNOSTIC_VALIDITY       = INVALID
V1 = true  (training log trusted; 0 NaN/Inf in 300 steps)
V2 = true  (loss fell; training log trusted — loss fell ~15% per training log)
V3 = UNMEASURED  (alpha tensor never returned; 0<0.9 vacuously true — CASE B)
V4 = UNRELIABLE  (fp16 measurement — CASE C)
V5 = UNRELIABLE  (L false; L2 float32 tie, not a valid pass — CASE C)
```

---

## F1 — Alpha diagnostics were never computed

**Classification:** CASE B (implementation bug)
**Evidence:**

`src/lhrm_baseline/model.py` lines 107-111 (pre-fix):
```python
diag = {
    "alpha_max_mean": max_alpha,
    "alpha_entropy_mean": alpha_entropy,
    "alpha_one_hot_frac": one_hot_frac,   # scalar only — no tensor
}
```
The `alpha` tensor (`[B, k]`) was computed at line 78 but never placed in `diag`.

`run_baseline.py` lines 527-547 (pre-fix, `_val_diagnostics`):
```python
alpha = diag.get("alpha")       # line 527 — always None
if alpha is not None:
    all_alphas.append(...)      # never reached
else:
    one_hot_frac = 0.0          # line 545
    mean_max_alpha = 0.0        # line 546 — impossible for softmax
    argmax_true = 0.0           # line 547 — ESCALATE can never fire
```

Notebook cell 9 had the identical reader bug.

**Consequence:**
- `mean_max_alpha = 0.0` is impossible for a valid softmax over 54 labels (minimum is 1/54 ≈ 0.0185).
- `alpha_one_hot_frac = 0.0`, so `0.0 < 0.9` → **V3 = True vacuously** — not a measurement of architecture behaviour.
- `argmax_alpha_is_true_frac = 0.0` → the ESCALATE condition (`> 0.5`) can never fire, hiding the escalation path.
- The unit tests only checked that aggregate keys exist (`alpha_max_mean` etc.) — never that the reader received the tensor.

**Fix:**
- `CSAM_L.forward` now returns `"alpha": alpha.detach()` and `"S": S.detach()` in `diag` when `model.training == False` (eval mode) or `return_alpha=True`.
- `diagnostics.compute_diagnostics` requires `diag["alpha"]` and raises `DiagnosticError` if it is `None` — fail closed.
- `_validate_alpha` checks: shape, NaN/Inf, rows sum to 1±1e-3, mean_max ≥ 1/k.

---

## F2 — V5 for L2 is a floating-point artefact; the true result is a tie

**Classification:** CASE C (numerical evaluation bug)
**Evidence:**

`run_baseline.py` line 315-316 (pre-fix):
```python
label_freq = train_labels.sum(axis=0) / max(len(train_labels), 1)
top1_label  = int(label_freq.argmax())   # column-order tie-break (wrong)
lf_p1 = float(v_labels[:, top1_label].mean())  # float32 array → 0.3149999976158142
```

L2's P@1 was reported as `0.315` (Python float64 = 63/200 exactly).
The comparison `0.315 > 0.3149999976…` is True by float32 representation error.
Both are mathematically 63/200 = 0.315. Under the preregistered *strict* `>`, L2's V5 is **False** (a tie).

Additionally, `label_freq.argmax()` uses column-order tie-breaking (whichever duplicate label appears first), not the preregistered seeded random tie-break (A16/A18).

**Consequence:**
- L2's V5 = True was a float representation artefact.
- L2 did not beat the prior; its top-1 prediction matches the prior's top-1 label for effectively every document.

**Fix:**
- `diagnostics.frequency_prior_hits_at_1` computes prior hits using **integer counts** and seeded random tie-breaking (`np.random.default_rng(seed)`).
- V5 is compared as `int(cand_hits) > int(prior_hits)` — float representation cannot affect it.

---

## F3 — V4 (and P@1) measured under fp16 autocast

**Classification:** CASE C (numerical evaluation bug)
**Evidence:**

`run_baseline.py` lines 524-526 (pre-fix, `_val_diagnostics`):
```python
ctx = torch.amp.autocast("cuda") if use_amp else contextlib.nullcontext()
with ctx:
    logits, diag = forward_pass(model, ids, msk, label_ids, label_mask)
```

`use_amp=True` in all runs. Logits were therefore fp16 (spacing ≈ 5×10⁻⁴ at |logit| ∼ 0.5).
The V4 threshold is `per_label_logit_std > 0.01`. With fp16 quantisation a true variation of
order 1e-3 becomes quantised zero or noise, making V4 unreliable.

**Fix:**
- `compute_diagnostics` runs the forward with `contextlib.nullcontext()` — no autocast.
- Logits are cast to fp32 explicitly: `logits.detach().float().cpu().numpy()`.
- The AMP setting for training is recorded in the output as `amp_for_training` (audit only).

---

## F4 — Analytic expectation (hypothesis; not a conclusion)

The following is a scientific observation from the measurements. It is **not asserted as a result** until the corrected diagnostics confirm it.

By the architecture:
- `logit_i = W · (α_i · h_{L_i} + Σ_j A_{ij} · h_{L_j}) + b`
- The second term `Σ_j A_{ij} · h_{L_j}` does not depend on the document (all docs see the same label embeddings; A is constant for a given label set).
- For variant **L** (unscaled `S = h_T · h_L`): BERT [CLS] vectors have norm ≈ 8–15 and high mutual cosine at initialisation, causing `S` scores to be large and similar, so softmax(S) is already peaked (near one-hot).
- For variant **L2** (scaled `S / √768`): dot-products are small and similar at init, so α ≈ uniform, with minimal document dependence.
- Both pathways can produce very little **document-dependent** logit variation, explaining low V4 and P@1 near the prior.

The training loss fell ≈ 15% (L: 0.0206 → 0.0176; L2 similar per training log), indicating the shared `W` and bias `b` are being updated toward a marginal (document-independent) prediction. This is consistent with learning the label co-occurrence structure rather than document-label relevance.

**These observations are informational.** The corrected rerun will produce the actual α distributions, S-score distributions, and per-label logit-std arrays that confirm or refute this hypothesis.

---

## F5 — Incomplete artifact set

**Evidence:** `SHA256SUMS.txt` lists four checkpoint files (step_150.pt, step_300.pt for L and L2) that were not in the returned ZIP. Per-variant `step0_diagnostics.json`, `step300_diagnostics.json`, `val_sample_predictions.json` were not produced (because `_val_diagnostics` only returns aggregates, not full-vector results). `resolved_config.json` contains `data_manifest_sha256: "PENDING"`.

**Fix:** `compute_diagnostics` returns `per_label_logit_std` as a full list (length k), saved to `step*_diagnostics.json`. `SHA256SUMS.txt` is written after all files exist and lists only present files.

---

## F6 — Duplicated diagnostic logic

**Evidence:** Notebook cell 9 (`val_diagnostics`) and `run_baseline.py::_val_diagnostics` were independent copies with identical bugs.

**Fix:** `src/lhrm_baseline/diagnostics.py` is the single implementation. `run_baseline.py` calls `_val_diagnostics_v2` which is a thin wrapper around `compute_diagnostics`. The notebook calls `run_baseline.py run0` or imports from `lhrm_baseline.diagnostics` directly — no copied logic.

---

## F7 — Registry/test hygiene

**Status:** Documented here; corrections applied in registry v5 (separate commit).

- A2: `"h_{L_i} = BERT([CLS], y_i, [SEP])"` is not in the paper; correct form is `h_CLS^{L_i} = BERT_CLS(Y_i)` (Eq.2). Changed to a structural description (not falsely verbatim).
- A7: "same pre-trained BERT encoder" is not a paper quote; marked as ASSUMED.
- A9: Section reference corrected from "III-C" to "IV-E" (focal loss is in the experiments section).
- A5b: Reconciled to one definition: `W ∈ ℝ^H` (shared scalar linear layer per Eq.8/Figure 2 caption). "Linear layer on the similarity score S_i" was incorrect — `W` acts on `h_final^(i) ∈ ℝ^H`, not on the scalar `S_i`.
- A13: `n_iter_no_change` corrected from 10 to `None` (no early stopping in code). Ambiguity noted: the paper's validation-fraction and tolerance parameters are listed but have no effect when early stopping is disabled. Recorded as NEA (not experimentally available).

---

## Changes made (Phase 1F)

| File | Change |
|---|---|
| `src/lhrm_baseline/model.py` | `CSAM_L.forward` returns `alpha` and `S` tensors in eval mode |
| `src/lhrm_baseline/diagnostics.py` | New — single source of truth for all RUN_0 diagnostics |
| `run_baseline.py` | Replaced `_val_diagnostics` + `_run0_decision` with wrappers; fixed prior |
| `notebooks/lhrm_run0_aapd_gpu.ipynb` | Thin wrapper — calls `lhrm_baseline.diagnostics` |
| `tests/test_diagnostics.py` | 15 new regression tests |
| `docs/phase1/decision_registry_v5.md` | Registry corrections (F7) |
| `docs/phase1/pre_training_checklist_v3.md` | Addendum noting Phase 1F fixes |

---

---

## Pre-run cross-check (Section 4) — COMPLETED

**File:** `docs/phase1/run0_local_step0_crosscheck.json`
**Completed:** 2026-10-08, CPU fp32, seed 42, 200 val doc IDs from first Kaggle run.
**Label hash:** `a3d662f3920259a94ed61d83f98f21cabe7f908d98a5662ccd9adc2413cc000b` (matches freeze).

**Reference values for GPU agreement check (±tolerance):**

| Metric | Variant L | Variant L2 | Tolerance |
|---|---|---|---|
| `mean_max_alpha` | 0.7608 | 0.0297 | ±0.001 |
| `alpha_one_hot_frac` | 0.41 | 0.00 | exact |
| `alpha_entropy_mean` | 0.714 | 3.964 | ±0.01 |
| `S_mean` | 167.28 | 6.04 | ±0.5 |
| `S_std` | 14.47 | 0.52 | ±0.1 |
| `prior_p_at_1_hits` | 63 | 63 | exact |
| `n_distinct_argmax` | 19 | 19 | ±2 |

After the corrected Kaggle run, the GPU step-0 values **must** agree with these within tolerance before
the step-300 measurements are trusted. If any required field disagrees → flag as cross-check failure.

---

## Section 5 — Cross-check findings (observations, not conclusions)

The following observations are derived from the CPU step-0 measurements. They are
**informational** — not assertions about the step-300 outcome. The step-300 corrected GPU
measurements decide all viability questions.

### 5a. Variant L — Structural alpha saturation at initialization

`S_mean = 167.3` (±14.5 std). The unscaled dot product `S_i = h_T · h_{L_i}` between
BERT [CLS] document and label representations at random initialization is ~167.
For context, with BERT embeddings of norm ~15 and ~54 labels, the unnormalized dot products
will be of this magnitude. Softmax over 54 values near 167 collapses to a near-one-hot
distribution over the most similar label, with remaining mass concentrated on a few others.

Consequence at step 0:
- `alpha_one_hot_frac = 0.41` — 41% of 200 val docs already have max(alpha) > 0.9
- `alpha_entropy_mean = 0.714` — very low entropy (vs theoretical max ln(54) = 3.989)
- `n_distinct_argmax = 19` out of 54 labels receive attention from any document

This is **structural** (determined by BERT embedding scale at init, not by training). Training
on 300 steps with a fixed BERT encoder may worsen saturation (the W parameter has no mechanism
to de-saturate alpha). If V3 fails for L at step 300, the ESCALATE check will determine
whether the argmax-alpha label is at least task-relevant (argmax_alpha_is_true_frac > 0.5).

At step 0, `argmax_alpha_is_true_frac = 0.025` — only 2.5% of docs have their argmax-alpha
pointing to a true label. This is far below the ESCALATE threshold (0.5). Whether training
improves this is determined by the corrected GPU step-300 diagnostics.

### 5b. Variant L2 — Near-uniform alpha at initialization

`S_mean = 6.04` (±0.52 std). Scaling S by 1/√768 ≈ 0.036 reduces the dot products to a
range where softmax produces a near-uniform distribution.

Consequence at step 0:
- `alpha_one_hot_frac = 0.000` — no saturation at all
- `alpha_entropy_mean = 3.964` — near-maximum (ln(54) = 3.989)
- `logit_std_frac = 0.000` — all per-label logit stds are below 0.01 (range: 0.0004–0.0022)

The per-label logit stds for L2 at step 0 are all in the 0.0004–0.0022 range — far below
the V4 threshold of 0.01. Whether training on 300 steps generates sufficient logit variation
(> 0.01 for > 50% of labels) to pass V4 is an open question.

### 5c. V5 context

Both variants have `P@1 hits = 0–1/200` at step 0 (random init, expected).
Prior hits = 63/200. Any trained model that produces at least 64/200 correct top-1 predictions
on the val sample beats the prior (integer, strict >). Whether either variant achieves this
at step 300 is decided by the corrected GPU measurements.

### 5d. No fabricated predictions

The above are observations from CPU fp32 measurements at step 0 (random init). No claims
are made about step-300 outcomes. These observations are recorded to:
1. Establish GPU-CPU cross-check agreement at step 0 (before claiming step-300 values are valid)
2. Document the F4 architectural hypothesis with measured evidence
3. Guide interpretation of the step-300 `alpha_one_hot_frac` once the corrected run returns

---

## What the clean rerun will resolve

The corrected GPU rerun will produce (for both L and L2 at step 300, fp32):
1. **Actual α distribution** — did training change one_hot_frac from 0.41 (L) or 0.00 (L2)?
2. **Per-label logit-std vector** — what fraction exceed 0.01 in fp32? (V4 — previously unreliable)
3. **Integer P@1 hit count vs prior** — is the candidate genuinely above 63/200? (V5 — previously artefact)
4. **ESCALATE check** — real argmax_alpha_is_true_frac at step 300 (not 2.5% as at step 0)
5. **V2 confirmation** — loss must have fallen from step-0 to step-300 range (cross-checked with training log)

**GPU-CPU step-0 agreement must be verified first.** If `mean_max_alpha` for L disagrees by
more than 0.001, or `alpha_one_hot_frac` disagrees at all, the run is flagged and not accepted
until the discrepancy is explained.

The scientific outcome (DECIDED/EXTEND/ESCALATE/NONE_VIABLE) is decided only by the
corrected, preregistered V1-V5 rules applied to the GPU step-300 measurements.
