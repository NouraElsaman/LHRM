# Phase 2 Candidates

Items that arose during Phase 1 that are DEFERRED (not in baseline).
Do not implement any of these until Phase 1 baseline is complete.

---

## PC-1: Focal Loss Parameter Tuning

Description: alpha_fl=0.25 under-weights positives in multi-label setting.
Tuning alpha toward 1.0 or using label-frequency-weighted alpha may improve results.
Why deferred: Would constitute PBC (tuning toward paper) during Phase 1.
Phase 2: Sweep alpha in {0.25, 0.5, 0.75} and gamma in {1, 2, 3}.

---

## PC-2: Head + Tail Truncation

Description: Reuters has 5.88% of docs > 512 tokens. Head+tail strategy retains
  first 128 and last 384 tokens instead of just first 512.
Why deferred: Changes preprocessing from the baseline; requires separate config.
Phase 2: Implement and compare to head-only (A10 baseline).

---

## PC-3: Sliding Window Encoding

Description: Encode long documents in overlapping 512-token windows and pool CLS.
Why deferred: Significantly more complex; not in paper.
Phase 2: If head-only truncation causes DIFFERENT verdict, investigate this.

---

## PC-4: Learning Rate Scheduling

Description: Paper does not mention LR schedule. Warmup + linear decay often helps.
Why deferred: Not in paper; adding would be NEA with potential impact.
Phase 2: Try linear warmup over 500 steps + linear decay.

---

## PC-5: Gradient Clipping

Description: Clipping at 1.0 is standard practice for BERT fine-tuning.
Why deferred: Not in paper.
Phase 2: Add max_norm=1.0 gradient clipping.

---

## PC-6: Alternative Label Text Formats

Description: Plain label name only (no definition), or definition only, or
  using the arXiv category hierarchy as context.
Why deferred: A4 choice is locked for Phase 1.
Phase 2: Ablate format choices.

---

## PC-7: AAPD Corrected Label Training

Description: If D3 is resolved via Option A, train LHRM on corrected AAPD labels.
Why deferred: D3 is unresolved.
Phase 2 (or Phase 1B for AAPD): Implement after Kaggle join confirms mapping.

---

## PC-8: BCE vs Focal Ablation

Description: A20 BCE control is part of Phase 1 (planned). Additional focal
  parameter ablations (PC-1) are Phase 2.
Phase 1: A20 (BCE control, same config, different loss only).
Phase 2: Focal parameter sweep.
