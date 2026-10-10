# Architecture Interpretation Screening v2

**Date:** 2026-10-06
**Status:** PENDING (awaiting GPU smoke run on Reuters)
**Changes from v1:** N4 corrections applied (equation numbers, doc dependence analysis).

---

## Section 1: Paper Equations (Correct Transcription)

Reading directly from the paper (S3-A/B):

| Eq | Formula | Notes |
|---|---|---|
| 1 | h^T = BERT_CLS(X) | Document CLS embedding |
| 2 | h^L_i = BERT_CLS(Y_i) | Label i CLS embedding |
| 3 | [h^T; h^L_i] (concatenation) | Present in paper but UNUSED in Eqs 4-8 |
| 4 | S_i = h^T . h^L_i | Doc-label dot product (NO sqrt scaling) |
| 5 | alpha_i = softmax_j(S_j) | Attention weight of label i over docs |
| 6 | A_ij = softmax_j(h^L_i . (h^L_j)^T / sqrt(d)) | Label-label self-attention (WITH sqrt(d)) |
| 7 | h_final^(i) = alpha_i * h^L_i + sum_j A_ij * h^L_j | Per-label refined representation |
| 8 | y_hat_i = W . h_final^(i) + b | Shared scalar classifier |

**v1 ERRORS RETRACTED (N4):**
  v1 listed wrong equation numbering: "Eq.4: d=BERT_CLS(x); Eq.6: S_i=d.l_i;
  Eq.7: A=softmax(S); Eq.8: h_final = A_i*h_L + b_i"
  These were incorrect. The code was always correct; the document was wrong.

---

## Section 2: Analytic Derivation (Corrected N4)

Substituting Eq. 7 into Eq. 8 with linearity of W:

  logit_i = W . h_final^(i) + b
           = W . (alpha_i * h^L_i + sum_j A_ij * h^L_j) + b
           = alpha_i * (W.h^L_i) + sum_j A_ij * (W.h^L_j) + b     [linearity of W]
           = alpha_i * u_i + sum_j A_ij * u_j + b

  where u_i = W.h^L_i in R is a label-level scalar (document-independent).

**Structural consequences:**

1. A in R^(k,k) (Eq. 6) depends ONLY on label embeddings {h^L_j}.
   Given fixed label texts (fixed at training), A is constant across documents.
   It is a label-label contextual weighting.

2. Document dependence is ENTIRELY through alpha_i (which depends on h^T via Eq. 4/5).

3. Saturation risk (Variant L):
   Without scaling in Eq. 4: S_i = h^T . h^L_i in O(H) magnitude at initialization.
   For H=768: magnitudes can reach O(10-100) -> one-hot softmax -> alpha collapses.
   Diagnose with criterion V3 (fraction with max_alpha > 0.9).

4. Labels with alpha_i ~ 0 receive document-independent logit:
   logit_i ~= sum_j A_ij * u_j + b = constant across documents.
   Reduces to label-frequency prior -- not document-specific.

---

## Section 3: Variants Considered

### Variant L (Literal -- A6 primary candidate)
  Eq. 4: S = h_T @ h_L.T          (no scaling)
  Eq. 6: A = softmax(h_L @ h_L.T / sqrt(H))   (scaled)
  Risk: alpha one-hot if S magnitudes are large at init.

### Variant L2 (Scaled doc-label similarity -- A6 secondary)
  Eq. 4 scaled: S = h_T @ h_L.T / sqrt(H)
  NEA: not in paper for Eq. 4. Justified by standard transformer self-attention scaling.

### Variant R (Repository style -- fallback ONLY)
  Uses concat([h_T, agg]) -> Linear(2H, k). NOT paper-faithful.
  Requires user approval if both L and L2 fail viability.

---

## Section 4: Viability Criteria (non-performance based)

| ID | Criterion | Value |
|---|---|---|
| V1 | Loss is finite after 1 batch | Required |
| V2 | Loss decreasing over 300 steps | Required |
| V3 | alpha not one-hot: frac(max_alpha < 0.9) > 90% val docs | Required |
| V4 | Logits vary across documents: std > 0.01 | Required |
| V5 | P@1 > label-frequency prior on val | Sanity check only |

Decision rule: choose first viable of L -> L2. V5 is NOT used to prefer L2 over L.

---

## Section 5: Screening Dataset

Reuters (VERIFIED_REUTERS_PROTOCOL) is the screening dataset.
AAPD is BLOCKED (UNRESOLVED_LABEL_MAPPING) -- may only be used for DIAGNOSTIC V1-V4.

---

## Section 6: Screening Results

**Status:** PENDING (awaiting FREEZE-A and GPU smoke run)
