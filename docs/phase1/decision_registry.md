# Decision Registry — LHRM Baseline Reproduction
**Version:** 3.0
**Date:** 2026-10-06
**Changes from v2:** N4/N5 corrections (A7, A8, A9 rewritten); A_R1b added (Reuters dedup+resplit);
  A18 added (tie-breaking); all PBC items removed.

---

## Decision Classification Scheme

- **PAPER-EXPLICIT:** Directly stated with value in the paper
- **PAPER-IMPLIED:** Follows necessarily from paper constraints without ambiguity
- **REPOSITORY-DEFINED:** From the authors' code (not the paper text)
- **NECESSARY ENGINEERING ASSUMPTION (NEA):** Required to run; paper is silent; inert or risk-minimised
- **POTENTIAL BASELINE CONTAMINATION (PBC):** Tuned toward paper results — FORBIDDEN

---

## A1: CLS Embedding Extraction

**Class:** PAPER-EXPLICIT
**Quote:** "we feed the document X into BERT and take the [CLS] representation h_T" (S3-A)
**Decision:** Use `last_hidden_state[:, 0, :]` (BERT CLS token from final layer)
**NOT:** `pooler_output` (a separate learned tanh projection)
**Inert:** No — pooler_output is different in value

---

## A3: Shared BERT for documents and labels

**Class:** PAPER-IMPLIED
**Evidence:** Eq. 1 and Eq. 2 both denote BERT_CLS(.) with no distinction.
**Decision:** Single shared BertModel for documents and labels.
**Inert:** No — separate BERTs would have >2x parameters.

---

## A5: W shape (Eq. 8 — scalar output)

**Class:** NECESSARY ENGINEERING ASSUMPTION
**Paper Eq. 8:** y_hat_i = W * h_final^(i) + b
  h_final^(i) in R^H, y_hat_i in R (scalar)
  Therefore W in R^(1 x H) (a single shared weight vector)
**Decision:** nn.Linear(H, 1, bias=True) shared across all k labels
**A5b:** Figure 2 could also be read as W in R^(k x H); requires user approval if A5 fails.

---

## A6: Architecture Variant Selection (L vs L2)

**Class:** NECESSARY ENGINEERING ASSUMPTION
**Paper Eq. 4:** S_i = h^T . h^L_i (bare dot product, no 1/sqrt(d))
**Paper Eq. 6:** A_ij = softmax_j(h^L_i . (h^L_j)^T / sqrt(d)) (label-label, scaled)
**Variant L:** Implements paper literally (Eq. 4 unscaled, Eq. 6 scaled)
**Variant L2:** Scales Eq. 4 by 1/sqrt(d) as well (stabilises alpha)
**Decision rule:** V1-V5 viability criteria on Reuters screening run; choose first viable: L -> L2.
**Status:** PENDING smoke run.

---

## A7: No learned projections in label-label attention (Eq. 6)

**Class:** PAPER-EXPLICIT
**CORRECTED from v2 (N4):** Previous A7 text was WRONG.
  Previous text said "Eq. 6 is a bare dot product d.l_i" — this is INCORRECT.

  Correct equations from paper:
    Eq. 4: S_i = h^T . h^L_i                               (doc-label dot product)
    Eq. 5: alpha_i = softmax_j(S_j)                        (attention over labels)
    Eq. 6: A_ij = softmax_j(h^L_i . (h^L_j)^T / sqrt(d))  (label-label self-attention)
    Eq. 7: h_final^(i) = alpha_i * h^L_i + sum_j A_ij * h^L_j
    Eq. 8: y_hat_i = W . h_final^(i) + b

**Decision:** Eq. 6 is label-label scaled dot-product self-attention WITHOUT
  learned Q/K/V projection matrices. This is exactly what the code implements.
  Code: `A_raw = (h_L @ h_L.T) / math.sqrt(H)` — CORRECT.
**Inert:** Yes — the paper explicitly has no projections in Eq. 6.

---

## A8: Document dependence via alpha only (Eq. 7 analysis)

**Class:** PAPER-IMPLIED
**CORRECTED from v2 (N4):**
  Previous A8 listed wrong equation numbers. Correct analysis:

  Substituting Eq. 7 into Eq. 8 with linearity of W:
    logit_i = W . h_final^(i) + b
             = W . (alpha_i * h^L_i + sum_j A_ij * h^L_j) + b
             = alpha_i * (W.h^L_i) + sum_j A_ij * (W.h^L_j) + b
             = alpha_i * u_i + sum_j A_ij * u_j + b

  where u_i = W.h^L_i is a label-level scalar (document-independent).

  Eq. 6 gives A in R^(k,k) computed from label embeddings alone.
  The document acts ONLY through alpha_i (via Eq. 4/5).
  Labels with alpha_i ~ 0 receive a nearly document-independent logit.
  This is the saturation risk that motivates Variant L2.

---

## A9: Focal Loss Parameters (gamma, alpha_fl)

**Class:** NECESSARY ENGINEERING ASSUMPTION
**CORRECTED from v2 (N5):**
  The LHRM paper does NOT specify gamma or alpha_fl values.
  Previous rationale was incorrect.

**Values used:** gamma = 2.0, alpha_fl = 0.25
**Source:** Lin et al. (2017) "Focal Loss for Dense Object Detection" — their defaults.
  These are NOT paper-specified.

**Corrected rationale:**
  With ~2.4 positives per 54 labels (AAPD):
  - alpha_fl = 0.25 weights positives LESS than negatives. Counter-intuitive for
    class imbalance where positives are rare.
  - gamma = 2.0 suppresses easy-to-classify examples (mostly easy negatives at
    convergence). The focal term (1-p_t)^2 down-weights examples already well-classified.
  - Combined effect: focuses on hard positives and hard negatives.
  - alpha_fl = 0.25 may under-weight the rare positive class; closer to 1.0 might
    better suit LHRM's multi-label setting. However, changing from Lin et al. defaults
    risks introducing a non-paper choice.

**Inert:** No — focal loss parameters affect training dynamics and final metrics.
**Report as:** ASSUMED in all result tables. BCE control (A20) is the counterfactual.

---

## A10: max_len per dataset

**Class:** NECESSARY ENGINEERING ASSUMPTION
**Rule:** Smallest of {128, 256, 512} covering >=99% of documents.
**AAPD:** max_len = 512 (train token-length p99 = 385)
  Dynamic padding: yes (collate_fn, N7 fix)
**Reuters:** PENDING token-length measurement.
**Inert:** Partial — truncating loses information; 512 covers 100% for AAPD.

---

## A12: Sigmoid of logits as HRRM input score

**Class:** PAPER-EXPLICIT
**Paper Eq. 9:** s_i = sigmoid(y_hat_i)
**Note:** GradientBoostingRegressor is rank-invariant to monotone transforms.
  Sigmoid vs logit produce same tree structure.
**Inert:** Yes.

---

## A13: GradientBoosting config

**Class:** PAPER-EXPLICIT for most parameters
**Quote:** "100 trees with learning rate 0.1, max depth 3" (S3-D, Table 3)
**Values:** n_estimators=100, lr=0.1, max_depth=3, subsample=1.0,
  criterion='friedman_mse' (PAPER-EXPLICIT; FutureWarning in sklearn>=1.9 but functional),
  loss='squared_error' (only valid choice in sklearn>=1.1).
**n_iter_no_change=None:** NEA (no early stopping; paper is silent).

---

## A15: Number of training epochs

**Class:** PAPER-EXPLICIT
**Quote:** "we set the number of epochs to 15 for AAPD and 10 for Reuters" (S4-B)
**Values:** AAPD: 15 epochs; Reuters: 10 epochs.

---

## A17: HRRM fitting data (non-leaky protocol)

**Class:** NECESSARY ENGINEERING ASSUMPTION
**Decision:** HRRM fits on validation-split predictions (non-leaky).
  The repository's code fits on test-set predictions (leaky; WRONG for reporting).
  Leaky version reproduced ONLY as LEAKY_DIAGNOSTIC, never headline.

---

## A18: Tie-breaking in ranking (N6 fix)

**Class:** NECESSARY ENGINEERING ASSUMPTION
**Decision:** Primary: raw score descending. Secondary: seeded-random permutation
  (seed = HRRM seed XOR doc_idx).
**NEVER:** column/label index order (biases toward low-index labels).
**Applied in:** metrics.py (P@k, nDCG@k), hrrm.py (rank feature, apply_to_predictions).
**Status:** IMPLEMENTED. Test: test_metrics_label_permutation_invariance PASS.

---

## A19: Final-epoch model for evaluation

**Class:** NECESSARY ENGINEERING ASSUMPTION
**Decision:** Use the model from the final epoch for test-set evaluation.
  best.pt checkpoint (best val metric) is DIAGNOSTIC ONLY.
  Reason: paper does not specify early stopping or best-model selection.

---

## A20: BCE control experiment

**Class:** NECESSARY ENGINEERING ASSUMPTION
**Decision:** Additional baseline with binary cross-entropy (identical config, same seed).
**Purpose:** Quantifies focal-loss contribution (only difference vs A9).

---

## A_R1b: Reuters Train/Dev Remerge + Dedup + Seeded Resplit

**Class:** NECESSARY ENGINEERING ASSUMPTION
**Background:**
  Zenodo Reuters has train=5,827 / dev=1,943 / test=3,019.
  NLTK ModApte has train=7,769 / test=3,019.
  Zenodo data quality issues found (D8):
    train/dev overlap: 31 docs
    train/test overlap: 4 docs
    dev/test overlap: 2 docs

**Decision:**
  1. Merge Zenodo train+dev (7,770 docs).
  2. Deduplicate by text: 7,770 -> 7,683 unique (87 dups removed).
  3. Remove docs also in test: 7,683 -> 7,677 clean traindev pool (6 removed).
  4. Seeded 90/10 resplit (seed=42): train=6,910 / val=767.
  5. Test set: Zenodo test (3,019) — UNTOUCHED.
  6. All overlaps verified 0 after fix.

**Evidence:** earn total=3,964 (matches paper). train+dev approx NLTK train.
**Inert:** Partially — larger training set vs Zenodo train-only (5,827 -> 6,910).
**Impact:** Closer to paper's 7,769 training size; val set is 767 (sufficient for HRRM).
**Annotate results:** "train=6,910 (A_R1b); val=767; test=3,019"

---

## A_R2: Reuters Text Field

**Class:** NECESSARY ENGINEERING ASSUMPTION
**Decision:** Use Zenodo 'text' column as-is (title+body, newlines->spaces).
**Evidence:** Text content = NLTK raw() with whitespace normalised.
**Inert:** Minimal — same tokens after BERT tokenization.

---

## Paper Inconsistencies (For Record)

- Reuters total: paper says 9,788 but 7,769+3,019=10,788 (known typo D6)
- Focal loss gamma/alpha values: not reported in paper
- HRRM fitting data: not reported (which split)
- Eq. 3 (concatenation): appears in paper but unused by Eqs. 4-8
- d/W shapes: undefined in paper notation
