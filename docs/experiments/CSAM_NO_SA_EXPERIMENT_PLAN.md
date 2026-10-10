# L2_NO_SA Ablation Experiment Plan

## Objective
The approved `L2` architecture diagnosis found that the numerical training was stable and document-level logits showed measurable variance, but the top-1 predicted label remained identical across all 200 documents in the evaluation sample, never exceeding the static frequency prior.

**Hypothesis**: The label-label self-attention mechanism introduces a document-independent component that numerically dominates the ranking. By cleanly ablating this component, we can isolate whether the document-label interaction (`alpha_i * h_L_i`) can establish useful document-sensitive ranking on its own.

## Architectural Delta
**What changes:**
In the original `L2` formulation (Eq. 7):
$h_{final}^{(i)} = \alpha_i h_{L_i} + \sum_j A_{ij} h_{L_j}$
The new `L2_NO_SA` variant removes the second term entirely:
$h_{final\_no\_sa}^{(i)} = \alpha_i h_{L_i}$

**Difference from historical client `csam_nosa`:**
The client's original historical `csam_nosa` variant utilized pooled BERT output, a completely different sequence of aggregations, concatenated document embeddings, and a multi-label linear classifier layer of size `2H -> k`. The `L2_NO_SA` ablation directly targets the paper-faithful CSAM derivation (`CSAM_L`), maintaining the exact `alpha_i` computation and shared classifier `W` in $\mathbb{R}^H$, isolating only the label self-attention effect.

## Invariants
The following components are held strictly constant against `L2`:
- Official BERT source and weights
- Shared document/label BERT encoder
- Document `[CLS]` token representation
- Label-description `[CLS]` embeddings (computed with gradients)
- Scaled Eq. 4 document-label similarity ($\frac{1}{\sqrt{d}}$)
- Softmax over document-label similarity
- Shared label-wise classifier ($W$)
- Optimizer settings, tokenization, evaluation definitions, and seed (42).

## Diagnostic Criteria
The next Kaggle screening will reuse the `RUN_0` sample (200 dev documents) over 300 steps.
- **V1 (Numerical Stability)**: No NaN/Inf loss.
- **V2 (Loss Reduction)**: Mean loss of final 50 steps < first 50.
- **V3 (Attention Concentration)**: `alpha` one-hot fraction < 0.9.
- **V4 (Logit Variance)**: At least 28/54 labels have logit standard deviation > 0.01.
- **V5 (Ranking Superiority)**: Top-1 hits strictly > 63/200 (the frequency prior).

*Additional success signals*: An increase in the number of distinct top-1 predicted labels, or variance in the ordered top-3 ranking patterns across the sample, directly establishing that document dependence has been restored.

## Output Artifacts & Success Interpretation
If `L2_NO_SA` passes all criteria (notably V5), it isolates label-label self-attention as a *contributing factor* to rank collapse, but does NOT strictly prove it was the sole cause, nor does it prove that the LHRM architecture is optimal.
If it fails V5, it indicates that this specific formulation struggles with document-sensitive ranking, but does NOT prove that text-label alignment is fundamentally incapable of learning.
Additionally, the BERT+BCE (DR2) run serves as a well-understood control reference for dataset capability, not an absolute theoretical performance ceiling.

## Next Execution Steps
1. Create a Kaggle notebook running `cmd_run0` with `--candidates L2_NO_SA` and `--steps 300`.
2. Attach the generated LHRM bundle, AAPD, and `bert-base-uncased` datasets.
3. Run on a GPU runtime with internet OFF.
4. Retrieve the `_EVIDENCE.zip` for local inspection.
