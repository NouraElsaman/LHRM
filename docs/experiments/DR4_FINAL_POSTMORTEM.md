# DR4 Final Postmortem: Evaluation of Variant L2_A5B & Trained-State Logit Decomposition

**Experiment ID:** `RUN_0_L2_A5B_20261012_A`  
**Diagnostic ID:** `DR4_POSTTRAIN_DECOMPOSITION_20261012`  
**Date:** 2026-10-12  
**Dataset:** AAPD (Train & Dev splits only; Test split strictly quarantined)  
**Status:** COMPLETE — Evaluated, Audited, and Archival Evidence Verified  

---

## 1. Experimental Motivation & Preregistered Hypothesis

In DR1 (`L2`) and DR3 (`L2_NO_SA`), the model failed viability criterion V5 (Ranking Superiority), collapsing to assigning the identical Top-1 label to all 200 Dev documents (at exactly the marginal frequency prior of 63/200, 31.50%). 

The DR3 trained-checkpoint diagnostic revealed that the static label self-attention term ($1.651$) dwarfed the document-dependent contribution ($0.034$) by $48.42\times$. Paired with a shared scalar projection classifier ($w \in \mathbb{R}^{1 \times H}$, paper Eq. 8), this locked label rankings across all documents.

The preregistered hypothesis of DR4 was based on Decision Registry **A5b**:
> *Hypothesis:* Unsharing the classifier head parameters per label ($W \in \mathbb{R}^{k \times H}$ with shared scalar bias $b \in \mathbb{R}$) allows each label to learn an independent linear decision boundary. If the shared projection was the primary bottleneck forcing a uniform ranking across documents, independent per-label projections should permit document-dependent cross-attention signals to drive distinct label rankings across documents.

---

## 2. Locked Architectural Comparison: L2 vs L2_A5B

Both variants operate on BERT CLS representations of document $h_T \in \mathbb{R}^H$ and label descriptions $h_{L_i} \in \mathbb{R}^H$:
- **Cross-Attention:** Scaled dot-product cross-attention: $\alpha_i = \operatorname{softmax}((h_T \cdot h_L^\top) / \sqrt{H})_i \in [0, 1]$.
- **Label Self-Attention:** $A_{ij} = \operatorname{softmax}((h_L \cdot h_L^\top) / \sqrt{H})_{ij}$, $SA_i = (A \cdot h_L)_i \in \mathbb{R}^H$.
- **Combined Representation:** $h_{\text{final}, i} = \alpha_i h_{L_i} + SA_i \in \mathbb{R}^H$.

The single isolated difference is the classifier parameterization:
- **Variant L2 (Baseline):** Shared vector $w \in \mathbb{R}^{1 \times H}$ with scalar bias $b \in \mathbb{R}$:
  $$\hat{y}_{b, i} = w \cdot h_{\text{final}, b, i} + b = \alpha_{b, i} (w \cdot h_{L_i}) + w \cdot SA_i + b$$
- **Variant L2_A5B (Candidate):** Unshared per-label vector matrix $W \in \mathbb{R}^{54 \times H}$ with shared scalar bias $b \in \mathbb{R}$:
  $$\hat{y}_{b, i} = W_i \cdot h_{\text{final}, b, i} + b = \alpha_{b, i} (W_i \cdot h_{L_i}) + W_i \cdot SA_i + b$$

---

## 3. Training Protocol & Dataset Provenance

- **Dataset:** AAPD (ArXiv Academic Paper Dataset), 54 categories.
- **Protocol:** Approved AAPD 300-step screening protocol (`batch_size=16`, `lr=2e-5`, linear warmup over 30 steps, AdamW, mixed precision fp16, seed 42).
- **Optimizer Steps:** Exactly 300 steps per variant.
- **Backbone:** Frozen official `bert-base-uncased` revision `86b5e0934494bd15c9632b12f734a8a67f723594`.
- **Label Tokenization:** Canonical 54 labels tokenized at `max_len=64`.
- **Document Tokenization:** Truncation at `max_len=512`.
- **Test Quarantine:** Test split strictly quarantined and never loaded or evaluated.

---

## 4. Step-0 Initialization Parity

At step 0 (initialization), both models exhibited numerical parity:
- Loss at step 0: L2 = 0.7005, L2_A5B = 0.7006.
- Cross-attention distribution was initialized identically diffuse ($\alpha \approx 1/54 \approx 0.0185$, entropy $\approx 3.989$).

---

## 5. V1–V5 Screening Results

Both models completed 300 steps of GPU training on Kaggle. The verified screening gates evaluated on the frozen 200 Dev documents:

| Screening Criterion | Required Threshold | Variant L2 | Variant L2_A5B | Disposition |
|---|---|---:|---:|:---:|
| **V1: Numerical Stability** | No NaNs, finite loss/logits | PASS | PASS | Valid |
| **V2: Loss Progression** | Final loss < initial loss | PASS (0.7005 $\to$ 0.1740) | PASS (0.7006 $\to$ 0.1729) | Valid |
| **V3: Attention Discrimination** | Mean max $\alpha > 0.025$ | PASS (0.0406) | PASS (0.0278) | Valid |
| **V4: Logit Variability** | $\ge 28/54$ labels with std $> 0.01$ | **FAIL (21/54)** | **FAIL (3/54)** | Rejected |
| **V5: Ranking Superiority** | P@1 hits $> 63/200$ (frequency prior) | **FAIL (63/200, 31.5%)** | **FAIL (63/200, 31.5%)** | Rejected |
| **Overall Viability Gate** | All V1–V5 must pass | **NOT VIABLE** | **NOT VIABLE** | Both Failed |

---

## 6. Saved Prediction Diversity & Ranking Behavior

| Metric | L2 Baseline | L2_A5B Candidate | Empirical Delta |
|---|---:|---:|---:|
| **P@1 Correct Hits** | 63 / 200 (31.50%) | 63 / 200 (31.50%) | 0.00% |
| **Marginal Frequency Prior** | 63 / 200 (31.50%) | 63 / 200 (31.50%) | Baseline match |
| **Distinct Top-1 Labels** | 1 (`cond-mat.dis-nn`) | 2 (`cmp-lg`, `cond-mat.dis-nn`) | +1 label |
| **Top-1 Distribution** | `cond-mat.dis-nn`: 200/200 (100.0%) | `cmp-lg`: 199/200 (99.5%)<br>`cond-mat.dis-nn`: 1/200 (0.5%) | Degenerate shift |
| **Distinct Top-3 Patterns** | 1 | 2 | +1 pattern |
| **Top-1 / Top-2 Margin** | 0.0173 | 0.0003 | -0.0170 (near-tie) |

---

## 7. Real Trained-Checkpoint Logit Decomposition

The trained step-300 model-only checkpoints were reloaded and evaluated through exact algebraic decomposition:
$$\text{logits}[b, i] = \underbrace{\alpha[b, i] (W_i \cdot h_{L_i})}_{\text{doc\_term}[b, i]} + \underbrace{W_i \cdot SA_i}_{\text{sa\_term}[i]} + b$$

Mathematical properties verified on the authentic 200 Dev documents:
1. **Algebraic Reconstruction Precision:**
   - L2 maximum reconstruction error: `7.15e-07` ($\le 10^{-5}$).
   - L2_A5B maximum reconstruction error: `4.77e-07` ($\le 10^{-5}$).
2. **Forward Pass Agreement:**
   - Independent model forward-pass logits matched algebraic decomposition within `7.15e-07` (L2) and `4.77e-07` (L2_A5B).
3. **Saved Predictions Agreement:**
   - Sigmoid probabilities matched saved prediction records across all $200 \times 54 = 10,800$ scores within `1.04e-07` (L2) and `7.45e-08` (L2_A5B).

---

## 8. Document-Dependent vs Static Label-SA Contributions

| Decomposition Quantity | L2 Baseline | L2_A5B Candidate | Comparison |
|---|---:|---:|:---:|
| **Mean Absolute Document Term** | 0.034790 | 0.031342 | -9.9% attenuation |
| **Mean Absolute Label-SA Term** | 1.670919 | 1.624252 | -2.8% magnitude |
| **SA-to-Document Magnitude Ratio** | **48.03x** | **51.82x** | **+3.79x increase in static dominance** |
| **Cross-Document Variance of SA Term** | **0.000000** | **0.000000** | Strictly invariant across documents |
| **Attention Entropy** | 3.6907 | 3.9537 | +0.2630 (more diffuse) |
| **Mean Maximum Attention $\alpha$** | 0.0406 | 0.0278 | -0.0128 (closer to uniform 0.0185) |

### Key Mathematical Invariant:
For any fixed label $i$, the additive label-SA term $W_i \cdot SA_i$ and the bias $b$ are completely invariant to document $b$ ($\operatorname{Var}_b(\text{sa\_term}) \equiv 0$). Therefore:
$$\operatorname{Var}_b(\text{logits}[:, i]) \equiv \operatorname{Var}_b(\text{doc\_term}[:, i])$$
**The document term is the sole source of cross-document variation in raw logits.**

---

## 9. Classifier Head Weight Geometry in L2_A5B

Analyzing the unshared weight matrix $W \in \mathbb{R}^{54 \times 768}$ across all labels:
- **Per-row L2 Norm:** Mean $\|W_i\|_2 = 0.5905 \pm 0.0045$ (Range: 0.5796 to 0.5986).
- **Diagonal Self-Similarity:** Identically `1.0000`.
- **Off-Diagonal Pairwise Cosine Similarity:**
  - Mean: **0.9985**
  - Standard Deviation: `0.0022`
  - Minimum: `0.9867`
  - Maximum: `1.0000`
- **Mean Alignment with L2 Shared Vector $w$:** **0.9986** (Min: 0.9880, Max: 0.9998).

### Empirical Finding:
Removing parameter sharing in the classifier **did not cause the classifier rows to diverge into label-specific subspaces**. All 54 rows collapsed into near-identical directions (mean angular separation $< 3^\circ$), mirroring the shared vector learned by L2.

---

## 10. Duplicate-Target Label Analysis (`cmp-lg` vs `cond-mat.dis-nn`)

In the AAPD dataset, `cmp-lg` (index 0) and `cond-mat.dis-nn` (index 1) possess identical binary target columns across both Train and Dev sets. They provide a controlled natural probe of how the architecture resolves degenerate targets:

| Measurement | Variant L2 | Variant L2_A5B |
|---|---:|---:|
| `cmp-lg` Top-1 Count | 0 / 200 (0.0%) | 199 / 200 (99.5%) |
| `cond-mat.dis-nn` Top-1 Count | 200 / 200 (100.0%) | 1 / 200 (0.5%) |
| Pairwise Weight Cosine Similarity | 1.0000 (shared $w$) | **0.999989** |
| Euclidean Weight Distance $\|W[0] - W[1]\|_2$ | 0.000000 | **0.002679** |
| `cmp-lg` Static SA Term | -0.627116 | -0.631173 |
| `cond-mat.dis-nn` Static SA Term | -0.551401 | -0.633980 |
| **Static SA Difference ($W_0 SA_0 - W_1 SA_1$)** | **-0.075715** | **+0.002807** |
| **Mean Document Term Difference** | **-0.000021** | **-0.001332** |
| **Net Mean Logit Difference** | **-0.075736** | **+0.001475** |

### Mechanistic Explanation:
In L2, `cond-mat.dis-nn` enjoyed a static SA advantage of $+0.0757$, winning Top-1 on 200/200 documents. In L2_A5B, the two weight vectors remained virtually identical ($\cos = 0.999989$, distance $0.0027$), but `cmp-lg` gained a tiny static SA advantage of $+0.0028$, while the mean document-term difference was only $-0.0013$. Because $+0.0028 > 0.0013$, `cmp-lg` overtook `cond-mat.dis-nn` by a wafer-thin net logit margin of $+0.001475$ (score margin $0.0003$), flipping 199/200 predictions.

---

## 11. Scientific Interpretation & Precise Limitations

1. **Unsharing the classifier failed to restore document-dependent ranking:**
   The number of V4 passing labels dropped from 21/54 to 3/54, and Top-1 accuracy remained locked at the prior (63/200).
2. **Static dominance intensified:**
   The SA-to-document magnitude ratio grew from $48.03\times$ to $51.82\times$.
3. **Cross-attention became more diffuse:**
   Mean max $\alpha$ decreased from 0.0406 to 0.0278, and attention entropy rose to 3.9537 (99.1% of maximum uniform entropy $\ln(54) \approx 3.989$).
4. **Limits of Observational Decomposition:**
   The post-training decomposition proves that the trained state is dominated by static offsets, but **does not establish the causal learning dynamics** that drove optimization into this state. Specifically, observational data cannot distinguish whether:
   - The dot-product similarity $h_T \cdot h_L / \sqrt{H}$ without learned query/key projections cannot produce sharp discriminative attention, OR
   - The additive structure $h_{\text{final}} = \alpha h_L + A h_L$ offers an easy optimization shortcut where the model minimizes loss by tuning static label offsets rather than learning document features.

---

## 12. Final Scientific Disposition

- **Variant L2_A5B is declared NOT VIABLE** under the 300-step screening protocol.
- Architecture A5b (unshared per-label classifier) is rejected as a solution to ranking collapse.
- Both L2 and L2_A5B remain locked at the 63/200 marginal prior.

---

## 13. Provenance of the Authoritative Notebook

An independent audit identified that the original automated notebook generator failed to unpack the tuple `(logits, diagnostics)` returned by `LHRMModel.forward()`. 

The corrected notebook ([`notebooks/lhrm_dr4_trained_decomposition_kaggle.ipynb`](file:///D:/LHRM_Publication/notebooks/lhrm_dr4_trained_decomposition_kaggle.ipynb)) properly handled `fwd_logits, _ = model(...)` and executed successfully on Kaggle GPU. The executed notebook is preserved in external artifacts at `notebook/lhrm_dr4_trained_decomposition_EXECUTED_20261012.ipynb`.

---

## 14. Verified Evidence Checksums

| Artifact Path | Size (Bytes) | SHA-256 Digest |
|---|---:|---|
| `checkpoints/L2_STEP300_MODEL_ONLY.pt` | 438,015,999 | `99a06f1e7ba5bd72f3d855b61672b95a52f97f83e8db519e09b7992f2ca48e49` |
| `checkpoints/L2_A5B_STEP300_MODEL_ONLY.pt` | 438,179,707 | `b80852c7daafb626860dc2c981290eb2f7905e5303531c551b2fa14e0bb27022` |
| `evidence/DR4_A5B_CORE_EVIDENCE.zip` | 233,244 | `7fa67b8c8bd1a49fa1464ecb443d6489ed17cb5de36eb062bf4affb2656c8aeb` |
| `evidence/DR4_A5B_EVIDENCE_RUN_0_L2_A5B_20261012_A.zip` | 229,177 | `7590c5178e7e8f0c9e44c0d5b76c393e8bd586137bf7be0ebaa082cc4083a316` |
| `diagnostics/DR4_POSTTRAIN_DECOMPOSITION_EVIDENCE.zip` | 8,079 | `552f0340b0bdb154f230817871668acc788bb7471affbd45171f7dbabafea4e9` |
| `diagnostics/DR4_POSTTRAIN_DECOMPOSITION_RESULTS.json` | 5,499 | `d54d97b60805a8debe439ce3d743b86282daf036a7db1533881292e871dff56e` |
| `diagnostics/DR4_POSTTRAIN_PER_LABEL_METRICS.csv` | 10,646 | `ef701b73a67f52b1d7a0c148b9901e7acd0c5a14c434aa521da0d0614c0abe51` |
| `diagnostics/DR4_POSTTRAIN_DECOMPOSITION_REPORT.md` | 5,549 | `825e37cc0276a0aa635139a04c37ea47cf1fcb5c8f7ed7b875182ad1c8698095` |
| `notebook/lhrm_dr4_trained_decomposition_EXECUTED_20261012.ipynb` | 35,444 | `64cb7320257df6ed364df1fb12807d59df8cbac5e4b4414b5cf9ca11cd8a0aff` |
| `notebooks/lhrm_dr4_trained_decomposition_kaggle.ipynb` (source) | 16,929 | `e0fcbc66865e2d26741cf2798e277866da2f7fd4fd2fa50eab50e9dcebd2a054` |
