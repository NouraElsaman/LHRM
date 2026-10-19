# LHRM Research Backlog & Future Work Roadmap

This backlog documents open scientific investigations, architectural ablations, and methodological improvements identified during the LHRM R_AUTHOR V1.0 development and audit cycle.

> [!IMPORTANT]
> **Status Notice**: Every item in this backlog is explicitly marked as **NOT YET COMPLETED**. No experimental claims, speculative inferences, or extrapolations are made regarding these items.

---

## 1. Compute-Matched CSAM-L2 Architecture Training
- **Status**: [ ] **NOT YET COMPLETED**
- **Objective**: Conduct full 15-epoch GPU training of the diagnostic variant `CSAM_L2` (incorporating full LayerNorm, dropout, and exact dimension matching as derived in DR4 forensic analysis) under identical batch size, learning rate schedule, and seed conditions as `R_AUTHOR`.
- **Key Questions**: Does CSAM-L2 achieve superior convergence or stability relative to the author-style CSAM residual structure?

---

## 2. Fair Compute-Matched BERT + BCE Baseline
- **Status**: [ ] **NOT YET COMPLETED**
- **Objective**: Train a standard BERT-base-uncased sequence classification model with standard linear pooling and BCE loss using the exact same 15-epoch budget, batch size (8), optimizer (AdamW, lr=2e-5), and warm-up schedule.
- **Key Questions**: What is the exact empirical delta attributable to Label-Aware Attention + CSAM over standard BERT document representation on AAPD?

---

## 3. Multi-Seed Robustness & Variance Bounds
- **Status**: [ ] **NOT YET COMPLETED**
- **Objective**: Train and evaluate R_AUTHOR across at least 3 distinct random seeds (e.g., seeds 42, 123, 456).
- **Key Questions**: What are the confidence intervals on P@1 (84.90%), Micro-F1 (72.78%), and Macro-F1 (55.53%)? Is the Epoch 6 selection epoch consistent across seeds?

---

## 4. HRRM (Hierarchical Recurrent Ranking Module) Error Analysis
- **Status**: [ ] **NOT YET COMPLETED**
- **Objective**: Conduct a deep residual failure analysis of the negative HRRM finding on the held-out Dev partition (where GBDT re-ranking degraded P@1 by -0.005 and Micro-F1 by -0.0009).
- **Key Questions**: Why did the tree-based re-ranker fail to improve upon neural logits? Was the 200-document fitting partition insufficient in sample diversity, or do cross-label correlation features overfit sparse tail labels?

---

## 5. Rare-Label (Tail Label) Performance & Imbalance Analysis
- **Status**: [ ] **NOT YET COMPLETED**
- **Objective**: Analyze the relationship between label training support and test F1-score across the 54 AAPD labels.
- **Key Questions**: For labels with support < 20 (e.g., rare ArXiv categories), does label-aware attention provide zero-shot/few-shot semantic transfer from precomputed label embeddings, or do tail labels suffer from low recall?

---

## 6. Semantic Label Taxonomy Verification
- **Status**: [ ] **NOT YET COMPLETED**
- **Objective**: Formally audit and verify the semantic alignment between raw AAPD category tags (e.g., `cmp-lg`, `cond-mat.stat-mech`) and human-curated taxonomy definitions from official ArXiv category taxonomies.
- **Key Questions**: Can taxonomy-augmented textual descriptions improve label representation quality over raw string embeddings?

---

## 7. Score Calibration & Decision Threshold Optimization
- **Status**: [ ] **NOT YET COMPLETED**
- **Objective**: Apply temperature scaling or Platt scaling to raw neural sigmoid outputs and evaluate optimal per-label threshold tuning on a dedicated validation split.
- **Key Questions**: Can post-hoc calibration improve Macro-F1 without requiring retraining?
