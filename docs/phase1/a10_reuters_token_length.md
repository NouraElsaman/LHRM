# A10-Reuters: Token Length Measurement

**Date:** 2026-10-06
**Dataset:** Reuters (train split, N=6,910, after A_R1b dedup+resplit)
**Tokenizer:** bert-base-uncased

| Percentile | Tokens |
|---|---|
| p50 | 128 |
| p90 | 387 |
| p95 | 555 |
| p99 | 899 |
| max | 1,633 |

| Threshold | % Truncated |
|---|---|
| 128 | 49.80% |
| 256 | 20.64% |
| 512 | 5.88% |

**A10-Reuters decision:**
  Rule: "smallest of {128, 256, 512} covering >=99% of documents"
  Result: max_len=512 (BERT hard limit; none of the three options cover >=99%)
  At 512: 94.12% of docs fit; 5.88% (407 docs) are truncated.
  Truncation is unavoidable given BERT's 512-token hard limit.
  Registered as NECESSARY ENGINEERING ASSUMPTION.
  Worst case: truncated documents lose information from the end of Reuters news articles,
  which typically have the most detail at the beginning (inverted pyramid structure).
  Impact is likely small (p90 = 387 < 512).
