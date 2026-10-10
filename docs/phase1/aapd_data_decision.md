# AAPD Dataset Decision Document

**Date received:** 2026-10-07 (this session)
**Status:** DECISION_REGISTERED

---

## 1. Decision

AAPD_DATASET_DECISION = USE_ZENODO_SOURCE_AS_PROVIDED

Client instruction: "proceed with the AAPD dataset downloaded from Zenodo as provided,
as the benchmark for the baseline." Received 2026-10-07.

No corrective relabeling is performed. The dataset is used as supplied under the
client-approved project decision.

---

## 2. Source

URL: https://zenodo.org/records/6344750/files/AAPD.zip
DOI: 10.5281/zenodo.6344750
Download: archive present at Data/AAPD.zip (obtained in prior session)

Archive hashes:
  Data/AAPD.zip SHA-256: e1b418427bcd2662f04c347099deb0d520b4066f782ed40c2a4aed15df694a08

Per-CSV SHA-256 (raw bytes from archive, no decompression artifacts):
  train.csv: e7e73064385f78fe092edd47d4e4059c35875a1ea2a95859ad3cde091ed9af95
  dev.csv:   a85f612d4c1cb2be6d1de94cbcd9468cba97d221f826014264f4d279c82e220b
  test.csv:  a0672754aeadeb60f7c8f6cf08a7263add2bc42095f3b547a61aae14f9749b75

Internal path structure: AAPD/{train,dev,test}.csv
Text column header: "abstract"
Label column headers: 54 columns following "abstract"

---

## 3. Dataset Facts (as supplied)

| Split | Rows | Labels | Mean labels/doc | Zero-label docs | Within-split text dups |
|---|---|---|---|---|---|
| train | 53840 | 54 | 2.407 | 0 | 68 |
| dev   | 1000 | 54 | 2.400 | 0 | 0 |
| test  | 1000 | 54 | 2.421 | 0 | 0 |

Label column order: 54 columns read exactly as supplied from CSV headers.
Label-order hash (SHA-256 of JSON-serialised list): a3d662f3920259a94ed61d83f98f21cabe7f908d98a5662ccd9adc2413cc000b
Label order consistent across all three splits: PASS

Cross-split duplicate text counts (observed; no rows dropped):
  train/dev: 4, train/test: 5, dev/test: 0

Label-matrix dimensions:
  train: [53840, 54]
  dev:   [1000, 54]
  test:  [1000, 54]

---

## 4. Processed Dataset

Produced by: src/lhrm_baseline/data.py (load_aapd_split + save_processed)
Output directory: data/processed/AAPD/
Pipeline rule: Zenodo source -> project preprocessing -> baseline.
NO reordering, renaming or remapping of columns at any pipeline stage.

File SHA-256 hashes:
  dev_labels.npy: 71508290e9b54e91990eebe27a24ef951b876cb3ae08a30494011d28e7209c4c
  dev_texts.json: ef53fb5d81941d759bd5703cd8a23ed18900c66e1926957472bef6fb68704a18
  label_texts.json: 22f264dc2997cdbf9beda7e15d05d55984fb503eaa48359497a2de72511e3093
  labels.json: 481f02abe49d22d4cf47c7ae29a8a1c8ac2718a2ecfa453e41cceac3c937358b
  test_labels.npy: 09263891d0ccb1e15c8d316aba8d2939cc391ff9a206bbdfaca8e22090e432a4
  test_texts.json: b0ff04fd49f4d0e59ae7e85b80a1b631a1d90e9b0c2a61210a30e2a5bda18ab7
  train_labels.npy: 28a286ef5390e40a8eed0a904e29945b97ac5db7264022e4f59a7b1c959f965d
  train_texts.json: 3c7147f1001e464e47efa28e99f73133229669b6d668ea1761ba5509fc70b720

Label order in processed data matches CSV headers: True
Label-order hash in DATA_MANIFEST matches computed: PROC_MISSING

Label texts: built as "{label_header_name}: {definition}" using case-insensitive
lookup against arxiv_labels_definitions.json by the SUPPLIED header name.
Because the header-to-column assignment is unverified (see Section 6), label texts
may not describe the true content of the corresponding documents.

---

## 5. Token-Length Statistics (bert-base-uncased, add_special_tokens=True)

AAPD train (N=53840):
  p50=180, p90=294,
  p95=328, p99=385,
  max=655

Truncation rates at max_len=512 (head truncation):
  train: 0.03%
  dev:   0.10%
  test:  0.00%

A10: max_len=512 (BERT hard limit, NEA). Head truncation only (Phase 2: sliding window).

---

## 6. Known Dataset Observation

Source: Zenodo record 6344750, AAPD.zip, files as supplied.

Observed facts (not conclusions about the correct mapping):
  (a) Byte-identical column pairs in the training label matrix:
      'cmp-lg' == 'cond-mat.dis-nn': 17152 active training documents
      'math.co' == 'math.it': 875 active training documents
      These columns are indistinguishable from their label vectors alone.

  (b) Partial arXiv-API join evidence (from prior session):
      29 documents matched of 300 attempted (low match rate due to preprocessing
      divergence between the AAPD abstracts and arXiv API text).
      In the 29 matched cases, the column header names did not correspond to the
      arXiv categories of the matched papers.
      This is a partial observation (sample=29, coverage=9.7%) and NOT a verified
      full mapping. It is documented as prior context only.

  (c) The arXiv recovery workflow (Route A/B) is DISABLED and OUT OF SCOPE per
      client instruction. No join, relabeling or remapping is performed.

---

## 7. Handling Statement

"No corrective relabeling is performed. The dataset is used as supplied under the
client-approved project decision."

---

## 8. Scientific Limitation

This is the supplied benchmark dataset used for project reproduction. The anomaly
described in Section 6 is an inherited dataset characteristic. Results on AAPD must
be interpreted with this in mind: label definitions are attached by the supplied
column header names, which may not correspond to the true arXiv categories of the
documents. This does not change the fact that this is the client-approved baseline.

Mapping status: UNRESOLVED_LABEL_MAPPING, ACCEPTED_BY_CLIENT_DECISION.
VERIFIED_LABEL_MAPPING must never be written.

---

## 9. Integrity Check Results

| Check | Result |
|---|---|
| Row counts (53840 / 1000 / 1000) | True |
| 54 labels per split | True |
| Label order consistent across splits | True |
| Zero-label docs = 0 in all splits | True |
| All 54 labels have definitions | True |
| All label texts <= 64 tokens | True |
| Processed label order matches CSV headers | True |

AAPD_BASELINE = READY_FOR_CLIENT_APPROVED_REPRODUCTION
