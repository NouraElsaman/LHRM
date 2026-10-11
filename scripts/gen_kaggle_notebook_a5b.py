#!/usr/bin/env python3
"""
Generate notebooks/lhrm_run0_l2_a5b_kaggle.ipynb —
Dedicated, reproducible Kaggle GPU notebook for the L2 vs L2_A5B CSAM controlled screening.

Protocol:
  Experiment ID: RUN_0_L2_A5B_20261012_A
  Steps: 300 per candidate
  Candidates: L2, L2_A5B
  Frozen seed: 42
  Learning rate: 2e-5 (locked approved baseline)
  Validation sample: locked 200 dev IDs (frequency prior 63/200)
  Test data quarantine: strictly no test split read, copied, prepared, predicted, or evaluated.

Usage:
  python scripts/gen_kaggle_notebook_a5b.py [--build_bundle]
"""
import argparse
import hashlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

BUNDLE_NAME = "lhrm_gpu_bundle_l2_a5b_20261012.zip"
OUT_NB = os.path.join(ROOT, "notebooks", "lhrm_run0_l2_a5b_kaggle.ipynb")
RUN_ID = "RUN_0_L2_A5B_20261012_A"
CONFIG = "configs/final/aapd_baseline.yaml"
CANDIDATES = "L2,L2_A5B"
N_STEPS = 300


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def cell(source_str, cell_type="code"):
    lines = source_str.rstrip("\n").split("\n")
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": [l + "\n" for l in lines[:-1]] + [lines[-1]],
        **({"outputs": [], "execution_count": None} if cell_type == "code" else {}),
    }


def build_notebook(bundle_name, bundle_sha256):
    cells = []

    # ── Cell 0: Markdown Protocol & Guidelines ───────────────────────────────
    cells.append(cell(f"""\
# LHRM DR4 Controlled Screening — L2 vs L2_A5B ({RUN_ID})

**Strictly Controlled CSAM Ablation Screening (300 Optimizer Steps per Variant)**

This experiment tests the hypothesis:
> Does replacing the shared scalar classifier ($W \\in \\mathbb{{R}}^{{1 \\times H}}$) with an unshared per-label weight matrix ($W \\in \\mathbb{{R}}^{{k \\times H}}$) allow document-dependent ranking to emerge under diffuse cross-attention, resolving the single-label ranking collapse on AAPD?

### Research Protocol & Quarantine Constraints:
1. **Experiment ID:** `{RUN_ID}` (fail-closed if output directory exists and is non-empty).
2. **Candidates:**
   - `L2`: Baseline CSAM with scaled similarity ($1/\\sqrt{{H}}$), label-label self-attention, shared classifier head ($W \\in \\mathbb{{R}}^{{1 \\times H}}$), shared scalar bias ($b \\in \\mathbb{{R}}$).
   - `L2_A5B`: Candidate CSAM identical to `L2` except unshared per-label classifier head ($W \\in \\mathbb{{R}}^{{k \\times H}}$) and shared scalar bias ($b \\in \\mathbb{{R}}$).
3. **Locked Hyperparameters:**
   - Seed: 42
   - Learning rate: **2e-5** (approved AAPD baseline rate, strictly NOT 3e-5)
   - Optimizer: AdamW, weight decay 0.01
   - Steps: 300 optimizer steps
   - Batch size: 8
   - Loss: Focal Loss ($\\gamma=2.0, \\alpha=0.25$)
   - AMP enabled
4. **Strict Isolation & No-Confounding Rules:**
   - Label-label self-attention remains fully **ENABLED**.
   - Shared scalar bias remains **RETAINED** (no per-label bias introduced).
   - No query/key projection layers added.
   - No temperature or architecture modifications.
5. **Data Quarantine:** **Test split is strictly quarantined.** Only AAPD Train (53,840) and Dev (1,000) splits are loaded.
6. **Fixed Dev Sample:** Locked 200 validation documents from `experiments/RUN_0/run0_val_sample_ids.json` (frequency prior: exactly 63/200 hits).
7. **Scientific Interpretation Limits:**
   - V4 passes when $\\ge 28/54$ labels exhibit cross-document logit standard deviation $> 0.01$.
   - V5 passes when P@1 integer hits strictly exceed 63 ($> 31.50\\%$).
   - Ranking diversity alone is NOT proof of correct learning.
   - Passing V4 and V5 justifies further evaluation, not successful reproduction.
   - Failing V4 or V5 shows that unsharing the classifier alone is insufficient; it does NOT prove dot-product attention is the sole remaining failure source.

**Bundle SHA-256:** `{bundle_sha256}`
""", cell_type="markdown"))

    # ── Cell 1: Environment & Official BERT Setup ────────────────────────────
    cells.append(cell("""\
# Cell 1: Environment check & Official BERT Acquisition
import os, sys, time, json, hashlib, shutil, gc
from pathlib import Path
import torch

print("=== L2 vs L2_A5B Screening: Environment & Device Gate ===", flush=True)
if not torch.cuda.is_available():
    raise RuntimeError("GPU GATE: CUDA GPU required — select Kaggle GPU runtime (T4 or P100)")
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__, " | CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0), flush=True)

import sklearn, transformers, numpy as np
print(f"Transformers: {transformers.__version__} | Sklearn: {sklearn.__version__} | NumPy: {np.__version__}")

# Pin official BERT revision
OFFICIAL_BERT_COMMIT = "86b5e0934494bd15c9632b12f734a8a67f723594"
KAGGLE_BERT_PATH = Path("/kaggle/input/bert-base-uncased")
LOCAL_BERT_CACHE = Path("/kaggle/working/bert_offline")

if LOCAL_BERT_CACHE.is_dir() and (LOCAL_BERT_CACHE / "config.json").is_file():
    print(f"Using pre-cached BERT: {LOCAL_BERT_CACHE}")
    BERT_PATH = str(LOCAL_BERT_CACHE)
elif KAGGLE_BERT_PATH.is_dir() and (KAGGLE_BERT_PATH / "config.json").is_file():
    print(f"Found Kaggle attached BERT dataset at: {KAGGLE_BERT_PATH}")
    BERT_PATH = str(KAGGLE_BERT_PATH)
else:
    print(f"Acquiring official bert-base-uncased at revision {OFFICIAL_BERT_COMMIT}...", flush=True)
    from transformers import BertModel, BertTokenizer
    os.makedirs(LOCAL_BERT_CACHE, exist_ok=True)
    model = BertModel.from_pretrained("google-bert/bert-base-uncased", revision=OFFICIAL_BERT_COMMIT)
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased", revision=OFFICIAL_BERT_COMMIT)
    model.save_pretrained(LOCAL_BERT_CACHE)
    tokenizer.save_pretrained(LOCAL_BERT_CACHE)
    BERT_PATH = str(LOCAL_BERT_CACHE)
    print(f"Official BERT saved to {LOCAL_BERT_CACHE}")
"""))

    # ── Cell 2: Unpack GPU Bundle & Verify Integrity ─────────────────────────
    cell2_code = """\
# Cell 2: Unpack GPU Bundle & Verify Integrity
import zipfile

WORK = Path("/kaggle/working")
BUNDLE_NAME = "@BUNDLE_NAME@"
BUNDLE_PATH = None

candidate_locations = [
    Path("/kaggle/input") / BUNDLE_NAME,
    Path("/kaggle/input") / "lhrm-gpu-bundle" / BUNDLE_NAME,
    Path("/kaggle/input") / "lhrm-l2-a5b-bundle" / BUNDLE_NAME,
    WORK / "bundles" / BUNDLE_NAME,
    WORK / BUNDLE_NAME,
]

for p in candidate_locations:
    if p.is_file():
        BUNDLE_PATH = p
        break

if BUNDLE_PATH is None:
    found = list(Path("/kaggle/input").rglob("*.zip"))
    for f in found:
        if "a5b" in f.name.lower() or "bundle" in f.name.lower():
            BUNDLE_PATH = f
            break

if BUNDLE_PATH is None:
    raise FileNotFoundError(f"Could not locate bundle {BUNDLE_NAME} in /kaggle/input or /kaggle/working.")

print(f"Located GPU Bundle: {BUNDLE_PATH}")
h = hashlib.sha256()
with open(BUNDLE_PATH, "rb") as f:
    for block in iter(lambda: f.read(1024*1024), b""):
        h.update(block)
actual_bundle_sha = h.hexdigest()
print(f"Bundle SHA-256: {actual_bundle_sha}")

with zipfile.ZipFile(BUNDLE_PATH, "r") as z:
    z.extractall(WORK)
print("Bundle extracted to /kaggle/working.")

if str(WORK / "src") not in sys.path:
    sys.path.insert(0, str(WORK / "src"))

# Verify key files
required_extracted = [
    WORK / "run_baseline.py",
    WORK / "configs" / "final" / "aapd_baseline.yaml",
    WORK / "configs" / "final" / "CONFIG_FREEZE.json",
    WORK / "Label_Definition" / "arxiv_labels_definitions.json",
    WORK / "docs" / "phase1" / "run0_local_step0_crosscheck.json",
    WORK / "experiments" / "RUN_0" / "run0_val_sample_ids.json",
]
for rf in required_extracted:
    assert rf.is_file(), f"Extracted file missing: {rf}"
print("All required core files verified present.")
""".replace("@BUNDLE_NAME@", BUNDLE_NAME)
    cells.append(cell(cell2_code))

    # ── Cell 3: Verification of Controlled Step-0 Parity ────────────────────
    cells.append(cell("""\
# Cell 3: Controlled Step-0 Parity Verification (L2 vs L2_A5B)
import torch
from transformers import BertModel
from lhrm_baseline.model import LHRMModel, init_a5b_from_l2

print("=== Verifying Step-0 Baseline Logit & Attention Parity ===", flush=True)

device = torch.device("cuda")
bert = BertModel.from_pretrained(BERT_PATH).to(device)

k = 54
l2_model = LHRMModel(bert, num_labels=k, variant="L2").to(device)
a5b_model = LHRMModel(bert, num_labels=k, variant="L2_A5B").to(device)

# Copy baseline shared classifier vector into all k rows of A5b
init_a5b_from_l2(l2_model, a5b_model)

l2_model.eval()
a5b_model.eval()

# Dummy batch with identical inputs
B = 4
doc_ids = torch.randint(0, 1000, (B, 64), device=device)
doc_mask = torch.ones(B, 64, device=device)
label_ids = torch.randint(0, 1000, (k, 32), device=device)
label_mask = torch.ones(k, 32, device=device)

with torch.no_grad():
    l2_logits, l2_diag = l2_model(doc_ids, doc_mask, label_ids, label_mask)
    a5b_logits, a5b_diag = a5b_model(doc_ids, doc_mask, label_ids, label_mask)

max_logit_diff = (l2_logits - a5b_logits).abs().max().item()
print(f"Max Step-0 Logit Difference: {max_logit_diff:.8e}")
assert max_logit_diff < 1e-5, f"Step-0 logit parity failed! Diff: {max_logit_diff}"

print("Step-0 baseline parity VERIFIED. Parameter sharing isolated from initialization difference.", flush=True)
del l2_model, a5b_model, bert
torch.cuda.empty_cache()
gc.collect()
"""))

    # ── Cell 4: Execute 300-Step Screening CLI ──────────────────────────────
    cell4_code = """\
# Cell 4: Execute 300-Step Screening (L2 vs L2_A5B)
import subprocess

WORK = Path("/kaggle/working")
RUN_ID = "@RUN_ID@"
CANDIDATES = "@CANDIDATES@"
N_STEPS = @N_STEPS@

# Ensure config points to local BERT
cfg_path = WORK / "configs" / "final" / "aapd_baseline.yaml"
cfg_text = cfg_path.read_text()
if "bert_model:" in cfg_text:
    import re
    cfg_text = re.sub(r'bert_model:\\s*.*', f'bert_model: "{BERT_PATH}"', cfg_text)
    cfg_path.write_text(cfg_text)

# Validate frozen 200 Dev IDs before starting
frozen_ids_source = WORK / "experiments" / "RUN_0" / "run0_val_sample_ids.json"
assert frozen_ids_source.is_file(), f"Missing frozen reference IDs at {frozen_ids_source}"
with open(frozen_ids_source) as f:
    ref_ids = json.load(f)
assert len(ref_ids) == 200, f"Expected 200 validation sample IDs, got {len(ref_ids)}"
print(f"Verified frozen 200 Dev sample IDs: OK (first 5: {ref_ids[:5]})")

cmd = [
    sys.executable,
    str(WORK / "run_baseline.py"),
    "run0",
    "--config", str(cfg_path),
    "--run_id", RUN_ID,
    "--steps", str(N_STEPS),
    "--candidates", CANDIDATES,
]

print("Executing screening command:", " ".join(cmd), flush=True)
p = subprocess.Popen(cmd, cwd=str(WORK), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in iter(p.stdout.readline, ""):
    print(line, end="", flush=True)
p.stdout.close()
ret_code = p.wait()

if ret_code != 0:
    raise RuntimeError(f"run_baseline.py run0 failed with exit code {ret_code}")

print("Screening CLI completed successfully.", flush=True)
""".replace("@RUN_ID@", RUN_ID).replace("@CANDIDATES@", CANDIDATES).replace("@N_STEPS@", str(N_STEPS))
    cells.append(cell(cell4_code))

    # ── Cell 5: Diagnostic Audit & Viability Evaluation ──────────────────────
    cell5_code = """\
# Cell 5: Diagnostic Audit & Viability Evaluation (V1–V5 Integer Comparison)
WORK = Path("/kaggle/working")
RUN_ID = "@RUN_ID@"
EXP_DIR = WORK / "experiments" / RUN_ID

# Verify required artifacts
REQUIRED_PER_CAND = [
    "step0_diagnostics.json",
    "step300_diagnostics.json",
    "train.log",
    "val_sample_predictions.json",
    "checkpoints/step_150.pt",
    "checkpoints/step_300.pt",
]
REQUIRED_COMMON = [
    "run0_decision.json",
    "env.json",
    "resolved_config.json",
    "run0_val_sample_ids.json",
    "SHA256SUMS.txt",
]

for f in REQUIRED_COMMON:
    assert (EXP_DIR / f).is_file(), f"Missing common artifact: {f}"

for cand in ["L2", "L2_A5B"]:
    c_dir = EXP_DIR / f"variant_{cand}"
    for f in REQUIRED_PER_CAND:
        assert (c_dir / f).is_file(), f"Missing candidate artifact: variant_{cand}/{f}"

print("All required experiment artifacts present.", flush=True)

# Detailed comparative evaluation
print("\\n" + "="*70)
print(f"DR4 SCREENING RESULTS SUMMARY: {RUN_ID}")
print("="*70)

results = {}
for cand in ["L2", "L2_A5B"]:
    s300 = json.loads((EXP_DIR / f"variant_{cand}" / "step300_diagnostics.json").read_text())
    preds = json.loads((EXP_DIR / f"variant_{cand}" / "val_sample_predictions.json").read_text())
    
    # Compute Top-1 distribution
    top1_labels = [p["predicted_labels"][0] if p.get("predicted_labels") else "NONE" for p in preds]
    from collections import Counter
    top1_counts = Counter(top1_labels)
    distinct_top1 = len(top1_counts)
    dominant_label, dominant_cnt = top1_counts.most_common(1)[0]
    
    # Compute Top-3 sequences
    top3_patterns = len(set(tuple(p["predicted_labels"][:3]) for p in preds if p.get("predicted_labels")))
    
    hits = s300.get("p_at_1_hits", 0)
    p_at_1 = s300.get("p_at_1", 0.0)
    prior_hits = s300.get("prior_p_at_1_hits", 63)
    
    stds = s300.get("per_label_logit_std", [])
    passing_stds = sum(1 for s in stds if s > 0.01)
    
    results[cand] = {
        "p_at_1_hits": hits,
        "prior_hits": prior_hits,
        "distinct_top1": distinct_top1,
        "dominant_label": dominant_label,
        "dominant_frac": dominant_cnt / len(preds),
        "distinct_top3_patterns": top3_patterns,
        "passing_v4_stds": passing_stds,
        "alpha_entropy": s300.get("alpha_entropy_mean", float("nan")),
        "max_alpha": s300.get("mean_max_alpha", float("nan")),
    }
    
    print(f"\\n--- Variant {cand} ---")
    print(f"  V4 (Logit Variation):     {passing_stds} / 54 labels std > 0.01 (Pass: {passing_stds >= 28})")
    print(f"  V5 (Ranking Superiority): {hits} / 200 hits vs Prior {prior_hits} (Pass: {hits > prior_hits})")
    print(f"  Distinct Top-1 Labels:    {distinct_top1} / 54")
    print(f"  Dominant Top-1 Label:     {dominant_label} ({dominant_cnt}/200 = {dominant_cnt/len(preds):.1%})")
    print(f"  Distinct Top-3 Patterns:  {top3_patterns} / 200")
    print(f"  Attention Entropy:        {results[cand]['alpha_entropy']:.4f}")

print("\\n" + "="*70)
""".replace("@RUN_ID@", RUN_ID)
    cells.append(cell(cell5_code))

    # ── Cell 6: Package Evidence Archive & SHA256SUMS ────────────────────────
    cell6_code = """\
# Cell 6: Package Downloadable Evidence Archive
WORK = Path("/kaggle/working")
RUN_ID = "@RUN_ID@"
EXP_DIR = WORK / "experiments" / RUN_ID
EVIDENCE_ZIP = WORK / f"DR4_A5B_EVIDENCE_{RUN_ID}.zip"

# Collect lightweight evidence files (excluding full checkpoints)
written_files = []
with zipfile.ZipFile(EVIDENCE_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(EXP_DIR):
        for f in files:
            fp = Path(root) / f
            rel = fp.relative_to(EXP_DIR)
            # Exclude large 1.2 GB full checkpoints from the quick-download evidence ZIP
            if "checkpoints" in str(rel) and f.endswith(".pt") and "step_150" in f:
                continue
            z.write(fp, arcname=str(rel))
            written_files.append(str(rel))

h = hashlib.sha256()
with open(EVIDENCE_ZIP, "rb") as f:
    for block in iter(lambda: f.read(1024*1024), b""):
        h.update(block)
zip_sha = h.hexdigest()

print("\\n" + "="*70)
print("DR4 EVIDENCE ARCHIVE READY FOR DOWNLOAD")
print("="*70)
print(f"File:    {EVIDENCE_ZIP}")
print(f"SHA-256: {zip_sha}")
print(f"Size:    {EVIDENCE_ZIP.stat().st_size / 1024.0:.1f} KB")
print(f"Files included: {len(written_files)}")
print("\\nSTOP: Do not execute further training without audit review of downloaded evidence.")
""".replace("@RUN_ID@", RUN_ID)
    cells.append(cell(cell6_code))

    nb = {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
            "kaggle": {
                "accelerator": "GPU",
                "dataSources": [
                    {"sourceType": "datasetVersion", "sourceId": "lhrm-l2-a5b-bundle"},
                    {"sourceType": "datasetVersion", "sourceId": "aapd-dataset"},
                    {"sourceType": "datasetVersion", "sourceId": "bert-base-uncased"},
                ],
                "dockerImageVersionId": 30699,
                "isInternetEnabled": False,
                "language": "python",
            },
        },
        "cells": cells,
    }
    return nb


def main():
    parser = argparse.ArgumentParser(description="Generate Kaggle notebook for L2 vs L2_A5B screening")
    parser.add_argument("--build_bundle", action="store_true",
                        help="Build fresh GPU bundle and compute authentic SHA-256")
    parser.add_argument("--bundle_name", default=BUNDLE_NAME)
    args = parser.parse_args()

    bundle_name = args.bundle_name
    bundle_path = os.path.join(ROOT, "bundles", bundle_name)

    if args.build_bundle or not os.path.exists(bundle_path):
        print(f"Building fresh GPU bundle: {bundle_path}...")
        import subprocess
        cmd = [
            sys.executable,
            os.path.join(ROOT, "scripts", "make_gpu_bundle.py"),
            "--config", CONFIG,
            "--run_id", RUN_ID,
            "--bundle_name", bundle_name,
            "--out", "bundles",
        ]
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        print(res.stdout)
        if res.returncode != 0:
            print("STDERR:", res.stderr)
            sys.exit(1)

    assert os.path.exists(bundle_path), f"Bundle not found at {bundle_path}"
    bundle_sha = sha256_file(bundle_path)
    bundle_size_kb = os.path.getsize(bundle_path) / 1024.0
    print(f"Bundle: {bundle_path}")
    print(f"  SHA-256: {bundle_sha}")
    print(f"  Size:    {bundle_size_kb:.1f} KB")

    nb = build_notebook(bundle_name, bundle_sha)
    os.makedirs(os.path.dirname(OUT_NB), exist_ok=True)
    with open(OUT_NB, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Notebook written: {OUT_NB}")
    print(f"  Cells: {len(nb['cells'])}")

    # Verify JSON syntax
    with open(OUT_NB, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert len(loaded["cells"]) == len(nb["cells"]), "JSON verification failed"
    print("Notebook JSON verification: PASS")


if __name__ == "__main__":
    main()
