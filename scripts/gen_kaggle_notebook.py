#!/usr/bin/env python3
"""
Write notebooks/lhrm_run0_kaggle_cli.ipynb — a clean Kaggle notebook that
delegates ALL training/diagnostic/decision logic to run_baseline.py run0.

This notebook has no duplicate diagnostic code. It:
1. Sets up the environment and mounts the bundle
2. Verifies the bundle SHA256
3. Calls: python run_baseline.py run0 --config configs/final/aapd_baseline.yaml
          --run_id RUN_0_P1F --steps 300 --candidates L,L2
4. Downloads all artifacts

Run this script once locally to (re)generate the notebook.
"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_NB = os.path.join(ROOT, "notebooks", "lhrm_run0_kaggle_cli.ipynb")

# ── Bundle SHA (Phase 1F build) ───────────────────────────────────────────────
BUNDLE_SHA256 = "4b3898c8096a20985ad6c0cb357ef671dfa4b8a43d53c42de058eb255f3b3753"
BUNDLE_NAME   = "lhrm_gpu_bundle_20261008_135414.zip"
RUN_ID        = "RUN_0_P1F"
CONFIG        = "configs/final/aapd_baseline.yaml"
CANDIDATES    = "L,L2"
N_STEPS       = 300


def cell(source_str, cell_type="code"):
    lines = source_str.rstrip("\n").split("\n")
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": [l + "\n" for l in lines[:-1]] + [lines[-1]],
        **({"outputs": [], "execution_count": None} if cell_type == "code" else {}),
    }


CELLS = []

# ── Cell 0: README ────────────────────────────────────────────────────────────
CELLS.append(cell("""\
# LHRM Baseline — RUN_0 (Phase 1F corrected, CLI-driven)

**Rules:** no test split; no hyperparameter tuning; no manual decision override.
All training, diagnostics and the V1-V5 decision are executed by `run_baseline.py`.

**Bundle SHA256:** `{}` (verified in Cell 2)
""".format(BUNDLE_SHA256), cell_type="markdown"))

# ── Cell 1: Environment check ─────────────────────────────────────────────────
CELLS.append(cell("""\
# Cell 1: Environment check
import subprocess, sys, os, torch

print("Python:", sys.version)
print("PyTorch:", torch.__version__, "  CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
if not torch.cuda.is_available():
    raise RuntimeError("GPU GATE: No CUDA GPU — cannot run RUN_0")

# Check scikit-learn version (must be ≥ 1.0 for HRRM)
import sklearn; print("sklearn:", sklearn.__version__)
import transformers; print("transformers:", transformers.__version__)
import numpy as np; print("numpy:", np.__version__)
"""))

# ── Cell 2: Mount bundle and verify SHA256 ────────────────────────────────────
CELLS.append(cell("""\
# Cell 2: Mount bundle and verify SHA256
import hashlib, zipfile, os, sys

BUNDLE_SHA256 = "{sha}"
BUNDLE_NAME   = "{bname}"

# Kaggle dataset mount point (adjust if different)
KAGGLE_BUNDLE_DIR = "/kaggle/input/lhrm-baseline-bundle"
BUNDLE_ZIP = os.path.join(KAGGLE_BUNDLE_DIR, BUNDLE_NAME)

if not os.path.exists(BUNDLE_ZIP):
    # Try finding the zip in the dataset directory
    for f in os.listdir(KAGGLE_BUNDLE_DIR):
        if f.endswith(".zip"):
            BUNDLE_ZIP = os.path.join(KAGGLE_BUNDLE_DIR, f)
            print("Found bundle:", BUNDLE_ZIP)
            break

if not os.path.exists(BUNDLE_ZIP):
    raise FileNotFoundError(
        "Bundle ZIP not found at {{}}. "
        "Did you attach the dataset 'lhrm-baseline-bundle'?".format(BUNDLE_ZIP))

# Verify SHA256
h = hashlib.sha256()
with open(BUNDLE_ZIP, "rb") as f:
    for chunk in iter(lambda: f.read(65536), b""):
        h.update(chunk)
actual_sha = h.hexdigest()
print("Bundle SHA256:", actual_sha)
if actual_sha != BUNDLE_SHA256:
    raise AssertionError(
        "BUNDLE SHA256 MISMATCH!\\n  got:      {{}}\\n  expected: {{}}".format(
            actual_sha, BUNDLE_SHA256))
print("Bundle SHA256: OK")

# Extract to /kaggle/working/bundle/
BUNDLE_ROOT = "/kaggle/working/bundle"
os.makedirs(BUNDLE_ROOT, exist_ok=True)
with zipfile.ZipFile(BUNDLE_ZIP, "r") as zf:
    zf.extractall(BUNDLE_ROOT)
print("Extracted to:", BUNDLE_ROOT)
os.listdir(BUNDLE_ROOT)
""".format(sha=BUNDLE_SHA256, bname=BUNDLE_NAME)))

# ── Cell 3: Mount AAPD data ───────────────────────────────────────────────────
CELLS.append(cell("""\
# Cell 3: Mount AAPD data (train + dev only — test is NOT loaded during RUN_0)
import os, shutil, zipfile

# AAPD dataset mount (attach the Zenodo AAPD.zip as a Kaggle dataset)
AAPD_DATASET_DIR = "/kaggle/input/aapd-zenodo"   # adjust to your dataset name

# Locate AAPD.zip
AAPD_ZIP = None
for candidate in [
    os.path.join(AAPD_DATASET_DIR, "AAPD.zip"),
    "/kaggle/input/aapd/AAPD.zip",
]:
    if os.path.exists(candidate):
        AAPD_ZIP = candidate
        break

if AAPD_ZIP is None:
    raise FileNotFoundError(
        "AAPD.zip not found. Attach the Zenodo AAPD dataset and adjust AAPD_DATASET_DIR.")

print("AAPD.zip:", AAPD_ZIP)

# Place it where the bundle expects it (Data/AAPD.zip)
DATA_DIR = os.path.join(BUNDLE_ROOT, "Data")
os.makedirs(DATA_DIR, exist_ok=True)
dest_zip = os.path.join(DATA_DIR, "AAPD.zip")
if not os.path.exists(dest_zip):
    shutil.copy2(AAPD_ZIP, dest_zip)
    print("Copied AAPD.zip to:", dest_zip)
else:
    print("AAPD.zip already present:", dest_zip)
"""))

# ── Cell 4: Prepare AAPD (if not already done in bundle) ─────────────────────
CELLS.append(cell("""\
# Cell 4: Prepare AAPD splits (train + dev only; test never loaded during RUN_0)
import subprocess, sys, os

BUNDLE_ROOT = "/kaggle/working/bundle"
PROCESSED = os.path.join(BUNDLE_ROOT, "data", "processed", "AAPD")

if os.path.exists(os.path.join(PROCESSED, "train_labels.npy")):
    print("Processed data already exists:", PROCESSED)
else:
    print("Running prepare...")
    result = subprocess.run(
        [sys.executable, "run_baseline.py", "prepare",
         "--config", "configs/final/aapd_baseline.yaml"],
        cwd=BUNDLE_ROOT, capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        raise RuntimeError("prepare failed with code {}".format(result.returncode))

# Verify processed files exist
for fname in ["train_labels.npy", "dev_labels.npy", "labels.json", "train_texts.json", "dev_texts.json"]:
    p = os.path.join(PROCESSED, fname)
    assert os.path.exists(p), "Missing: " + p
print("Processed data OK:", os.listdir(PROCESSED))

# CRITICAL: test split must NOT be present in bundle working dir
# (test_labels.npy may exist in the processed dir from prepare but is never passed to run0)
print("Test split isolation: run_baseline.py run0 never loads test_* files.")
"""))

# ── Cell 5: Mount saved val IDs and BERT ─────────────────────────────────────
CELLS.append(cell("""\
# Cell 5: Mount BERT weights and copy pre-registered val sample IDs
import os, shutil, json, hashlib

BUNDLE_ROOT = "/kaggle/working/bundle"

# BERT from Kaggle dataset (attach bert-base-uncased dataset)
BERT_DATASET = "/kaggle/input/bert-base-uncased"
if not os.path.isdir(BERT_DATASET):
    print("WARNING: bert-base-uncased not attached as dataset; will download.")
else:
    # Symlink into bundle root so transformers finds it
    BERT_DEST = os.path.join(BUNDLE_ROOT, "bert-base-uncased")
    if not os.path.exists(BERT_DEST):
        shutil.copytree(BERT_DATASET, BERT_DEST)
    print("BERT cached at:", BERT_DEST)

# Copy pre-registered val sample IDs into the new run dir to lock them
RUN_ID = "{run_id}"
NEW_RUN_DIR = os.path.join(BUNDLE_ROOT, "experiments", RUN_ID)
os.makedirs(NEW_RUN_DIR, exist_ok=True)

REF_IDS_PATH = os.path.join(BUNDLE_ROOT, "experiments", "RUN_0", "run0_val_sample_ids.json")
NEW_IDS_PATH = os.path.join(NEW_RUN_DIR, "run0_val_sample_ids.json")

if os.path.exists(REF_IDS_PATH) and not os.path.exists(NEW_IDS_PATH):
    shutil.copy2(REF_IDS_PATH, NEW_IDS_PATH)
    print("Copied pre-registered val IDs to:", NEW_IDS_PATH)
elif os.path.exists(NEW_IDS_PATH):
    print("Val IDs already in run dir:", NEW_IDS_PATH)
else:
    print("WARNING: No pre-registered val IDs found at", REF_IDS_PATH)
    print("  run_baseline.py will generate them from seed=42 (first run).")

# Verify val IDs if present
if os.path.exists(NEW_IDS_PATH):
    ids = json.load(open(NEW_IDS_PATH))
    assert len(ids) == 200, "Expected 200 val IDs; got {{}}".format(len(ids))
    assert ids[:5] == [3, 6, 11, 25, 27], "Val IDs mismatch (first 5)"
    print("Val IDs: 200 docs, first 5:", ids[:5], " OK")
""".format(run_id=RUN_ID)))

# ── Cell 6: RUN_0 via CLI ─────────────────────────────────────────────────────
CELLS.append(cell("""\
# Cell 6: RUN_0 via CLI (single source of truth)
# This cell calls run_baseline.py which handles:
#   - label-order hash verification
#   - GPU-vs-CPU step-0 preflight cross-check (fails-closed)
#   - 300-step training for L then L2
#   - V1-V5 evaluation via diagnostics.py
#   - scripted decision via diagnostics.run0_decision
#   - all artifact writes + SHA256SUMS.txt
import subprocess, sys, os, time

BUNDLE_ROOT = "/kaggle/working/bundle"
RUN_ID      = "{run_id}"
CONFIG      = "{config}"
CANDIDATES  = "{candidates}"
N_STEPS     = {n_steps}

t_start = time.time()
result = subprocess.run(
    [sys.executable, "run_baseline.py", "run0",
     "--config", CONFIG,
     "--run_id", RUN_ID,
     "--steps", str(N_STEPS),
     "--candidates", CANDIDATES],
    cwd=BUNDLE_ROOT,
    # Stream output so we can see progress
    stdout=None, stderr=None,   # inherit stdout/stderr for live logging
)
elapsed = time.time() - t_start
print("\\nExit code:", result.returncode, "  Elapsed: {{:.1f}}s".format(elapsed))

if result.returncode == 2:
    raise RuntimeError(
        "PREFLIGHT CROSS-CHECK FAILED — see preflight_fail_*.json in {{}}".format(
            os.path.join(BUNDLE_ROOT, "experiments", RUN_ID)))
if result.returncode != 0:
    raise RuntimeError(
        "run_baseline.py run0 failed with code {{}}".format(result.returncode))
print("RUN_0 CLI completed OK.")
""".format(run_id=RUN_ID, config=CONFIG, candidates=CANDIDATES, n_steps=N_STEPS)))

# ── Cell 7: Artifact verification ────────────────────────────────────────────
CELLS.append(cell("""\
# Cell 7: Verify artifacts exist and SHA256SUMS.txt is consistent
import os, json, hashlib

BUNDLE_ROOT = "/kaggle/working/bundle"
RUN_ID      = "{run_id}"
OUT_BASE    = os.path.join(BUNDLE_ROOT, "experiments", RUN_ID)

REQUIRED = [
    "run0_decision.json",
    "run0_val_sample_ids.json",
    "resolved_config.json",
    "env.json",
    "SHA256SUMS.txt",
    "variant_L/step0_diagnostics.json",
    "variant_L/step300_diagnostics.json",
    "variant_L/train.log",
    "variant_L/val_sample_predictions.json",
    "variant_L/checkpoints/step_150.pt",
    "variant_L/checkpoints/step_300.pt",
    "variant_L2/step0_diagnostics.json",
    "variant_L2/step300_diagnostics.json",
    "variant_L2/train.log",
    "variant_L2/val_sample_predictions.json",
    "variant_L2/checkpoints/step_150.pt",
    "variant_L2/checkpoints/step_300.pt",
]

missing = []
for rel in REQUIRED:
    p = os.path.join(OUT_BASE, rel)
    if not os.path.exists(p):
        missing.append(rel)

if missing:
    raise AssertionError("MISSING ARTIFACTS:\\n" + "\\n".join("  " + m for m in missing))
print("All {{}} required artifacts present.".format(len(REQUIRED)))

# Verify SHA256SUMS.txt
sums_path = os.path.join(OUT_BASE, "SHA256SUMS.txt")
sums = {{}}
for line in open(sums_path):
    line = line.strip()
    if line:
        sha, rel = line.split("  ", 1)
        sums[rel] = sha

failures = []
for rel, expected_sha in sums.items():
    fp = os.path.join(OUT_BASE, rel)
    if not os.path.exists(fp):
        failures.append("MISSING: " + rel)
        continue
    h = hashlib.sha256()
    with open(fp, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected_sha:
        failures.append("SHA MISMATCH: {{}} got={{}} expected={{}}".format(rel, actual[:12], expected_sha[:12]))

if failures:
    raise AssertionError("SHA256SUMS verification failed:\\n" + "\\n".join(failures))
print("SHA256SUMS.txt verified: {{}} files OK".format(len(sums)))

# Print decision
decision = json.load(open(os.path.join(OUT_BASE, "run0_decision.json")))
print("\\n=== SCRIPTED DECISION ===")
print("Status:", decision["status"])
print("Chosen variant:", decision.get("chosen_variant"))
print("Reason:", decision.get("reason"))
""".format(run_id=RUN_ID)))

# ── Cell 8: Print key diagnostics summary ─────────────────────────────────────
CELLS.append(cell("""\
# Cell 8: Print diagnostic summary for both variants
import os, json

BUNDLE_ROOT = "/kaggle/working/bundle"
RUN_ID      = "{run_id}"
OUT_BASE    = os.path.join(BUNDLE_ROOT, "experiments", RUN_ID)

for variant in ("L", "L2"):
    cand_dir = os.path.join(OUT_BASE, "variant_" + variant)
    s0   = json.load(open(os.path.join(cand_dir, "step0_diagnostics.json")))
    s300 = json.load(open(os.path.join(cand_dir, "step300_diagnostics.json")))
    print("\\n--- Variant {{}} ---".format(variant))
    print("  Step 0:   mean_max_alpha={{:.6f}}  one_hot={{:.4f}}  S_mean={{:.2f}}  P@1={{}}/{{}}".format(
        s0.get("mean_max_alpha",0), s0.get("alpha_one_hot_frac",0),
        s0.get("S_mean",float("nan")), s0.get("p_at_1_hits",0), s0.get("n_docs",0)))
    print("  Step 300: mean_max_alpha={{:.6f}}  one_hot={{:.4f}}  logit_std_frac={{:.4f}}  P@1={{}}/{{}}  prior={{}}".format(
        s300.get("mean_max_alpha",0), s300.get("alpha_one_hot_frac",0),
        s300.get("logit_std_frac",0), s300.get("p_at_1_hits",0),
        s300.get("n_docs",0), s300.get("prior_p_at_1_hits",0)))
    print("  V5 (P@1 > prior):", s300.get("p_at_1_hits",0) > s300.get("prior_p_at_1_hits",0))
""".format(run_id=RUN_ID)))

# ── Cell 9: Package for download ──────────────────────────────────────────────
CELLS.append(cell("""\
# Cell 9: Package artifacts for download
import os, zipfile, hashlib, json, time

BUNDLE_ROOT = "/kaggle/working/bundle"
RUN_ID      = "{run_id}"
OUT_BASE    = os.path.join(BUNDLE_ROOT, "experiments", RUN_ID)
ZIP_OUT     = "/kaggle/working/run0_artifacts_{{}}_{{}}.zip".format(
    RUN_ID, time.strftime("%Y%m%d_%H%M%S"))

with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as zf:
    for dirpath, _, fnames in os.walk(OUT_BASE):
        for fname in sorted(fnames):
            full = os.path.join(dirpath, fname)
            rel  = os.path.relpath(full, OUT_BASE)
            zf.write(full, rel)

h = hashlib.sha256()
with open(ZIP_OUT, "rb") as f:
    for chunk in iter(lambda: f.read(65536), b""):
        h.update(chunk)
print("Artifact ZIP:", ZIP_OUT)
print("SHA256:", h.hexdigest())
print("Size: {{:.1f}} MB".format(os.path.getsize(ZIP_OUT) / 1e6))
print("\\nDownload this file and run locally:")
print("  python run_baseline.py import_artifacts --path /path/to/extracted/")
""".format(run_id=RUN_ID)))

# ── Assemble notebook ─────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 4,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "kaggle": {
            "accelerator": "GPU",
            "dataSources": [
                {"sourceType": "datasetVersion", "sourceId": "lhrm-baseline-bundle"},
                {"sourceType": "datasetVersion", "sourceId": "aapd-zenodo"},
                {"sourceType": "datasetVersion", "sourceId": "bert-base-uncased"},
            ],
            "dockerImageVersionId": 30699,
            "isInternetEnabled": False,
            "language": "python",
        },
    },
    "cells": CELLS,
}

os.makedirs(os.path.dirname(OUT_NB), exist_ok=True)
with open(OUT_NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Notebook written:", OUT_NB)
print("Cells:", len(CELLS))
