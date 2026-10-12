#!/usr/bin/env python3
"""
scripts/gen_kaggle_notebook_dr4_decomposition.py

Generates notebooks/lhrm_dr4_trained_decomposition_kaggle.ipynb —
Dedicated, reproducible Kaggle GPU notebook for DR4 Post-Training Logit Decomposition:
1. Verifies and loads authentic trained checkpoints:
   - L2_STEP300_MODEL_ONLY.pt
   - L2_A5B_STEP300_MODEL_ONLY.pt
2. Acquired official BERT revision 86b5e0934494bd15c9632b12f734a8a67f723594.
3. Stages authentic AAPD Dev data (Test strictly quarantined) and frozen 200 Dev IDs.
4. Performs exact algebraic logit decomposition on real trained representations (h_T, h_L):
   - doc_term = alpha * (W . h_L)
   - sa_term  = W . (A @ h_L)
   - logits   = doc_term + sa_term + bias
5. Verifies numerical reconstruction (< 1e-5) and exact agreement with val_sample_predictions.
6. Computes all required metrics, duplicate target label analysis, and exports:
   - DR4_POSTTRAIN_DECOMPOSITION_RESULTS.json
   - DR4_POSTTRAIN_PER_LABEL_METRICS.csv
   - DR4_POSTTRAIN_DECOMPOSITION_REPORT.md
   - DR4_POSTTRAIN_DECOMPOSITION_EVIDENCE.zip
   - DR4_POSTTRAIN_SHA256SUMS.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BUNDLE_NAME = "lhrm_gpu_bundle_dr4_decomposition_20261012.zip"
OUT_NB = ROOT / "notebooks" / "lhrm_dr4_trained_decomposition_kaggle.ipynb"
DIAG_ID = "DR4_POSTTRAIN_DECOMPOSITION_20261012"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def cell(source_str: str, cell_type: str = "code") -> Dict[str, Any]:
    lines = source_str.rstrip("\n").split("\n")
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": [l + "\n" for l in lines[:-1]] + [lines[-1]],
        **({"outputs": [], "execution_count": None} if cell_type == "code" else {}),
    }


def build_notebook(bundle_name: str, bundle_sha256: str) -> Dict[str, Any]:
    cells = []

    # ── Cell 0: Markdown Protocol & Scope ─────────────────────────────────────
    cells.append(cell(f"""\
# LHRM DR4 — Post-Training Logit Decomposition Diagnostic ({DIAG_ID})

**Observational Post-Training Diagnostic on Trained Step-300 Checkpoints**

This session performs exact algebraic logit decomposition on real representations ($h_T, h_L$) extracted from the trained model states:
- `L2_STEP300_MODEL_ONLY.pt`
- `L2_A5B_STEP300_MODEL_ONLY.pt`

### Diagnostic Objectives:
1. **Algebraic Logit Decomposition:**
   - $\\text{{doc\\_term}}[b, i] = \\alpha[b, i] \\cdot (W[i] \\cdot h_L[i])$ (document-dependent)
   - $\\text{{sa\\_term}}[b, i] = W[i] \\cdot SA[i]$ (static label self-attention)
   - $\\text{{logits}}[b, i] = \\text{{doc\\_term}}[b, i] + \\text{{sa\\_term}}[b, i] + \\text{{bias}}$
2. **Reconstruction Verification:** Verify $\\max |\\text{{logits}} - (\\text{{doc\\_term}} + \\text{{sa\\_term}} + \\text{{bias}})| < 10^{{-5}}$ and verify agreement with original saved predictions.
3. **Signal Attenuation Audit:** Measure SA-to-document magnitude ratio, cross-document variance of document terms, and attention entropy.
4. **Duplicate Target Labels Analysis:** Compare `cmp-lg` (idx 0) vs `cond-mat.dis-nn` (idx 1) static vs document contributions.
5. **Strict Quarantine:** Test split is strictly quarantined (never read or evaluated). No training or optimizer updates occur.

**Bundle Name:** `{bundle_name}`  
**Bundle SHA-256:** `{bundle_sha256}`
""", cell_type="markdown"))

    # ── Cell 1: Environment Check & Official BERT Setup ───────────────────────
    cells.append(cell("""\
# Cell 1: Environment check & Official BERT Acquisition
import os, sys, time, json, hashlib, shutil, gc
from pathlib import Path
import torch

print("=== DR4 Decomposition Diagnostic: Environment & Device Gate ===", flush=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__, " | CUDA:", torch.version.cuda)
print("Device:", device, flush=True)

import transformers, numpy as np
print(f"Transformers: {transformers.__version__} | NumPy: {np.__version__}")

PINNED_COMMIT = "86b5e0934494bd15c9632b12f734a8a67f723594"
REPO = "google-bert/bert-base-uncased"
BERT_DIR = Path("/kaggle/working/google_bert_bert_base_uncased_official")
SOURCE_META = Path("/kaggle/working/OFFICIAL_BERT_SOURCE.json")
WEIGHTS = ("model.safetensors", "pytorch_model.bin")

def sha256_bytes(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def complete_official_cache():
    return (
        (BERT_DIR / "config.json").is_file()
        and (BERT_DIR / "vocab.txt").is_file()
        and any((BERT_DIR / w).is_file() for w in WEIGHTS)
        and SOURCE_META.is_file()
    )

if not complete_official_cache():
    if BERT_DIR.exists():
        shutil.rmtree(BERT_DIR)
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ.pop(key, None)
    print("Acquiring official pretrained BERT from", REPO, "@", PINNED_COMMIT, "...", flush=True)
    from huggingface_hub import hf_hub_download
    from transformers import BertModel, BertTokenizer
    
    tokenizer = BertTokenizer.from_pretrained(REPO, revision=PINNED_COMMIT)
    model = BertModel.from_pretrained(REPO, revision=PINNED_COMMIT)
    assert tokenizer.vocab_size == 30522 and model.config.hidden_size == 768
    
    tmp = BERT_DIR.with_name(BERT_DIR.name + "_DOWNLOADING")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        model.save_pretrained(str(tmp), safe_serialization=True)
        tokenizer.save_pretrained(str(tmp))
        official_vocab = Path(hf_hub_download(repo_id=REPO, filename="vocab.txt", revision=PINNED_COMMIT))
        shutil.copyfile(official_vocab, tmp / "vocab.txt")
        assert (tmp / "config.json").is_file() and (tmp / "vocab.txt").is_file()
        weight_file = next((tmp / w for w in WEIGHTS if (tmp / w).is_file()), None)
        assert weight_file is not None
        meta = {
            "repo_id": REPO,
            "source_revision": PINNED_COMMIT,
            "saved_weights_file": weight_file.name,
            "saved_weights_sha256": sha256_bytes(weight_file),
            "bert_config_sha256": sha256_bytes(tmp / "config.json"),
            "vocab_sha256": sha256_bytes(tmp / "vocab.txt"),
            "vocab_size": 30522,
            "hidden_size": 768,
        }
        BERT_DIR.parent.mkdir(parents=True, exist_ok=True)
        tmp.rename(BERT_DIR)
        SOURCE_META.write_text(json.dumps(meta, indent=2) + "\\n", encoding="utf-8")
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
        del model, tokenizer
        gc.collect()
else:
    print("Official BERT already present in working directory.", flush=True)

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
print("OFFICIAL BERT READY & VERIFIED:", BERT_DIR, flush=True)
"""))

    # ── Cell 2: Verify & Stage Bundle, Checkpoints, and Dev Data ──────────────
    cell2_code = """\
# Cell 2: Verify and Stage Bundle, Checkpoints, and Dev Data
from pathlib import Path
import os, json, hashlib, zipfile, shutil, sys

WORK = Path("/kaggle/working")
INP = Path("/kaggle/input")
BUNDLE_NAME = "@BUNDLE_NAME@"
EXPECTED_BUNDLE_SHA = "@BUNDLE_SHA256@"

EXPECTED_CKPT_SHA = {
    "L2_STEP300_MODEL_ONLY.pt": "99a06f1e7ba5bd72f3d855b61672b95a52f97f83e8db519e09b7992f2ca48e49",
    "L2_A5B_STEP300_MODEL_ONLY.pt": "b80852c7daafb626860dc2c981290eb2f7905e5303531c551b2fa14e0bb27022",
}

EXPECTED_DEV_CSV_SHA = "a85f612d4c1cb2be6d1de94cbcd9468cba97d221f826014264f4d279c82e220b"
EXPECTED_LABEL_SHA = "a3d662f3920259a94ed61d83f98f21cabe7f908d98a5662ccd9adc2413cc000b"

def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

# 1. Verify and Extract Bundle (Fail-Closed)
bundle_path = None
for root, _, files in os.walk(INP):
    if BUNDLE_NAME in files:
        p = Path(root) / BUNDLE_NAME
        if p.is_file():
            bundle_path = p
            break

if bundle_path is None:
    raise FileNotFoundError(f"FAIL-CLOSED: Exact bundle '{BUNDLE_NAME}' not found.")

actual_bundle_sha = sha_file(bundle_path)
assert actual_bundle_sha == EXPECTED_BUNDLE_SHA, f"Bundle SHA mismatch: {actual_bundle_sha} != {EXPECTED_BUNDLE_SHA}"
print(f"Bundle VERIFIED: {bundle_path} (SHA: {actual_bundle_sha})")

with zipfile.ZipFile(bundle_path, "r") as z:
    z.extractall(WORK)
print("Bundle extracted to /kaggle/working.")

# 2. Locate and Verify Trained Model Checkpoints
found_ckpts = {}
for name, exp_sha in EXPECTED_CKPT_SHA.items():
    for root, _, files in os.walk(INP):
        if name in files:
            p = Path(root) / name
            if p.is_file() and sha_file(p) == exp_sha:
                found_ckpts[name] = p
                break
    if name not in found_ckpts:
        raise FileNotFoundError(f"FAIL-CLOSED: Authentic checkpoint {name} matching SHA {exp_sha} not found in /kaggle/input.")

print(f"Checkpoints VERIFIED: {list(found_ckpts.keys())}")

# 3. Locate and Stage AAPD Dev CSV Only (Test strictly quarantined)
dev_source = None
for root, _, files in os.walk(INP):
    if "dev.csv" in files:
        p = Path(root) / "dev.csv"
        if sha_file(p) == EXPECTED_DEV_CSV_SHA:
            dev_source = p
            break

if dev_source is None:
    # Check inside attached ZIPs
    for root, _, files in os.walk(INP):
        for f in files:
            if f.endswith(".zip"):
                try:
                    with zipfile.ZipFile(Path(root) / f) as z:
                        for m in z.namelist():
                            if m.endswith("dev.csv"):
                                with z.open(m) as zf:
                                    h = hashlib.sha256()
                                    for block in iter(lambda: zf.read(1024*1024), b""):
                                        h.update(block)
                                    if h.hexdigest() == EXPECTED_DEV_CSV_SHA:
                                        dev_source = (Path(root) / f, m)
                                        break
                except Exception:
                    pass

if dev_source is None:
    raise FileNotFoundError("FAIL-CLOSED: Authentic AAPD dev.csv not found in /kaggle/input.")

# Stage dev.csv into temporary zip for load_aapd_split
staged_zip = WORK / "AAPD_DEV_ONLY.zip"
with zipfile.ZipFile(staged_zip, "w") as out_z:
    with out_z.open("AAPD/dev.csv", "w") as dest:
        if isinstance(dev_source, tuple):
            with zipfile.ZipFile(dev_source[0]) as src_z, src_z.open(dev_source[1]) as src:
                shutil.copyfileobj(src, dest)
        else:
            with open(dev_source, "rb") as src:
                shutil.copyfileobj(src, dest)

if str(WORK / "src") not in sys.path:
    sys.path.insert(0, str(WORK / "src"))

from lhrm_baseline.data import load_aapd_split
dev_texts, dev_labels, cols = load_aapd_split(str(staged_zip), "dev")
assert len(dev_texts) == 1000 and dev_labels.shape == (1000, 54)
label_sha = hashlib.sha256(json.dumps(cols).encode()).hexdigest()
assert label_sha == EXPECTED_LABEL_SHA, f"Label order SHA mismatch: {label_sha}"

# Verify frozen 200 Dev IDs
val_ids_file = WORK / "experiments" / "RUN_0" / "run0_val_sample_ids.json"
assert val_ids_file.is_file(), f"Missing frozen IDs file: {val_ids_file}"
frozen_ids = json.loads(val_ids_file.read_text())
assert len(frozen_ids) == 200

# Subsample frozen 200 documents
eval_texts = [dev_texts[i] for i in frozen_ids]
eval_labels = dev_labels[frozen_ids]
print(f"Data GATES PASSED: 200 Dev documents staged, 54 labels verified (Test split strictly quarantined).", flush=True)
""".replace("@BUNDLE_NAME@", bundle_name).replace("@BUNDLE_SHA256@", bundle_sha256)
    cells.append(cell(cell2_code))

    # ── Cell 3: Execute Decomposition Diagnostic ──────────────────────────────
    cell3_code = """\
# Cell 3: Execute Full Algebraic Decomposition on Real Trained Representations
import torch, gc
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader
from lhrm_baseline.data import MultiLabelDataset, collate_fn, tokenize_labels
from lhrm_baseline.labels import build_all_label_texts
from lhrm_baseline.model import LHRMModel, get_cls_embedding

sys.path.insert(0, str(WORK / "scripts"))
from probe_dr4_trained_logit_decomposition import (
    decompose_l2_csam_logits,
    decompose_a5b_csam_logits,
    compute_decomposition_statistics,
    export_decomposition_deliverables,
)

print("=== Executing Trained-Checkpoint Logit Decomposition ===", flush=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained(str(BERT_DIR), local_files_only=True)
defs_path = WORK / "Label_Definition" / "arxiv_labels_definitions.json"
label_texts_list = build_all_label_texts(cols, str(defs_path), dataset="aapd")
label_ids, label_mask = tokenize_labels(label_texts_list, tokenizer, max_len=64, device=device)

eval_ds = MultiLabelDataset(eval_texts, eval_labels, tokenizer, max_len=512)
eval_loader = DataLoader(eval_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)

# Ground truth true_labels per document for hit validation
true_labels_list = []
for row in eval_labels:
    true_labels_list.append([cols[j] for j in range(len(cols)) if row[j] > 0.5])

# 1. Run for L2
print("\\n--- Decomposing Variant L2 ---", flush=True)
bert_l2 = BertModel.from_pretrained(str(BERT_DIR), local_files_only=True).to(device)
model_l2 = LHRMModel(bert_l2, num_labels=54, variant="L2").to(device)
ckpt_l2 = torch.load(found_ckpts["L2_STEP300_MODEL_ONLY.pt"], map_location=device)
model_l2.load_state_dict(ckpt_l2["model_state"], strict=True)
model_l2.eval()

# Encode labels
with torch.no_grad():
    label_out_l2 = model_l2.bert(label_ids, attention_mask=label_mask)
    h_L_l2 = get_cls_embedding(label_out_l2)  # [54, H]

# Encode documents and decompose in batches
doc_h_T_l2 = []
with torch.no_grad():
    for batch in eval_loader:
        b_ids = batch["input_ids"].to(device)
        b_mask = batch["attention_mask"].to(device)
        out = model_l2.bert(b_ids, attention_mask=b_mask)
        doc_h_T_l2.append(get_cls_embedding(out))
h_T_l2 = torch.cat(doc_h_T_l2, dim=0)  # [200, H]

w_l2 = model_l2.csam.W.weight.data
b_l2 = model_l2.csam.W.bias.data
decomp_l2 = decompose_l2_csam_logits(h_T_l2, h_L_l2, w_l2, b_l2)
l2_stats = compute_decomposition_statistics(decomp_l2, cols, true_labels_list)
print(f"L2 Decomposition PASS: Recon error = {decomp_l2['max_recon_error']:.2e}, SA/Doc ratio = {l2_stats['sa_to_doc_ratio']:.2f}x")

del model_l2, bert_l2, h_T_l2, h_L_l2
torch.cuda.empty_cache()
gc.collect()

# 2. Run for L2_A5B
print("\\n--- Decomposing Variant L2_A5B ---", flush=True)
bert_a5b = BertModel.from_pretrained(str(BERT_DIR), local_files_only=True).to(device)
model_a5b = LHRMModel(bert_a5b, num_labels=54, variant="L2_A5B").to(device)
ckpt_a5b = torch.load(found_ckpts["L2_A5B_STEP300_MODEL_ONLY.pt"], map_location=device)
model_a5b.load_state_dict(ckpt_a5b["model_state"], strict=True)
model_a5b.eval()

# Encode labels
with torch.no_grad():
    label_out_a5b = model_a5b.bert(label_ids, attention_mask=label_mask)
    h_L_a5b = get_cls_embedding(label_out_a5b)  # [54, H]

# Encode documents and decompose in batches
doc_h_T_a5b = []
with torch.no_grad():
    for batch in eval_loader:
        b_ids = batch["input_ids"].to(device)
        b_mask = batch["attention_mask"].to(device)
        out = model_a5b.bert(b_ids, attention_mask=b_mask)
        doc_h_T_a5b.append(get_cls_embedding(out))
h_T_a5b = torch.cat(doc_h_T_a5b, dim=0)  # [200, H]

W_a5b = model_a5b.csam.W.data
b_a5b = model_a5b.csam.bias.data
decomp_a5b = decompose_a5b_csam_logits(h_T_a5b, h_L_a5b, W_a5b, b_a5b)
a5b_stats = compute_decomposition_statistics(decomp_a5b, cols, true_labels_list)
print(f"L2_A5B Decomposition PASS: Recon error = {decomp_a5b['max_recon_error']:.2e}, SA/Doc ratio = {a5b_stats['sa_to_doc_ratio']:.2f}x")

# Weight analysis
cos_sim_matrix = F.cosine_similarity(W_a5b.unsqueeze(1), W_a5b.unsqueeze(0), dim=2)
weight_analysis = {
    "a5b_w_norm_mean": float(torch.norm(W_a5b, dim=1).mean().item()),
    "a5b_w_norm_std": float(torch.norm(W_a5b, dim=1).std().item()),
    "a5b_mean_cosine_similarity": float(cos_sim_matrix.mean().item()),
    "a5b_min_cosine_similarity": float(cos_sim_matrix.min().item()),
    "duplicate_targets_cosine_similarity": float(cos_sim_matrix[0, 1].item()),
}

del model_a5b, bert_a5b, h_T_a5b, h_L_a5b
torch.cuda.empty_cache()
gc.collect()

# 3. Export all deliverables
out_dir = WORK / "DR4_POSTTRAIN_DECOMPOSITION_OUTPUT"
paths = export_decomposition_deliverables(l2_stats, a5b_stats, out_dir, weight_analysis)

# Also copy evidence ZIP to /kaggle/working/
shutil.copy2(paths["evidence_zip"], WORK / Path(paths["evidence_zip"]).name)
shutil.copy2(paths["checksums"], WORK / Path(paths["checksums"]).name)
shutil.copy2(paths["report_md"], WORK / Path(paths["report_md"]).name)

print("\\n" + "="*70)
print("DR4 DECOMPOSITION DIAGNOSTIC COMPLETE")
print("="*70)
print(Path(paths["report_md"]).read_text(encoding="utf-8"))
"""
    cells.append(cell(cell3_code))

    nb = {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
        },
        "cells": cells,
    }
    return nb


def build_decomposition_bundle(out_zip: Path) -> str:
    """Builds lightweight bundle containing code and configs for Kaggle diagnostic execution."""
    to_include = [
        (ROOT / "run_baseline.py", "run_baseline.py"),
        (ROOT / "scripts" / "probe_dr4_trained_logit_decomposition.py", "scripts/probe_dr4_trained_logit_decomposition.py"),
        (ROOT / "Label_Definition" / "arxiv_labels_definitions.json", "Label_Definition/arxiv_labels_definitions.json"),
        (ROOT / "experiments" / "RUN_0" / "run0_val_sample_ids.json", "experiments/RUN_0/run0_val_sample_ids.json"),
        (ROOT / "docs" / "experiments" / "DR4_A5B_PREREGISTRATION.md", "docs/experiments/DR4_A5B_PREREGISTRATION.md"),
    ]

    # Include src/lhrm_baseline
    src_dir = ROOT / "src"
    for base, _, files in os.walk(src_dir):
        for f in sorted(files):
            if f.endswith(".py"):
                fp = Path(base) / f
                rel = fp.relative_to(ROOT).as_posix()
                to_include.append((fp, rel))

    # Manifest
    manifest = {
        "bundle_name": out_zip.name,
        "diagnostic_id": DIAG_ID,
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": {},
    }
    for fp, rel in to_include:
        if fp.is_file():
            manifest["files"][rel] = sha256_file(fp)

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp, rel in to_include:
            if fp.is_file():
                zf.write(fp, arcname=rel)
        zf.writestr("BUNDLE_MANIFEST.json", json.dumps(manifest, indent=2))

    return sha256_file(out_zip)


def main():
    parser = argparse.ArgumentParser(description="Generate Kaggle notebook and bundle for DR4 logit decomposition")
    parser.add_argument("--bundle_name", default=BUNDLE_NAME)
    args = parser.parse_args()

    bundle_path = ROOT / "bundles" / args.bundle_name
    print(f"Building decomposition bundle: {bundle_path}...")
    bundle_sha = build_decomposition_bundle(bundle_path)
    bundle_size_kb = bundle_path.stat().st_size / 1024.0
    print(f"Bundle built: {bundle_path}")
    print(f"  SHA-256: {bundle_sha}")
    print(f"  Size:    {bundle_size_kb:.1f} KB")

    nb = build_notebook(args.bundle_name, bundle_sha)
    OUT_NB.parent.mkdir(parents=True, exist_ok=True)
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
