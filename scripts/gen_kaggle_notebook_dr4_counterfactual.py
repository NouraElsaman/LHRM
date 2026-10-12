#!/usr/bin/env python3
"""
scripts/gen_kaggle_notebook_dr4_counterfactual.py

Generates notebooks/lhrm_dr4_frozen_sa_cf_kaggle.ipynb and
bundles/lhrm_gpu_bundle_dr4_counterfactual_20261012.zip —
Dedicated, reproducible Kaggle GPU notebook for DR4 Frozen-Checkpoint Label-SA Counterfactual:
1. Supports BOTH:
   - Exact original ZIP bundle with SHA-256 verification.
   - Kaggle auto-extracted directory with pinned BUNDLE_MANIFEST.json SHA-256 and verification of all files.
2. Verifies and loads authentic trained checkpoints:
   - L2_STEP300_MODEL_ONLY.pt
   - L2_A5B_STEP300_MODEL_ONLY.pt
3. Uses official BERT revision 86b5e0934494bd15c9632b12f734a8a67f723594.
4. Stages authentic AAPD Dev data (Test split strictly quarantined) and frozen 200 Dev IDs.
5. Verifies pre-intervention baseline against authentic saved predictions (P@1 == 63/200, <= 1e-5).
6. Executes exact counterfactual intervention:
   counterfactual_logits[b, i] = doc_term[b, i] + bias (subtraction of static sa_term).
7. Verifies mathematical invariant:
   Var_b(counterfactual_logits[:, i]) == Var_b(original_logits[:, i]) (tol <= 1e-6).
   V4 passing count is strictly identical.
8. Measures changes in P@1 accuracy, ranking diversity, dominant label fraction, and duplicate targets.
9. Exports complete diagnostic evidence package and SHA-256 checksums.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BUNDLE_NAME = "lhrm_gpu_bundle_dr4_counterfactual_20261012.zip"
OUT_NB = ROOT / "notebooks" / "lhrm_dr4_frozen_sa_cf_kaggle.ipynb"
DIAG_ID = "DR4_FROZEN_SA_CF_20261012_A"


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


def build_counterfactual_notebook(bundle_name: str, bundle_sha256: str, manifest_sha256: str) -> Dict[str, Any]:
    cells = []

    # ── Cell 0: Markdown Protocol & Scope ─────────────────────────────────────
    cells.append(cell(f"""\\
# LHRM DR4 — Frozen-Checkpoint Label-SA Counterfactual Diagnostic ({DIAG_ID})

**Inference-Only Counterfactual Diagnostic on Frozen Trained Step-300 Checkpoints**

This session tests whether subtracting the static, document-invariant label self-attention offset (`sa_term`) from the raw logits of the frozen trained models materially restores ranking diversity and Top-1 accuracy:
- `L2_STEP300_MODEL_ONLY.pt` (shared classifier $w \\in \\mathbb{{R}}^{{1 \\times H}}$)
- `L2_A5B_STEP300_MODEL_ONLY.pt` (unshared per-label classifier $W \\in \\mathbb{{R}}^{{k \\times H}}$)

### Counterfactual Definition:
- $\\text{{original\\_logits}}[b, i] = \\text{{doc\\_term}}[b, i] + \\text{{sa\\_term}}[i] + b$
- $\\text{{counterfactual\\_logits}}[b, i] = \\text{{doc\\_term}}[b, i] + b \\equiv \\text{{original\\_logits}}[b, i] - \\text{{sa\\_term}}[i]$

### Non-Negotiable Invariants:
1. **Zero Parameter Updates:** Checkpoint weights and embeddings are unmodified. No optimizer is instantiated.
2. **Subtraction Parity:** $\\max |\\text{{counterfactual\\_logits}} - (\\text{{doc\\_term}} + b)| \\le 10^{{-5}}$.
3. **Cross-Document Variance Invariance:** Because $\\text{{sa\\_term}}[i]$ is document-invariant, $\\operatorname{{Var}}_b(\\text{{counterfactual\\_logits}}[:, i]) \\equiv \\operatorname{{Var}}_b(\\text{{original\\_logits}}[:, i])$. The V4 passing label count must remain strictly unchanged.
4. **Pre-Intervention Baseline Verification:** Original model outputs must reproduce historical screening scores within $\\le 10^{{-5}}$ and exactly 63/200 Top-1 hits.
5. **Strict Scope:** Test split is strictly quarantined (never read or evaluated). Evaluates only the frozen 200 Dev documents.

**Bundle Name:** `{bundle_name}`  
**Bundle SHA-256:** `{bundle_sha256}`  
**Manifest SHA-256:** `{manifest_sha256}`
""", cell_type="markdown"))

    # ── Cell 1: Environment Check & Official BERT Setup ───────────────────────
    cells.append(cell("""\
# Cell 1: Environment Check & Official BERT Acquisition
import os, sys, time, json, hashlib, shutil, gc
from pathlib import Path
import torch

print("=== DR4 Counterfactual Diagnostic: Environment & Device Gate ===", flush=True)
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
EXPECTED_MANIFEST_SHA = "@MANIFEST_SHA256@"

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

# 1. Verify and Stage Bundle (Supporting BOTH exact ZIP and auto-extracted dataset)
bundle_path = None
for root, _, files in os.walk(INP):
    if BUNDLE_NAME in files:
        p = Path(root) / BUNDLE_NAME
        if p.is_file():
            bundle_path = p
            break

manifest_path = None
for root, _, files in os.walk(INP):
    if "BUNDLE_MANIFEST.json" in files:
        p = Path(root) / "BUNDLE_MANIFEST.json"
        if p.is_file():
            manifest_path = p
            break

staged = False
if bundle_path is not None:
    actual_bundle_sha = sha_file(bundle_path)
    if actual_bundle_sha == EXPECTED_BUNDLE_SHA:
        print(f"Bundle ZIP VERIFIED: {bundle_path} (SHA: {actual_bundle_sha})", flush=True)
        with zipfile.ZipFile(bundle_path, "r") as z:
            z.extractall(WORK)
        staged = True
    else:
        print(f"Notice: Found {bundle_path} but SHA mismatch: {actual_bundle_sha} != {EXPECTED_BUNDLE_SHA}", flush=True)

if not staged and manifest_path is not None:
    actual_manifest_sha = sha_file(manifest_path)
    assert actual_manifest_sha == EXPECTED_MANIFEST_SHA, f"FAIL-CLOSED: BUNDLE_MANIFEST.json SHA mismatch: {actual_manifest_sha} != {EXPECTED_MANIFEST_SHA}"
    print(f"Bundle manifest VERIFIED: {manifest_path} (SHA: {actual_manifest_sha})", flush=True)
    manifest_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for rel_path, exp_file_sha in manifest["files"].items():
        src_file = manifest_dir / rel_path
        if not src_file.is_file():
            raise FileNotFoundError(f"FAIL-CLOSED: Extracted bundle constituent file missing: {src_file}")
        actual_file_sha = sha_file(src_file)
        assert actual_file_sha == exp_file_sha, f"FAIL-CLOSED: File SHA mismatch for {rel_path}: {actual_file_sha} != {exp_file_sha}"
        dest_file = WORK / rel_path
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dest_file)
    print(f"All {len(manifest['files'])} constituent files verified and staged from extracted dataset.", flush=True)
    staged = True

if not staged:
    raise FileNotFoundError(
        f"FAIL-CLOSED: Neither exact bundle ZIP '{BUNDLE_NAME}' (SHA: {EXPECTED_BUNDLE_SHA}) "
        f"nor verified extracted directory with BUNDLE_MANIFEST.json (SHA: {EXPECTED_MANIFEST_SHA}) found under /kaggle/input."
    )

# Verify reference predictions exist in working directory
ref_l2_path = WORK / "reference_predictions" / "variant_L2" / "val_sample_predictions.json"
ref_a5b_path = WORK / "reference_predictions" / "variant_L2_A5B" / "val_sample_predictions.json"
assert ref_l2_path.is_file(), f"FAIL-CLOSED: Staged L2 reference predictions not found at {ref_l2_path}"
assert ref_a5b_path.is_file(), f"FAIL-CLOSED: Staged L2_A5B reference predictions not found at {ref_a5b_path}"
print("Reference predictions verified in working directory.", flush=True)

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

print(f"Checkpoints VERIFIED: {list(found_ckpts.keys())}", flush=True)

# 3. Locate and Stage AAPD Dev CSV Only (Test strictly quarantined)
dev_source = None
for root, _, files in os.walk(INP):
    if "dev.csv" in files:
        p = Path(root) / "dev.csv"
        if sha_file(p) == EXPECTED_DEV_CSV_SHA:
            dev_source = p
            break

if dev_source is None:
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
""".replace("@BUNDLE_NAME@", bundle_name).replace("@BUNDLE_SHA256@", bundle_sha256).replace("@MANIFEST_SHA256@", manifest_sha256)
    cells.append(cell(cell2_code))

    # ── Cell 3: Execute Counterfactual Diagnostic ─────────────────────────────
    cell3_code = """\
# Cell 3: Execute Frozen-Checkpoint Label-SA Counterfactual
import os, sys, json, shutil, gc
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer

from lhrm_baseline.data import MultiLabelDataset, collate_fn, tokenize_labels
from lhrm_baseline.labels import build_all_label_texts
from lhrm_baseline.model import LHRMModel, get_cls_embedding

sys.path.insert(0, str(WORK / "scripts"))
from probe_dr4_trained_logit_decomposition import (
    decompose_l2_csam_logits,
    decompose_a5b_csam_logits,
    verify_saved_predictions,
)
from probe_dr4_frozen_sa_counterfactual import (
    compute_counterfactual_logits,
    verify_cross_document_variance_invariance,
    compare_original_vs_counterfactual,
    export_counterfactual_deliverables,
)

print("=== Executing DR4 Frozen-Checkpoint Label-SA Counterfactual ===", flush=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained(str(BERT_DIR), local_files_only=True)
defs_path = WORK / "Label_Definition" / "arxiv_labels_definitions.json"
label_texts_list = build_all_label_texts(cols, str(defs_path), dataset="aapd")
label_ids, label_mask = tokenize_labels(label_texts_list, tokenizer, max_len=64, device=device)

eval_ds = MultiLabelDataset(eval_texts, eval_labels, tokenizer, max_len=512)
eval_loader = DataLoader(eval_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)

true_labels_list = []
for row in eval_labels:
    true_labels_list.append([cols[j] for j in range(len(cols)) if row[j] > 0.5])

# 1. Run Counterfactual for L2
print("\\n--- Counterfactual Evaluation: Variant L2 ---", flush=True)
ckpt_l2 = torch.load(found_ckpts["L2_STEP300_MODEL_ONLY.pt"], map_location=device)
assert ckpt_l2["step"] == 300 and ckpt_l2["variant"] == "L2"

bert_l2 = BertModel.from_pretrained(str(BERT_DIR), local_files_only=True).to(device)
model_l2 = LHRMModel(bert_l2, num_labels=54, variant="L2").to(device)
model_l2.load_state_dict(ckpt_l2["model_state"], strict=True)
model_l2.eval()

with torch.no_grad():
    label_out_l2 = model_l2.bert(label_ids, attention_mask=label_mask)
    h_L_l2 = get_cls_embedding(label_out_l2)

doc_h_T_l2 = []
fwd_logits_l2_list = []
with torch.no_grad():
    for batch in eval_loader:
        b_ids = batch["input_ids"].to(device)
        b_mask = batch["attention_mask"].to(device)
        out = model_l2.bert(b_ids, attention_mask=b_mask)
        doc_h_T_l2.append(get_cls_embedding(out))
        fwd_logits, _ = model_l2(b_ids, b_mask, label_ids, label_mask)
        fwd_logits_l2_list.append(fwd_logits)

h_T_l2 = torch.cat(doc_h_T_l2, dim=0)
fwd_logits_l2 = torch.cat(fwd_logits_l2_list, dim=0)

w_l2 = model_l2.csam.W.weight.data
b_l2 = model_l2.csam.W.bias.data

decomp_l2 = decompose_l2_csam_logits(h_T_l2, h_L_l2, w_l2, b_l2)
assert (decomp_l2["logits"] - fwd_logits_l2).abs().max().item() <= 1e-5
pred_check_l2 = verify_saved_predictions(decomp_l2["logits"], cols, ref_l2_path, true_labels_list, tol=1e-5)
assert pred_check_l2["p_at_1_hits"] == 63 and pred_check_l2["v4_passing_count"] == 21

l2_comp = compare_original_vs_counterfactual(decomp_l2, cols, true_labels_list)
print(f"L2 Counterfactual: Orig P@1 = {l2_comp['original_ranking']['p_at_1_hits']}/200, CF P@1 = {l2_comp['counterfactual_ranking']['p_at_1_hits']}/200 (Delta: {l2_comp['delta_hits']:+d})")
print(f"L2 Diversity: Orig Distinct Top1 = {l2_comp['original_ranking']['distinct_top1_count']}, CF Distinct Top1 = {l2_comp['counterfactual_ranking']['distinct_top1_count']}")
print(f"L2 Invariant: Max std diff = {l2_comp['invariant_verification']['max_std_difference']:.2e}, V4 passing count = {l2_comp['invariant_verification']['cf_v4_passing_count']}/54 (unchanged)")

del model_l2, bert_l2, h_T_l2, h_L_l2
torch.cuda.empty_cache()
gc.collect()

# 2. Run Counterfactual for L2_A5B
print("\\n--- Counterfactual Evaluation: Variant L2_A5B ---", flush=True)
ckpt_a5b = torch.load(found_ckpts["L2_A5B_STEP300_MODEL_ONLY.pt"], map_location=device)
assert ckpt_a5b["step"] == 300 and ckpt_a5b["variant"] == "L2_A5B"

bert_a5b = BertModel.from_pretrained(str(BERT_DIR), local_files_only=True).to(device)
model_a5b = LHRMModel(bert_a5b, num_labels=54, variant="L2_A5B").to(device)
model_a5b.load_state_dict(ckpt_a5b["model_state"], strict=True)
model_a5b.eval()

with torch.no_grad():
    label_out_a5b = model_a5b.bert(label_ids, attention_mask=label_mask)
    h_L_a5b = get_cls_embedding(label_out_a5b)

doc_h_T_a5b = []
fwd_logits_a5b_list = []
with torch.no_grad():
    for batch in eval_loader:
        b_ids = batch["input_ids"].to(device)
        b_mask = batch["attention_mask"].to(device)
        out = model_a5b.bert(b_ids, attention_mask=b_mask)
        doc_h_T_a5b.append(get_cls_embedding(out))
        fwd_logits, _ = model_a5b(b_ids, b_mask, label_ids, label_mask)
        fwd_logits_a5b_list.append(fwd_logits)

h_T_a5b = torch.cat(doc_h_T_a5b, dim=0)
fwd_logits_a5b = torch.cat(fwd_logits_a5b_list, dim=0)

W_a5b = model_a5b.csam.W.data
b_a5b = model_a5b.csam.bias.data

decomp_a5b = decompose_a5b_csam_logits(h_T_a5b, h_L_a5b, W_a5b, b_a5b)
assert (decomp_a5b["logits"] - fwd_logits_a5b).abs().max().item() <= 1e-5
pred_check_a5b = verify_saved_predictions(decomp_a5b["logits"], cols, ref_a5b_path, true_labels_list, tol=1e-5)
assert pred_check_a5b["p_at_1_hits"] == 63 and pred_check_a5b["v4_passing_count"] == 3

a5b_comp = compare_original_vs_counterfactual(decomp_a5b, cols, true_labels_list)
print(f"L2_A5B Counterfactual: Orig P@1 = {a5b_comp['original_ranking']['p_at_1_hits']}/200, CF P@1 = {a5b_comp['counterfactual_ranking']['p_at_1_hits']}/200 (Delta: {a5b_comp['delta_hits']:+d})")
print(f"L2_A5B Diversity: Orig Distinct Top1 = {a5b_comp['original_ranking']['distinct_top1_count']}, CF Distinct Top1 = {a5b_comp['counterfactual_ranking']['distinct_top1_count']}")
print(f"L2_A5B Invariant: Max std diff = {a5b_comp['invariant_verification']['max_std_difference']:.2e}, V4 passing count = {a5b_comp['invariant_verification']['cf_v4_passing_count']}/54 (unchanged)")

del model_a5b, bert_a5b, h_T_a5b, h_L_a5b
torch.cuda.empty_cache()
gc.collect()

# 3. Export Deliverables
out_dir = WORK / "diagnostics" / "DR4_FROZEN_SA_CF"
pred_meta = {
    "L2_pre_check": pred_check_l2,
    "L2_A5B_pre_check": pred_check_a5b,
    "max_diff": max(pred_check_l2["max_score_diff"], pred_check_a5b["max_score_diff"]),
}
paths = export_counterfactual_deliverables(l2_comp, a5b_comp, out_dir, pred_meta)
print("\\n=== Deliverables Exported ===", flush=True)
for k_name, p in paths.items():
    print(f"  {k_name}: {p}")
"""
    cells.append(cell(cell3_code))

    # ── Cell 4: Interactive File Links ────────────────────────────────────────
    cells.append(cell("""\
# Cell 4: Download links for exported deliverables
from IPython.display import FileLink, display
from pathlib import Path

out_dir = Path("/kaggle/working/diagnostics/DR4_FROZEN_SA_CF")
if out_dir.is_dir():
    print("Downloadable Counterfactual Evidence Deliverables:")
    for f in sorted(out_dir.iterdir()):
        display(FileLink(str(f)))
"""))

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "cells": cells,
    }
    return nb


def build_counterfactual_bundle(out_zip: Path) -> Tuple[str, str]:
    """
    Builds lightweight bundle containing code, configs, reference predictions,
    and counterfactual diagnostics. Returns (bundle_sha256, manifest_sha256).
    """
    to_include = [
        (ROOT / "run_baseline.py", "run_baseline.py"),
        (ROOT / "scripts" / "probe_dr4_trained_logit_decomposition.py", "scripts/probe_dr4_trained_logit_decomposition.py"),
        (ROOT / "scripts" / "probe_dr4_frozen_sa_counterfactual.py", "scripts/probe_dr4_frozen_sa_counterfactual.py"),
        (ROOT / "Label_Definition" / "arxiv_labels_definitions.json", "Label_Definition/arxiv_labels_definitions.json"),
        (ROOT / "experiments" / "RUN_0" / "run0_val_sample_ids.json", "experiments/RUN_0/run0_val_sample_ids.json"),
        (ROOT / "docs" / "experiments" / "DR4_FROZEN_SA_COUNTERFACTUAL_PROTOCOL.md", "docs/experiments/DR4_FROZEN_SA_COUNTERFACTUAL_PROTOCOL.md"),
        (ROOT / "reference_predictions" / "variant_L2" / "val_sample_predictions.json", "reference_predictions/variant_L2/val_sample_predictions.json"),
        (ROOT / "reference_predictions" / "variant_L2_A5B" / "val_sample_predictions.json", "reference_predictions/variant_L2_A5B/val_sample_predictions.json"),
    ]

    src_dir = ROOT / "src"
    for base, _, files in os.walk(src_dir):
        for f in sorted(files):
            if f.endswith(".py"):
                fp = Path(base) / f
                rel = fp.relative_to(ROOT).as_posix()
                to_include.append((fp, rel))

    manifest = {
        "bundle_name": out_zip.name,
        "diagnostic_id": DIAG_ID,
        "date": "2026-10-12T00:00:00Z",
        "files": {},
    }
    for fp, rel in to_include:
        if fp.is_file():
            manifest["files"][rel] = sha256_file(fp)

    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp, rel in to_include:
            if fp.is_file():
                zf.write(fp, arcname=rel)
        zf.writestr("BUNDLE_MANIFEST.json", manifest_bytes)

    bundle_sha256 = sha256_file(out_zip)
    return bundle_sha256, manifest_sha256


def main():
    parser = argparse.ArgumentParser(description="Generate Kaggle notebook and bundle for DR4 counterfactual")
    parser.add_argument("--bundle_name", default=BUNDLE_NAME)
    args = parser.parse_args()

    bundle_path = ROOT / "bundles" / args.bundle_name
    print(f"Building counterfactual bundle: {bundle_path}...")
    bundle_sha, manifest_sha = build_counterfactual_bundle(bundle_path)
    bundle_size_kb = bundle_path.stat().st_size / 1024.0
    print(f"Bundle built: {bundle_path}")
    print(f"  Bundle SHA-256:   {bundle_sha}")
    print(f"  Manifest SHA-256: {manifest_sha}")
    print(f"  Size:             {bundle_size_kb:.1f} KB")

    nb = build_counterfactual_notebook(args.bundle_name, bundle_sha, manifest_sha)
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
