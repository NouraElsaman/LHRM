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
import subprocess
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
   - Pre-training parity gate: Real training-path initialization copies `L2` shared $W$ into all 54 rows and verifies pre-step logit parity before step 1.
5. **Data Quarantine:** **Test split is strictly quarantined.** Only AAPD Train (53,840) and Dev (1,000) splits are loaded.
6. **Fixed Dev Sample:** Locked 200 validation documents from `experiments/RUN_0/run0_val_sample_ids.json` (frequency prior: exactly 63/200 hits).
7. **Scientific Interpretation Limits:**
   - V4 passes when $\\ge 28/54$ labels exhibit cross-document logit standard deviation $> 0.01$.
   - V5 passes when P@1 integer hits strictly exceed 63 ($> 31.50\\%$).
   - Ranking diversity alone is NOT proof of correct learning.
   - Passing V4 and V5 justifies further evaluation, not successful reproduction.
   - Failing V4 or V5 shows that unsharing the classifier alone is insufficient; it does NOT prove dot-product attention is the sole remaining failure source.

**Bundle Name:** `{bundle_name}`  
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
    print("Acquiring official pretrained BERT from", REPO, "@", PINNED_COMMIT, "(Internet ON required)...", flush=True)
    from huggingface_hub import hf_hub_download
    from transformers import BertModel, BertTokenizer
    
    revision = PINNED_COMMIT
    print("Pinned official BERT commit:", revision, flush=True)
    
    tokenizer = BertTokenizer.from_pretrained(REPO, revision=revision)
    model = BertModel.from_pretrained(REPO, revision=revision)
    assert tokenizer.vocab_size == 30522 and model.config.hidden_size == 768
    
    tmp = BERT_DIR.with_name(BERT_DIR.name + "_DOWNLOADING")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        model.save_pretrained(str(tmp), safe_serialization=True)
        tokenizer.save_pretrained(str(tmp))
        official_vocab = Path(hf_hub_download(repo_id=REPO, filename="vocab.txt", revision=revision))
        shutil.copyfile(official_vocab, tmp / "vocab.txt")
        assert (tmp / "config.json").is_file() and (tmp / "vocab.txt").is_file()
        weight_file = next((tmp / w for w in WEIGHTS if (tmp / w).is_file()), None)
        assert weight_file is not None, "Missing BERT weights after saving"
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

meta = json.loads(SOURCE_META.read_text("utf-8"))
assert meta.get("repo_id") == REPO and meta.get("source_revision") == PINNED_COMMIT, (
    f"Official BERT source revision mismatch: expected {PINNED_COMMIT}, got {meta.get('source_revision')}"
)
assert sha256_bytes(BERT_DIR / meta["saved_weights_file"]) == meta["saved_weights_sha256"]
assert sha256_bytes(BERT_DIR / "config.json") == meta["bert_config_sha256"]
assert sha256_bytes(BERT_DIR / "vocab.txt") == meta["vocab_sha256"]

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
print("OFFICIAL BERT READY & VERIFIED:", BERT_DIR, flush=True)
"""))

    # ── Cell 2: Verify & Stage Bundle and AAPD Train/Dev Data ────────────────
    cell2_code = """\
# Cell 2: Verify & Stage Bundle and AAPD Train/Dev Data
from pathlib import Path
import os, json, hashlib, zipfile, shutil, sys

WORK = Path("/kaggle/working")
INP = Path("/kaggle/input")
BUNDLE_NAME = "@BUNDLE_NAME@"
EXPECTED_BUNDLE_SHA = "@BUNDLE_SHA256@"

EXPECTED_CSV_SHA = {
    "train": "e7e73064385f78fe092edd47d4e4059c35875a1ea2a95859ad3cde091ed9af95",
    "dev": "a85f612d4c1cb2be6d1de94cbcd9468cba97d221f826014264f4d279c82e220b",
}
EXPECTED_LABEL_SHA = "a3d662f3920259a94ed61d83f98f21cabe7f908d98a5662ccd9adc2413cc000b"

def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def sha_stream(f):
    h = hashlib.sha256()
    for chunk in iter(lambda: f.read(1024*1024), b""):
        h.update(chunk)
    return h.hexdigest()

# ── 1. Locate and Verify GPU Bundle (Fail-Closed) ───────────────────────────
candidate_paths = [
    INP / BUNDLE_NAME,
    INP / "lhrm-l2-a5b-bundle" / BUNDLE_NAME,
    INP / "lhrm-gpu-bundle-l2-a5b-20261012" / BUNDLE_NAME,
    WORK / "bundles" / BUNDLE_NAME,
    WORK / BUNDLE_NAME,
]
bundle_path = None
for p in candidate_paths:
    if p.is_file() and p.name == BUNDLE_NAME:
        bundle_path = p
        break

if bundle_path is None:
    for root, _, files in os.walk(INP):
        if BUNDLE_NAME in files:
            p = Path(root) / BUNDLE_NAME
            if p.is_file():
                bundle_path = p
                break

if bundle_path is None:
    raise FileNotFoundError(
        f"FAIL-CLOSED: Exact bundle file '{BUNDLE_NAME}' not found in /kaggle/input or /kaggle/working.\\n"
        "Arbitrary zip files are strictly rejected."
    )

print(f"Located GPU Bundle: {bundle_path}")
actual_bundle_sha = sha_file(bundle_path)
print(f"Bundle SHA-256: {actual_bundle_sha}")
if actual_bundle_sha != EXPECTED_BUNDLE_SHA:
    raise RuntimeError(
        f"FAIL-CLOSED BUNDLE VERIFICATION FAILED!\\n"
        f"Expected SHA-256: {EXPECTED_BUNDLE_SHA}\\n"
        f"Actual SHA-256:   {actual_bundle_sha}\\n"
        "Bundle content does not match the preregistered cryptographic digest."
    )
print("Bundle SHA-256 VERIFIED. Proceeding to extract...", flush=True)

with zipfile.ZipFile(bundle_path, "r") as z:
    z.extractall(WORK)
print("Bundle extracted to /kaggle/working.")

manifest_path = WORK / "BUNDLE_MANIFEST.json"
assert manifest_path.is_file(), "FAIL-CLOSED: BUNDLE_MANIFEST.json missing from extracted bundle"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
for rel_path, expected_hash in manifest.get("files", {}).items():
    file_p = WORK / rel_path
    assert file_p.is_file(), f"FAIL-CLOSED: Bundle file missing after extraction: {rel_path}"
    file_hash = sha_file(file_p)
    assert file_hash == expected_hash, (
        f"FAIL-CLOSED: Bundle internal file integrity failure for {rel_path}!\\n"
        f"Expected: {expected_hash}\\n"
        f"Actual:   {file_hash}"
    )
print(f"Bundle integrity PASS: {len(manifest['files'])} files verified against manifest.", flush=True)

# ── 2. Locate and Verify AAPD Train/Dev Data (Quarantine Test) ───────────────
csv_dirs, data_zips = [], []
for base, ds, fs in os.walk(INP):
    root = Path(base)
    if "train.csv" in fs and "dev.csv" in fs:
        csv_dirs.append(root)
    for name in fs:
        if name.lower().endswith(".zip") and "aapd" in str(root / name).lower():
            data_zips.append(root / name)

valid = []
for zp in sorted(data_zips):
    try:
        with zipfile.ZipFile(zp) as z:
            members = {}
            for split in ("train", "dev"):
                choices = [x for x in z.namelist() if x.replace("\\\\", "/").endswith("/" + split + ".csv") or x == split + ".csv"]
                if len(choices) == 1:
                    members[split] = choices[0]
            if len(members) == 2:
                hashes = {s: sha_stream(z.open(members[s])) for s in ("train", "dev")}
                if hashes == EXPECTED_CSV_SHA:
                    valid.append(("zip", zp, members, sha_file(zp)))
    except Exception:
        pass

for dr in sorted(csv_dirs):
    hashes = {s: sha_file(dr / (s + ".csv")) for s in ("train", "dev")}
    if hashes == EXPECTED_CSV_SHA:
        valid.append(("directory", dr, None, None))

if not valid:
    raise RuntimeError("FAIL-CLOSED: Authentic AAPD train/dev source CSVs not found in /kaggle/input.")

kind, data_path, members, outer_hash = valid[0]
print("AAPD source VERIFIED:", kind, data_path, flush=True)

# Stage Train & Dev Only (Test strictly excluded)
STAGED = WORK / "AAPD_A5B_TRAIN_DEV_ONLY.zip"
with zipfile.ZipFile(STAGED, "w", compression=zipfile.ZIP_STORED) as out:
    for split in ("train", "dev"):
        with out.open("AAPD/" + split + ".csv", "w") as dest:
            if kind == "zip":
                with zipfile.ZipFile(data_path) as inz, inz.open(members[split]) as src:
                    shutil.copyfileobj(src, dest, 1024*1024)
            else:
                with open(data_path / (split + ".csv"), "rb") as src:
                    shutil.copyfileobj(src, dest, 1024*1024)

# ── 3. Produce Processed Data (train_texts.json, train_labels.npy, etc.) ────
if str(WORK / "src") not in sys.path:
    sys.path.insert(0, str(WORK / "src"))
from lhrm_baseline.data import load_aapd_split, save_processed
from lhrm_baseline.labels import build_all_label_texts
from lhrm_baseline.diagnostics import frequency_prior_hits_at_1

processed = WORK / "data" / "processed" / "AAPD"
processed.mkdir(parents=True, exist_ok=True)
label_order = None
train_labels_cached = None
dev_labels_cached = None

for split, sz in [("train", 53840), ("dev", 1000)]:
    texts, labels, cols = load_aapd_split(str(STAGED), split)
    assert len(texts) == sz and labels.shape == (sz, 54), (
        f"Split {split} shape mismatch: len={len(texts)}, shape={labels.shape}"
    )
    if label_order is None:
        label_order = cols
    else:
        assert cols == label_order, "Label column order inconsistent between train and dev!"
    if split == "train":
        train_labels_cached = labels
    else:
        dev_labels_cached = labels
    save_processed(str(processed), split, texts, labels, cols)
    print(f"Prepared {split}: {sz} docs, 54 labels (Test strictly quarantined)", flush=True)

# Verify label order hash
label_sha = hashlib.sha256(json.dumps(label_order).encode()).hexdigest()
assert label_sha == EXPECTED_LABEL_SHA, f"Label order SHA mismatch: {label_sha} != {EXPECTED_LABEL_SHA}"

# Build label definitions text
defs_path = WORK / "Label_Definition" / "arxiv_labels_definitions.json"
assert defs_path.is_file(), f"Label definition file missing at {defs_path}"
label_texts = build_all_label_texts(label_order, str(defs_path), dataset="aapd")
assert len(label_texts) == 54
(processed / "label_texts.json").write_text(json.dumps(label_texts, indent=2, ensure_ascii=False), encoding="utf-8")
assert not list(processed.glob("test_*")), "QUARANTINE VIOLATION: Test files detected in processed data!"

# Verify frozen 200 Dev IDs
val_ids_file = WORK / "experiments" / "RUN_0" / "run0_val_sample_ids.json"
assert val_ids_file.is_file(), "Missing frozen validation sample IDs"
frozen_ids = json.loads(val_ids_file.read_text())
assert len(frozen_ids) == 200 and len(set(frozen_ids)) == 200

# Verify frequency prior hits == 63/200
v_labels_sub = dev_labels_cached[frozen_ids]
prior_hits = frequency_prior_hits_at_1(train_labels_cached, v_labels_sub, seed=42)
print(f"Frequency prior hits on frozen 200 Dev docs: {prior_hits}/200", flush=True)
assert prior_hits == 63, f"Prior hit mismatch: expected 63, got {prior_hits}"

# Prepare runtime config pointing to official local BERT
orig_cfg = WORK / "configs" / "final" / "aapd_baseline.yaml"
cfg_text = orig_cfg.read_text(encoding="utf-8")
runtime_cfg = WORK / "configs" / "final" / "aapd_a5b_runtime.yaml"
runtime_cfg.write_text(
    cfg_text.replace("bert_model: bert-base-uncased", f"bert_model: {WORK / 'google_bert_bert_base_uncased_official'}"),
    encoding="utf-8"
)

print("\\n=== DATA & CONFIG GATES PASSED: Train+Dev Staged, Test Quarantined, Prior Hits = 63 ===", flush=True)
""".replace("@BUNDLE_NAME@", bundle_name).replace("@BUNDLE_SHA256@", bundle_sha256)
    cells.append(cell(cell2_code))

    # ── Cell 3: Standalone Pre-Flight Step-0 Parity Check ────────────────────
    cells.append(cell("""\
# Cell 3: Standalone Step-0 Parity Verification (L2 vs L2_A5B)
import torch, gc
from transformers import BertModel
from lhrm_baseline.model import LHRMModel, init_a5b_from_l2

print("=== Standalone Pre-Flight: Step-0 Baseline Logit & Attention Parity ===", flush=True)

device = torch.device("cuda")
bert = BertModel.from_pretrained(str(BERT_DIR)).to(device)

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
import subprocess, time
from pathlib import Path

WORK = Path("/kaggle/working")
RUN_ID = "@RUN_ID@"
CANDIDATES = "@CANDIDATES@"
N_STEPS = @N_STEPS@
EXP_DIR = WORK / "experiments" / RUN_ID

# Fail-closed guard: do not overwrite an existing run
if EXP_DIR.exists() and any(EXP_DIR.iterdir()):
    raise RuntimeError(
        f"FAIL-CLOSED GUARD: Target experiment directory {EXP_DIR} already exists and is non-empty!\\n"
        "Refusing to overwrite historical or existing run."
    )

runtime_cfg = WORK / "configs" / "final" / "aapd_a5b_runtime.yaml"
assert runtime_cfg.is_file(), f"Missing runtime config: {runtime_cfg}"

cmd = [
    sys.executable, "-u",
    str(WORK / "run_baseline.py"),
    "run0",
    "--config", str(runtime_cfg),
    "--run_id", RUN_ID,
    "--steps", str(N_STEPS),
    "--candidates", CANDIDATES,
]

print("=== STARTING SCREENING CLI ===", flush=True)
print("Command:", " ".join(cmd), flush=True)

env = os.environ.copy()
env.update({
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "PYTHONUNBUFFERED": "1",
})

start_time = time.monotonic()
proc = subprocess.Popen(cmd, cwd=str(WORK), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in proc.stdout:
    print(line, end="", flush=True)
ret_code = proc.wait()
duration = round(time.monotonic() - start_time, 1)

print(f"\\nScreening CLI finished in {duration}s with exit code {ret_code}", flush=True)
if ret_code != 0:
    raise RuntimeError(f"run_baseline.py run0 failed with exit code {ret_code}")

print("Screening CLI completed successfully.", flush=True)
""".replace("@RUN_ID@", RUN_ID).replace("@CANDIDATES@", CANDIDATES).replace("@N_STEPS@", str(N_STEPS))
    cells.append(cell(cell4_code))

    # ── Cell 5: Diagnostic Audit & Viability Evaluation ──────────────────────
    cell5_code = """\
# Cell 5: Diagnostic Audit & Viability Evaluation (Authentic Prediction Schema)
from pathlib import Path
import json, math
from collections import Counter
import numpy as np

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

# Load canonical label order
canonical_label_file = WORK / "data" / "processed" / "AAPD" / "labels.json"
canonical_labels = json.loads(canonical_label_file.read_text(encoding="utf-8"))
assert len(canonical_labels) == 54

print("\\n" + "="*70)
print(f"DR4 SCREENING RESULTS SUMMARY: {RUN_ID}")
print("="*70)

# Seeded rank order matching lhrm_baseline.metrics
def seeded_rank_order(scores, seed_val):
    rng = np.random.default_rng(seed_val)
    jitter = rng.random(len(scores)) * 1e-9
    return np.argsort(-(scores + jitter), kind="stable")

results = {}
for cand in ["L2", "L2_A5B"]:
    s300 = json.loads((EXP_DIR / f"variant_{cand}" / "step300_diagnostics.json").read_text())
    preds = json.loads((EXP_DIR / f"variant_{cand}" / "val_sample_predictions.json").read_text())
    
    # 1. Validate authentic schema
    assert len(preds) == 200, f"Expected 200 records, got {len(preds)}"
    for idx, p in enumerate(preds):
        assert "scores" in p and "labels" in p and "true_labels" in p, f"Schema missing keys in record {idx}"
        assert len(p["scores"]) == 54, f"Record {idx} scores length {len(p['scores'])} != 54"
        assert all(math.isfinite(s) for s in p["scores"]), f"Record {idx} contains non-finite scores"
        assert p["labels"] == canonical_labels, f"Record {idx} labels order does not match canonical label order"

    # 2. Numerical ranking directly from scores
    top1_labels = []
    top3_patterns = []
    independent_hits = 0

    for i, p in enumerate(preds):
        s_arr = np.array(p["scores"], dtype=np.float64)
        ranking = seeded_rank_order(s_arr, seed_val=42 ^ i)
        
        top1_idx = int(ranking[0])
        top1_lbl = p["labels"][top1_idx]
        top1_labels.append(top1_lbl)
        
        top3_lbls = tuple(p["labels"][int(j)] for j in ranking[:3])
        top3_patterns.append(top3_lbls)
        
        if top1_lbl in p["true_labels"]:
            independent_hits += 1

    top1_counts = Counter(top1_labels)
    distinct_top1 = len(top1_counts)
    dominant_label, dominant_cnt = top1_counts.most_common(1)[0]
    distinct_top3 = len(set(top3_patterns))
    
    hits = int(s300.get("p_at_1_hits", 0))
    assert independent_hits == hits, (
        f"Independent P@1 hits ({independent_hits}) != step300_diagnostics hits ({hits})"
    )
    p_at_1 = float(s300.get("p_at_1", 0.0))
    prior_hits = int(s300.get("prior_p_at_1_hits", 63))
    
    stds = s300.get("per_label_logit_std", [])
    passing_stds = sum(1 for s in stds if float(s) > 0.01)
    
    results[cand] = {
        "p_at_1_hits": hits,
        "independent_hits": independent_hits,
        "prior_hits": prior_hits,
        "distinct_top1": distinct_top1,
        "dominant_label": dominant_label,
        "dominant_cnt": dominant_cnt,
        "dominant_frac": dominant_cnt / len(preds),
        "distinct_top3_patterns": distinct_top3,
        "passing_v4_stds": passing_stds,
        "alpha_entropy": float(s300.get("alpha_entropy_mean", float("nan"))),
        "max_alpha": float(s300.get("mean_max_alpha", float("nan"))),
    }
    
    print(f"\\n--- Variant {cand} ---")
    print(f"  V4 (Logit Variation):     {passing_stds} / 54 labels std > 0.01 (Pass: {passing_stds >= 28})")
    print(f"  V5 (Ranking Superiority): {hits} / 200 hits vs Prior {prior_hits} (Pass: {hits > prior_hits})")
    print(f"  Independent Hit Check:    {independent_hits} / 200 (Matches step300: OK)")
    print(f"  Distinct Top-1 Labels:    {distinct_top1} / 54")
    print(f"  Dominant Top-1 Label:     {dominant_label} ({dominant_cnt}/200 = {dominant_cnt/len(preds):.1%})")
    print(f"  Distinct Top-3 Patterns:  {distinct_top3} / 200")
    print(f"  Attention Entropy:        {results[cand]['alpha_entropy']:.4f}")

print("\\n" + "="*70)
""".replace("@RUN_ID@", RUN_ID)
    cells.append(cell(cell5_code))

    # ── Cell 6: Package Evidence Archive & SHA256SUMS ────────────────────────
    cell6_code = """\
# Cell 6: Package Downloadable Evidence Archive (Excluding All Model Binaries)
import zipfile, hashlib, os
from pathlib import Path

WORK = Path("/kaggle/working")
RUN_ID = "@RUN_ID@"
EXP_DIR = WORK / "experiments" / RUN_ID
EVIDENCE_ZIP = WORK / f"DR4_A5B_EVIDENCE_{RUN_ID}.zip"

def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

# Collect lightweight evidence files (excluding all model checkpoints and binaries)
written_files = []
with zipfile.ZipFile(EVIDENCE_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(EXP_DIR):
        for f in sorted(files):
            fp = Path(root) / f
            rel = fp.relative_to(EXP_DIR)
            # Exclude ALL model checkpoints and binaries (.pt, .pth, .bin, .ckpt) from the lightweight evidence archive
            if f.endswith((".pt", ".pth", ".bin", ".ckpt")) or "checkpoints" in str(rel):
                continue
            z.write(fp, arcname=str(rel))
            written_files.append((str(rel), sha_file(fp)))

zip_sha = sha_file(EVIDENCE_ZIP)

# Write evidence manifest
manifest_data = {
    "run_id": RUN_ID,
    "evidence_archive": EVIDENCE_ZIP.name,
    "evidence_archive_sha256": zip_sha,
    "archive_size_bytes": EVIDENCE_ZIP.stat().st_size,
    "files": {rel: digest for rel, digest in written_files},
}
manifest_path = WORK / f"DR4_A5B_EVIDENCE_MANIFEST_{RUN_ID}.json"
manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

# Write SHA256SUMS
sha_lines = [
    f"{zip_sha}  {EVIDENCE_ZIP.name}",
    f"{sha_file(manifest_path)}  {manifest_path.name}",
]
sums_path = WORK / "DR4_A5B_SHA256SUMS.txt"
sums_path.write_text("\\n".join(sha_lines) + "\\n", encoding="utf-8")

print("\\n" + "="*70)
print("DR4 EVIDENCE ARCHIVE READY FOR DOWNLOAD")
print("="*70)
print(f"File:     {EVIDENCE_ZIP}")
print(f"SHA-256:  {zip_sha}")
print(f"Size:     {EVIDENCE_ZIP.stat().st_size / 1024.0:.1f} KB")
print(f"Files included: {len(written_files)}")
print(f"Manifest: {manifest_path}")
print(f"Checksums: {sums_path}")
print("\\nSTOP: Original model checkpoints step_150.pt and step_300.pt remain preserved separately in:")
print(f"  {EXP_DIR}/variant_L2/checkpoints/")
print(f"  {EXP_DIR}/variant_L2_A5B/checkpoints/")
print("Do not execute further training without audit review of downloaded evidence.")
""".replace("@RUN_ID@", RUN_ID)
    cells.append(cell(cell6_code))

    # Standard clean Jupyter notebook metadata without proprietary Kaggle fields
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
