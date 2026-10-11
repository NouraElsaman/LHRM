#!/usr/bin/env python3
"""
Generate notebooks/lhrm_run0_l2_no_sa_ablation_kaggle.ipynb —
Dedicated, reproducible Kaggle GPU notebook for the L2 vs L2_NO_SA CSAM ablation screening.

Protocol:
  Experiment ID: RUN_0_L2_NO_SA_ABLATION_20261011_A
  Steps: 300 per candidate
  Candidates: L2, L2_NO_SA
  Frozen seed: 42
  Validation sample: locked 200 dev IDs (frequency prior 63/200)
  Test data quarantine: strictly no test split read, copied, prepared, predicted, or evaluated.

Usage:
  python scripts/gen_kaggle_notebook_ablation.py [--build_bundle]
"""
import argparse
import hashlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

BUNDLE_NAME = "lhrm_gpu_bundle_l2_no_sa_ablation_20261011.zip"
OUT_NB = os.path.join(ROOT, "notebooks", "lhrm_run0_l2_no_sa_ablation_kaggle.ipynb")
RUN_ID = "RUN_0_L2_NO_SA_ABLATION_20261011_A"
CONFIG = "configs/final/aapd_baseline.yaml"
CANDIDATES = "L2,L2_NO_SA"
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
# LHRM Ablation Screening — L2 vs L2_NO_SA ({RUN_ID})

**Controlled CSAM Ablation Screening (300 Optimizer Steps per Variant)**

This experiment tests the hypothesis:
> Does removing the label-label self-attention (SA) contribution in CSAM allow the model to produce document-sensitive and correct label rankings, eliminating the static label ranking observed in RUN_0 and DR1?

### Research Protocol & Quarantine Constraints:
1. **Experiment ID:** `{RUN_ID}` (fail-closed if output exists).
2. **Candidates:** `L2` (full CSAM with label SA) vs `L2_NO_SA` (CSAM with label SA ablated: h_final = alpha_i * h_Li).
3. **Hyperparameters:** Seed 42, Focal loss (gamma=2.0, alpha=0.25), LR 2e-5, weight decay 0.01, AMP enabled, 300 optimizer steps.
4. **Data Isolation:** **Test split is strictly quarantined.** Never read, copied, prepared, predicted, or evaluated. Only AAPD Train (53,840) and Dev (1,000) splits are loaded.
5. **Fixed Dev Sample:** Locked 200 validation documents from `experiments/RUN_0/run0_val_sample_ids.json` (frequency prior: exactly 63/200).
6. **Scientific Interpretation:**
   - V3 passes when alpha_one_hot_frac < 0.9.
   - V5 compares integer Top-1 hit counts against the frequency prior (P@1 > 63).
   - Passing V5 in a 300-step screening is positive preliminary evidence, **not** proof of root cause.
   - Failing V5 does not prove text-label alignment is incapable of learning.
   - Ranking diversity alone is not proof of improved correctness.
   - Do **not** automatically extend to 1000 steps or run full training.
   - The historical `run0_decision` rule was designed for original architecture selection; per-variant diagnostic metrics and ranking diversity must be evaluated independently.

**Bundle SHA-256:** `{bundle_sha256}`
""", cell_type="markdown"))

    # ── Cell 1: Environment & Official BERT Setup ────────────────────────────
    cells.append(cell("""\
# Cell 1: Environment check & Official BERT Acquisition
import os, sys, time, json, hashlib, shutil, gc
from pathlib import Path
import torch

print("=== L2 vs L2_NO_SA Screening: Environment & Device Gate ===", flush=True)
if not torch.cuda.is_available():
    raise RuntimeError("GPU GATE: CUDA GPU required — select Kaggle GPU runtime (T4 or P100)")
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__, " | CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0), flush=True)

# Check dependencies
import sklearn, transformers, numpy as np
print(f"Transformers: {transformers.__version__} | Sklearn: {sklearn.__version__} | NumPy: {np.__version__}")

# Setup official BERT
REPO = "google-bert/bert-base-uncased"
PINNED_COMMIT = "86b5e0934494bd15c9632b12f734a8a67f723594"
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
    return (BERT_DIR / "config.json").is_file() and (BERT_DIR / "vocab.txt").is_file() and \\
        any((BERT_DIR / w).is_file() for w in WEIGHTS) and SOURCE_META.is_file()

if not complete_official_cache():
    # Check if bert-base-uncased is already attached as a Kaggle dataset
    attached_bert = None
    for cand in ["/kaggle/input/bert-base-uncased", "/kaggle/input/bertbaseuncased"]:
        if os.path.isdir(cand) and os.path.exists(os.path.join(cand, "config.json")):
            attached_bert = Path(cand)
            break
    
    if attached_bert:
        print(f"Found attached BERT dataset at {attached_bert}. Copying to working dir...", flush=True)
        if BERT_DIR.exists(): shutil.rmtree(BERT_DIR)
        shutil.copytree(str(attached_bert), str(BERT_DIR))
        meta = {
            "source": str(attached_bert),
            "config_sha256": sha256_bytes(BERT_DIR / "config.json"),
            "vocab_sha256": sha256_bytes(BERT_DIR / "vocab.txt") if (BERT_DIR / "vocab.txt").is_file() else "N/A"
        }
        SOURCE_META.write_text(json.dumps(meta, indent=2) + "\\n", encoding="utf-8")
    else:
        print(f"Downloading official pretrained BERT from {REPO} (commit {PINNED_COMMIT})...", flush=True)
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            os.environ.pop(key, None)
        from huggingface_hub import HfApi, hf_hub_download
        from transformers import BertModel, BertTokenizer
        
        tokenizer = BertTokenizer.from_pretrained(REPO, revision=PINNED_COMMIT)
        assert tokenizer.vocab_size == 30522, f"Unexpected BERT vocab size: {tokenizer.vocab_size}"
        model = BertModel.from_pretrained(REPO, revision=PINNED_COMMIT)
        assert model.config.hidden_size == 768
        
        tmp = BERT_DIR.with_name(BERT_DIR.name + "_DOWNLOADING")
        if tmp.exists(): shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        try:
            model.save_pretrained(str(tmp), safe_serialization=True)
            tokenizer.save_pretrained(str(tmp))
            official_vocab = Path(hf_hub_download(repo_id=REPO, filename="vocab.txt", revision=PINNED_COMMIT))
            shutil.copyfile(official_vocab, tmp / "vocab.txt")
            weight_file = next((tmp / w for w in WEIGHTS if (tmp / w).is_file()), None)
            assert weight_file is not None, "Missing BERT weights after download"
            meta = {
                "repo_id": REPO,
                "source_revision": PINNED_COMMIT,
                "saved_weights_file": weight_file.name,
                "saved_weights_sha256": sha256_bytes(weight_file),
                "bert_config_sha256": sha256_bytes(tmp / "config.json"),
                "vocab_sha256": sha256_bytes(tmp / "vocab.txt"),
                "vocab_size": 30522,
                "hidden_size": 768
            }
            tmp.rename(BERT_DIR)
            SOURCE_META.write_text(json.dumps(meta, indent=2) + "\\n", encoding="utf-8")
        finally:
            if tmp.exists(): shutil.rmtree(tmp)
            del model, tokenizer
            gc.collect()

# Set offline mode after download/cache
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

print(f"Official BERT ready at: {BERT_DIR}")
print("Transformers offline mode: ENABLED", flush=True)
"""))

    # ── Cell 2: Mount & Verify Bundle ────────────────────────────────────────
    cell2_code = """\
# Cell 2: Mount GPU Bundle & Fail-Closed Integrity Verification
import os, sys, json, hashlib, zipfile, shutil
from pathlib import Path

WORK = Path("/kaggle/working")
INP = Path("/kaggle/input")
EXPECTED_BUNDLE_NAME = "@BUNDLE_NAME@"
EXPECTED_BUNDLE_SHA256 = "@BUNDLE_SHA256@"

def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

print(f"Looking for bundle '{EXPECTED_BUNDLE_NAME}' (SHA256: {EXPECTED_BUNDLE_SHA256[:12]}...)...", flush=True)

bundle_source = None
# 1. Search for matching zip
for root, dirs, files in os.walk(INP):
    for f in files:
        if f.lower().endswith(".zip") and ("lhrm" in f.lower() or "bundle" in f.lower()):
            candidate = Path(root) / f
            c_sha = sha_file(candidate)
            if c_sha == EXPECTED_BUNDLE_SHA256:
                bundle_source = ("zip", candidate)
                print(f"Found verified bundle ZIP: {candidate}")
                break
    if bundle_source:
        break

# 2. Search for pre-extracted bundle dataset
if not bundle_source:
    for root, dirs, files in os.walk(INP):
        cand_dir = Path(root)
        manifest_p = cand_dir / "BUNDLE_MANIFEST.json"
        if manifest_p.is_file() and (cand_dir / "run_baseline.py").is_file():
            try:
                m = json.loads(manifest_p.read_text("utf-8"))
                if m.get("bundle_name") == EXPECTED_BUNDLE_NAME:
                    all_match = all((cand_dir / rel).is_file() and sha_file(cand_dir / rel) == d
                                    for rel, d in m.get("files", {}).items())
                    if all_match:
                        bundle_source = ("directory", cand_dir)
                        print(f"Found verified bundle directory: {cand_dir}")
                        break
            except Exception:
                pass
        if bundle_source:
            break

if not bundle_source:
    raise FileNotFoundError(
        f"Approved bundle NOT found or SHA256 mismatch!\\n"
        f"Expected filename: {EXPECTED_BUNDLE_NAME}\\n"
        f"Expected SHA-256:  {EXPECTED_BUNDLE_SHA256}\\n"
        f"Please upload and attach {EXPECTED_BUNDLE_NAME} as a Kaggle dataset."
    )

# Extract / stage bundle into /kaggle/working
if bundle_source[0] == "zip":
    with zipfile.ZipFile(bundle_source[1], "r") as z:
        for n in z.namelist():
            if Path(n).is_absolute() or ".." in Path(n).parts:
                raise ValueError(f"Unsafe zip member: {n}")
        z.extractall(WORK)
else:
    for root, dirs, files in os.walk(bundle_source[1]):
        for f in files:
            src = Path(root) / f
            rel = src.relative_to(bundle_source[1])
            dest = WORK / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if src.resolve() != dest.resolve():
                shutil.copy2(src, dest)

# Verify manifest and files
manifest = json.loads((WORK / "BUNDLE_MANIFEST.json").read_text("utf-8"))
for rel, digest in manifest["files"].items():
    actual_d = sha_file(WORK / rel)
    assert actual_d == digest, f"Corrupted bundle file: {rel} (expected {digest}, got {actual_d})"
print(f"Bundle files verified: PASS ({len(manifest['files'])} files OK)", flush=True)

# Verify locked 200 validation sample IDs
val_ids_file = WORK / "experiments/RUN_0/run0_val_sample_ids.json"
assert val_ids_file.is_file(), "Locked validation IDs missing from bundle!"
val_ids = json.loads(val_ids_file.read_text())
assert len(val_ids) == 200, f"Expected 200 validation IDs; got {len(val_ids)}"
assert val_ids[:5] == [3, 6, 11, 25, 27], f"Locked val IDs prefix mismatch: {val_ids[:5]}"
print("Locked validation IDs verified: PASS (200 IDs, exact RUN_0 subset)")

# Verify CPU step-0 preflight reference contains L2 and L2_NO_SA
ref_file = WORK / "docs/phase1/run0_local_step0_crosscheck.json"
assert ref_file.is_file(), "Missing step-0 cross-check reference file!"
ref_data = json.load(open(ref_file))
assert "L2" in ref_data.get("variants", {}), "L2 missing from crosscheck reference"
assert "L2_NO_SA" in ref_data.get("variants", {}), "L2_NO_SA missing from crosscheck reference"
print("CPU Step-0 cross-check reference verified: PASS (L, L2, L2_NO_SA present)")
""".replace("@BUNDLE_NAME@", bundle_name).replace("@BUNDLE_SHA256@", bundle_sha256)
    cells.append(cell(cell2_code))

    # ── Cell 3: Mount AAPD Data & Quarantine Test Split ──────────────────────
    cells.append(cell("""\
# Cell 3: Mount AAPD (Train + Dev ONLY) — STRICT TEST DATA QUARANTINE
import os, sys, json, hashlib, zipfile, shutil
from pathlib import Path

WORK = Path("/kaggle/working")
INP = Path("/kaggle/input")
sys.path.insert(0, str(WORK / "src"))

EXPECTED_CSV_SHA = {
    "train": "e7e73064385f78fe092edd47d4e4059c35875a1ea2a95859ad3cde091ed9af95",
    "dev":   "a85f612d4c1cb2be6d1de94cbcd9468cba97d221f826014264f4d279c82e220b",
}
EXPECTED_LABEL_ORDER_SHA = "a3d662f3920259a94ed61d83f98f21cabe7f908d98a5662ccd9adc2413cc000b"

def sha_stream(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1024*1024), b""):
        h.update(block)
    return h.hexdigest()

def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

# Locate authoritative raw Zenodo CSV files
valid_sources = []
for root, dirs, files in os.walk(INP):
    p = Path(root)
    # Check for direct CSVs
    if "train.csv" in files and "dev.csv" in files:
        hashes = {s: sha_file(p / f"{s}.csv") for s in ("train", "dev")}
        if hashes == EXPECTED_CSV_SHA:
            valid_sources.append(("directory", p, None))
    # Check for zip
    for f in files:
        if f.lower().endswith(".zip") and "aapd" in f.lower():
            zp = p / f
            try:
                with zipfile.ZipFile(zp) as z:
                    names = z.namelist()
                    members = {}
                    for s in ("train", "dev"):
                        m = [n for n in names if n.replace("\\\\", "/").endswith(f"/{s}.csv") or n == f"{s}.csv"]
                        if len(m) == 1:
                            members[s] = m[0]
                    if len(members) == 2:
                        hashes = {s: sha_stream(z.open(members[s])) for s in ("train", "dev")}
                        if hashes == EXPECTED_CSV_SHA:
                            valid_sources.append(("zip", zp, members))
            except Exception:
                pass

if not valid_sources:
    raise FileNotFoundError(
        "Authoritative Zenodo AAPD dataset (train.csv & dev.csv) NOT found or hash mismatch!\\n"
        "Please attach the official Zenodo AAPD dataset."
    )

kind, src_path, members = valid_sources[0]
print(f"AAPD raw source: VERIFIED (kind={kind}, path={src_path})", flush=True)

# Stage Train & Dev into a quarantined zip (deliberately excluding test.csv)
STAGED_ZIP = WORK / "AAPD_TRAIN_DEV_QUARANTINED.zip"
with zipfile.ZipFile(STAGED_ZIP, "w", compression=zipfile.ZIP_STORED) as out_z:
    for split in ("train", "dev"):
        with out_z.open(f"AAPD/{split}.csv", "w") as dest:
            if kind == "zip":
                with zipfile.ZipFile(src_path) as in_z, in_z.open(members[split]) as src:
                    shutil.copyfileobj(src, dest, 1024*1024)
            else:
                with open(src_path / f"{split}.csv", "rb") as src:
                    shutil.copyfileobj(src, dest, 1024*1024)

# Prepare processed data using canonical loader
from lhrm_baseline.data import load_aapd_split, save_processed
from lhrm_baseline.labels import build_all_label_texts

processed_dir = WORK / "data/processed/AAPD"
processed_dir.mkdir(parents=True, exist_ok=True)

label_order = None
for split, expected_n in (("train", 53840), ("dev", 1000)):
    texts, labels, cols = load_aapd_split(str(STAGED_ZIP), split)
    assert len(texts) == expected_n, f"Unexpected doc count for {split}: {len(texts)}"
    assert labels.shape == (expected_n, 54), f"Unexpected label shape for {split}: {labels.shape}"
    if label_order is None:
        label_order = cols
    else:
        assert cols == label_order, f"Label order mismatch between train and dev"
    save_processed(str(processed_dir), split, texts, labels, cols)
    print(f"Prepared {split} split: {expected_n} docs, 54 labels OK", flush=True)

# Verify label order hash
lo_hash = hashlib.sha256(json.dumps(label_order).encode()).hexdigest()
assert lo_hash == EXPECTED_LABEL_ORDER_SHA, f"Label order hash mismatch: {lo_hash}"
print("Label order hash: OK (a3d662f39202...)", flush=True)

# Build label definitions
defs_file = WORK / "Label_Definition/arxiv_labels_definitions.json"
label_texts = build_all_label_texts(label_order, str(defs_file), dataset="aapd")
assert len(label_texts) == 54 and all(t.strip() for t in label_texts)
(processed_dir / "label_texts.json").write_text(json.dumps(label_texts, indent=2, ensure_ascii=False), encoding="utf-8")

# STRICT TEST SPLIT QUARANTINE ASSERTION
test_files = list(processed_dir.glob("test_*"))
assert len(test_files) == 0, f"VIOLATION: Test split detected in processed directory: {test_files}"
print("TEST SPLIT QUARANTINE VERIFIED: No test files read, copied, or prepared.", flush=True)
"""))

    # ── Cell 4: Staging Experiment & Config Preparation ──────────────────────
    cell4_code = """\
# Cell 4: Setup Runtime Config, Locked Validation IDs & Fail-Closed Guard
import os, sys, json, shutil
from pathlib import Path

WORK = Path("/kaggle/working")
RUN_ID = "@RUN_ID@"
BASE_CFG = WORK / "@CONFIG@"
EXP_DIR = WORK / "experiments" / RUN_ID

# 1. Fail-closed check if experiment directory already exists with prior results
# Note: Do not create or populate EXP_DIR here. run_baseline.py::cmd_run0 requires
# the target directory to either not exist or be empty, and creates/populates it itself.
if EXP_DIR.exists() and (EXP_DIR.is_file() or any(EXP_DIR.iterdir())):
    raise RuntimeError(
        f"FAIL-CLOSED: Target experiment directory {EXP_DIR} already exists and is non-empty!\\n"
        f"Never overwrite existing experiment artifacts."
    )

# 2. Validate original frozen 200 Dev sample IDs directly from RUN_0 reference
src_ids_file = WORK / "experiments/RUN_0/run0_val_sample_ids.json"
assert src_ids_file.is_file(), "Locked validation IDs missing from bundle at experiments/RUN_0/run0_val_sample_ids.json!"
val_ids = json.load(open(src_ids_file))
assert len(val_ids) == 200, f"Expected 200 validation IDs, got {len(val_ids)}"
assert val_ids[:5] == [3, 6, 11, 25, 27], "Validation IDs prefix mismatch"
print("Validated locked 200 validation IDs directly from RUN_0: PASS", flush=True)

# 3. Verify frequency prior on the 200 validation docs is exactly 63/200
import numpy as np
from lhrm_baseline.diagnostics import frequency_prior_hits_at_1
train_labels = np.load(WORK / "data/processed/AAPD/train_labels.npy")
dev_labels = np.load(WORK / "data/processed/AAPD/dev_labels.npy")
v_labels = dev_labels[val_ids]
prior_hits = frequency_prior_hits_at_1(train_labels, v_labels, seed=42)
assert prior_hits == 63, f"Prior hit mismatch: expected 63, got {prior_hits}"
print(f"Frequency prior on locked 200 dev docs: {prior_hits}/200 ({prior_hits/200:.4f}) — PASS", flush=True)

# 4. Create runtime config overriding only the local BERT model path
BERT_PATH = WORK / "google_bert_bert_base_uncased_official"
cfg_text = BASE_CFG.read_text(encoding="utf-8")
assert "bert_model: bert-base-uncased" in cfg_text, "Expected 'bert_model: bert-base-uncased' in base config"
RUNTIME_CFG = WORK / "configs/final/aapd_ablation_runtime.yaml"
RUNTIME_CFG.parent.mkdir(parents=True, exist_ok=True)
RUNTIME_CFG.write_text(cfg_text.replace("bert_model: bert-base-uncased", f"bert_model: {str(BERT_PATH)}"), encoding="utf-8")

from run_baseline import load_yaml_config
before = load_yaml_config(str(BASE_CFG))
after = load_yaml_config(str(RUNTIME_CFG))
diff = {k: (before[k], after[k]) for k in before if before[k] != after[k]}
assert diff == {"bert_model": ("bert-base-uncased", str(BERT_PATH))}, f"Unexpected config alteration: {diff}"
print(f"Runtime config verified: {RUNTIME_CFG} (BERT override only)", flush=True)
""".replace("@RUN_ID@", RUN_ID).replace("@CONFIG@", CONFIG)
    cells.append(cell(cell4_code))

    # ── Cell 5: Run 300-Step Screening CLI ────────────────────────────────────
    cell5_code = """\
# Cell 5: Execute 300-Step Screening CLI (L2 vs L2_NO_SA)
import os, sys, subprocess, time
from pathlib import Path

WORK = Path("/kaggle/working")
RUN_ID = "@RUN_ID@"
RUNTIME_CFG = WORK / "configs/final/aapd_ablation_runtime.yaml"
CANDIDATES = "@CANDIDATES@"
N_STEPS = @N_STEPS@

cmd = [
    sys.executable, "-u", str(WORK / "run_baseline.py"), "run0",
    "--config", str(RUNTIME_CFG),
    "--run_id", RUN_ID,
    "--steps", str(N_STEPS),
    "--candidates", CANDIDATES
]

print("=== STARTING ABLATION SCREENING CLI ===", flush=True)
print("Command:", " ".join(cmd), flush=True)

env = os.environ.copy()
env.update({
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "PYTHONUNBUFFERED": "1",
})

t_start = time.time()
with subprocess.Popen(cmd, cwd=str(WORK), env=env,
                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                      bufsize=1, text=True) as p:
    for line in p.stdout:
        print(line, end="", flush=True)
    ret_code = p.wait()

elapsed = time.time() - t_start
print(f"\\nCLI finished in {elapsed:.1f}s with exit code {ret_code}", flush=True)

if ret_code == 2:
    raise RuntimeError(
        "FAIL-STOP: Step-0 preflight cross-check failed!\\n"
        "GPU values deviated from CPU fp32 reference. Check preflight_fail_*.json."
    )
elif ret_code != 0:
    raise RuntimeError(f"run_baseline.py run0 failed with exit code {ret_code}")

print("Screening CLI completed successfully. Proceeding to diagnostics and diversity evaluation.", flush=True)
""".replace("@RUN_ID@", RUN_ID).replace("@CANDIDATES@", CANDIDATES).replace("@N_STEPS@", str(N_STEPS))
    cells.append(cell(cell5_code))

    # ── Cell 6: Diagnostic Audit (V1–V5 Integer Comparison) ───────────────────
    cell6_code = """\
# Cell 6: Artifact Integrity & Criteria Audit (V1–V5 Integer Comparison)
import os, sys, json, hashlib
from pathlib import Path

WORK = Path("/kaggle/working")
RUN_ID = "@RUN_ID@"
EXP_DIR = WORK / "experiments" / RUN_ID

# Verify required artifacts for both candidates
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

for cand in ["L2", "L2_NO_SA"]:
    c_dir = EXP_DIR / f"variant_{cand}"
    for f in REQUIRED_PER_CAND:
        assert (c_dir / f).is_file(), f"Missing candidate artifact: variant_{cand}/{f}"

print("All required experiment artifacts present.", flush=True)

# Verify SHA256SUMS.txt consistency
sums_file = EXP_DIR / "SHA256SUMS.txt"
sums_dict = {}
for line in sums_file.read_text().splitlines():
    if line.strip():
        sha, rel = line.split("  ", 1)
        sums_dict[rel] = sha

def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

mismatches = []
for rel, expected_sha in sums_dict.items():
    p = EXP_DIR / rel
    if not p.is_file():
        mismatches.append(f"MISSING: {rel}")
    elif sha_file(p) != expected_sha:
        mismatches.append(f"SHA MISMATCH: {rel}")

assert not mismatches, "SHA256SUMS.txt verification failed:\\n" + "\\n".join(mismatches)
print(f"SHA256SUMS.txt verified: PASS ({len(sums_dict)} files)", flush=True)

# Load diagnostics and evaluate V1-V5
print("\\n" + "="*65)
print("CRITERIA EVALUATION (V1–V5)")
print("="*65)

results = {}
for cand in ["L2", "L2_NO_SA"]:
    s0 = json.load(open(EXP_DIR / f"variant_{cand}" / "step0_diagnostics.json"))
    s300 = json.load(open(EXP_DIR / f"variant_{cand}" / "step300_diagnostics.json"))
    
    # Read training log for losses
    losses = []
    for line in (EXP_DIR / f"variant_{cand}" / "train.log").read_text().splitlines():
        if line.startswith("step="):
            losses.append(float(line.split("loss=")[1]))
    
    nan_inf = s300.get("nan_inf_count", 0)
    V1 = nan_inf == 0
    first50 = losses[:50] if len(losses) >= 50 else losses
    last50 = losses[-50:] if len(losses) >= 50 else losses
    mean_first50 = sum(first50) / len(first50)
    mean_last50 = sum(last50) / len(last50)
    V2 = mean_last50 < mean_first50
    
    alpha_one_hot = float(s300.get("alpha_one_hot_frac", 1.0))
    V3 = alpha_one_hot < 0.9
    
    logit_std = s300.get("per_label_logit_std", [])
    v4_labels = sum(float(x) > 0.01 for x in logit_std)
    V4 = v4_labels >= 28  # strictly > 50% of 54 labels
    
    cand_hits = int(s300.get("p_at_1_hits", 0))
    prior_hits = int(s300.get("prior_p_at_1_hits", 63))
    V5 = cand_hits > prior_hits  # strict integer comparison
    
    viable = V1 and V2 and V3 and V4 and V5
    results[cand] = {
        "V1": V1, "V2": V2, "V3": V3, "V4": V4, "V5": V5,
        "viable": viable,
        "p_at_1_hits": cand_hits,
        "prior_p_at_1_hits": prior_hits,
        "v4_labels_passing": v4_labels,
        "alpha_one_hot_frac": alpha_one_hot,
        "mean_loss_last50": mean_last50,
    }
    
    print(f"\\n--- Variant {cand} ---")
    print(f"  V1 (No NaN/Inf):     {V1} (nan_count={nan_inf})")
    print(f"  V2 (Loss Decreased): {V2} (first50={mean_first50:.4f} -> last50={mean_last50:.4f})")
    print(f"  V3 (Alpha One-Hot):  {V3} (one_hot_frac={alpha_one_hot:.4f} < 0.9)")
    print(f"  V4 (Logit Variance): {V4} ({v4_labels}/54 labels > 0.01; threshold=28)")
    print(f"  V5 (P@1 > Prior):    {V5} (hits: {cand_hits}/200 vs prior: {prior_hits}/200 [integer strict >])")
    print(f"  VIABLE (All Pass):   {viable}")
""".replace("@RUN_ID@", RUN_ID)
    cells.append(cell(cell6_code))

    # ── Cell 7: Invariants & Ranking Diversity Analysis ──────────────────────
    cell7_code = """\
# Cell 7: Invariant Checks & Ranking Diversity Analysis
import os, sys, json
import numpy as np
from pathlib import Path

WORK = Path("/kaggle/working")
RUN_ID = "@RUN_ID@"
EXP_DIR = WORK / "experiments" / RUN_ID

# 1. Step-0 Invariant Cross-Check between L2 and L2_NO_SA
s0_l2 = json.load(open(EXP_DIR / "variant_L2/step0_diagnostics.json"))
s0_nosa = json.load(open(EXP_DIR / "variant_L2_NO_SA/step0_diagnostics.json"))

print("="*65)
print("STEP-0 MATHEMATICAL INVARIANT VERIFICATION")
print("="*65)
print(f"mean_max_alpha:     L2={s0_l2['mean_max_alpha']:.6f}  L2_NO_SA={s0_nosa['mean_max_alpha']:.6f}  diff={abs(s0_l2['mean_max_alpha']-s0_nosa['mean_max_alpha']):.2e}")
print(f"alpha_one_hot_frac: L2={s0_l2['alpha_one_hot_frac']:.6f}  L2_NO_SA={s0_nosa['alpha_one_hot_frac']:.6f}")
print(f"S_mean:             L2={s0_l2['S_mean']:.6f}  L2_NO_SA={s0_nosa['S_mean']:.6f}  diff={abs(s0_l2['S_mean']-s0_nosa['S_mean']):.2e}")
print(f"n_distinct_argmax:  L2={s0_l2['n_distinct_argmax']}  L2_NO_SA={s0_nosa['n_distinct_argmax']}")

# Verify identity of alpha distribution under identical initial weights
assert abs(s0_l2['mean_max_alpha'] - s0_nosa['mean_max_alpha']) < 1e-4, "Step-0 alpha mismatch!"
assert abs(s0_l2['S_mean'] - s0_nosa['S_mean']) < 1e-3, "Step-0 S_mean mismatch!"
print("Step-0 Invariant Verification: PASS (Identical attention distribution under shared init)")

# 2. Ranking Diversity Analysis at Step 300
print("\\n" + "="*65)
print("RANKING DIVERSITY ANALYSIS (STEP 300)")
print("="*65)

# Load canonical label order for AAPD to verify consistent ordering across records
labels_file = WORK / "data/processed/AAPD/labels.json"
canonical_labels = json.load(open(labels_file))
assert len(canonical_labels) == 54, f"Expected 54 canonical labels, got {len(canonical_labels)}"

diversity_results = {}
for cand in ["L2", "L2_NO_SA"]:
    preds_p = EXP_DIR / f"variant_{cand}/val_sample_predictions.json"
    records = json.load(open(preds_p))
    assert isinstance(records, list), f"Expected list of prediction records, got {type(records)}"
    assert len(records) == 200, f"Expected 200 prediction records, got {len(records)}"
    
    # Extract prediction scores vector (probabilities from sigmoid) for each document
    scores_matrix = []
    for idx, rec in enumerate(records):
        assert "scores" in rec and "labels" in rec, f"Record {idx} missing required fields ('scores', 'labels')"
        assert rec["labels"] == canonical_labels, f"Record {idx} label order does not match canonical label order"
        rec_scores = rec["scores"]
        assert len(rec_scores) == 54, f"Record {idx} scores length {len(rec_scores)} != 54"
        scores_matrix.append(rec_scores)
    
    scores = np.array(scores_matrix, dtype=np.float32)  # [200, 54] prediction scores / probabilities
    assert scores.shape == (200, 54), f"Unexpected prediction scores shape: {scores.shape}"
    assert np.all(np.isfinite(scores)), f"Variant {cand} prediction scores contain NaN or Inf!"
    
    # Top-1 label per doc (based on prediction scores/probabilities)
    top1_indices = np.argmax(scores, axis=1)
    distinct_top1 = len(set(top1_indices.tolist()))
    
    # Ordered Top-3 labels per doc
    top3_indices = np.argsort(-scores, axis=1)[:, :3]
    top3_tuples = [tuple(row.tolist()) for row in top3_indices]
    distinct_top3 = len(set(top3_tuples))
    
    # Most frequent top-1 label frequency
    counts = np.bincount(top1_indices, minlength=54)
    dominant_label_idx = int(np.argmax(counts))
    dominant_label_count = int(np.max(counts))
    
    diversity_results[cand] = {
        "distinct_top1_labels": distinct_top1,
        "distinct_top3_rankings": distinct_top3,
        "dominant_top1_label_index": dominant_label_idx,
        "dominant_top1_count": dominant_label_count,
        "dominant_top1_fraction": dominant_label_count / 200.0,
    }
    
    print(f"--- Variant {cand} ---")
    print(f"  Distinct Top-1 labels across 200 docs:      {distinct_top1} / 54")
    print(f"  Distinct ordered Top-3 ranking patterns:     {distinct_top3} / 200")
    print(f"  Dominant Top-1 label: index {dominant_label_idx} appears on {dominant_label_count}/200 docs ({dominant_label_count/2:.1f}%)")

# Side-by-side comparative summary
print("\\n" + "="*65)
print("COMPARATIVE SUMMARY TABLE")
print("="*65)
print("{:<32} | {:<15} | {:<15}".format("Metric", "Variant L2", "Variant L2_NO_SA"))
print("-" * 65)
print("{:<32} | {:<15} | {:<15}".format("V1 (no NaN)", str(results['L2']['V1']), str(results['L2_NO_SA']['V1'])))
print("{:<32} | {:<15} | {:<15}".format("V2 (loss down)", str(results['L2']['V2']), str(results['L2_NO_SA']['V2'])))
print("{:<32} | {:<15} | {:<15}".format("V3 (alpha one-hot < 0.9)", str(results['L2']['V3']), str(results['L2_NO_SA']['V3'])))
v4_l2 = "{} ({}/54)".format(results['L2']['V4'], results['L2']['v4_labels_passing'])
v4_nosa = "{} ({}/54)".format(results['L2_NO_SA']['V4'], results['L2_NO_SA']['v4_labels_passing'])
print("{:<32} | {:<15} | {:<15}".format("V4 (logit std > 0.01 on >28)", v4_l2, v4_nosa))
v5_l2 = "{} ({}/200)".format(results['L2']['V5'], results['L2']['p_at_1_hits'])
v5_nosa = "{} ({}/200)".format(results['L2_NO_SA']['V5'], results['L2_NO_SA']['p_at_1_hits'])
print("{:<32} | {:<15} | {:<15}".format("V5 (P@1 hits > 63 integer)", v5_l2, v5_nosa))
print("{:<32} | {:<15} | {:<15}".format("VIABLE (all 5 criteria)", str(results['L2']['viable']), str(results['L2_NO_SA']['viable'])))
d1_l2 = "{}/54".format(diversity_results['L2']['distinct_top1_labels'])
d1_nosa = "{}/54".format(diversity_results['L2_NO_SA']['distinct_top1_labels'])
print("{:<32} | {:<15} | {:<15}".format("Distinct Top-1 Labels", d1_l2, d1_nosa))
d3_l2 = "{}/200".format(diversity_results['L2']['distinct_top3_rankings'])
d3_nosa = "{}/200".format(diversity_results['L2_NO_SA']['distinct_top3_rankings'])
print("{:<32} | {:<15} | {:<15}".format("Distinct Top-3 Rankings", d3_l2, d3_nosa))
print("="*65)
""".replace("@RUN_ID@", RUN_ID)
    cells.append(cell(cell7_code))

    # ── Cell 8: Package Lightweight Evidence ZIP ─────────────────────────────
    cell8_code = """\
# Cell 8: Package Lightweight Evidence ZIP (Excluding Heavy Checkpoints)
import os, sys, json, zipfile, hashlib, time
from pathlib import Path

WORK = Path("/kaggle/working")
RUN_ID = "@RUN_ID@"
EXP_DIR = WORK / "experiments" / RUN_ID

# Create detailed execution summary
exec_summary = {
    "experiment_id": RUN_ID,
    "purpose": "CONTROLLED_CSAM_NO_SA_ABLATION_SCREENING",
    "steps_per_candidate": @N_STEPS@,
    "candidates": ["L2", "L2_NO_SA"],
    "seed": 42,
    "frequency_prior_hits": 63,
    "n_val_docs": 200,
    "criteria_results": results,
    "diversity_results": diversity_results,
    "scientific_notes": [
        "V5 compares integer P@1 hit count against frequency prior 63.",
        "Passing V5 after 300 steps is positive preliminary screening evidence, not proof of root cause.",
        "Increased ranking diversity alone is not proof of improved correctness.",
        "No automatic 1000-step extension or full training run authorized.",
        "Test split was quarantined byte-for-byte."
    ],
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}

summary_file = EXP_DIR / "ABLATION_SUMMARY.json"
summary_file.write_text(json.dumps(exec_summary, indent=2), encoding="utf-8")

# Package lightweight evidence zip (exclude *.pt, *.safetensors)
timestamp = time.strftime("%Y%m%d_%H%M%S")
EVIDENCE_ZIP = WORK / f"{RUN_ID}_EVIDENCE_{timestamp}.zip"

written_files = []
with zipfile.ZipFile(EVIDENCE_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for path in sorted(EXP_DIR.rglob("*")):
        if path.is_file() and path.suffix.lower() not in (".pt", ".pth", ".ckpt", ".bin", ".safetensors"):
            rel = f"{RUN_ID}/" + path.relative_to(EXP_DIR).as_posix()
            z.write(path, rel)
            written_files.append(rel)
    # Include input provenance and configs
    if (WORK / "configs/final/aapd_ablation_runtime.yaml").is_file():
        z.write(WORK / "configs/final/aapd_ablation_runtime.yaml", "aapd_ablation_runtime.yaml")
    if (WORK / "OFFICIAL_BERT_SOURCE.json").is_file():
        z.write(WORK / "OFFICIAL_BERT_SOURCE.json", "OFFICIAL_BERT_SOURCE.json")

def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

zip_sha = sha_file(EVIDENCE_ZIP)
zip_size_kb = EVIDENCE_ZIP.stat().st_size / 1024.0

print("\\n" + "="*65)
print("EVIDENCE PACKAGE COMPLETE")
print("="*65)
print(f"Evidence ZIP: {EVIDENCE_ZIP}")
print(f"SHA-256:      {zip_sha}")
print(f"Size:         {zip_size_kb:.1f} KB")
print(f"Files included: {len(written_files)} (heavy .pt checkpoints excluded)")
print("\\nDownload this lightweight evidence ZIP and inspect results.")
print("Do not run full training or modify model architecture without audit review.")
""".replace("@RUN_ID@", RUN_ID).replace("@N_STEPS@", str(N_STEPS))
    cells.append(cell(cell8_code))

    nb = {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
            "kaggle": {
                "accelerator": "GPU",
                "dataSources": [
                    {"sourceType": "datasetVersion", "sourceId": "lhrm-l2-no-sa-ablation-bundle"},
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
    parser = argparse.ArgumentParser(description="Generate Kaggle notebook for L2 vs L2_NO_SA ablation")
    parser.add_argument("--build_bundle", action="store_true",
                        help="Build fresh GPU bundle and compute authentic SHA-256")
    parser.add_argument("--bundle_name", default=BUNDLE_NAME)
    args = parser.parse_args()

    # Step 1: Build or locate bundle
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

    # Step 2: Assemble notebook with authentic bundle SHA
    nb = build_notebook(bundle_name, bundle_sha)
    os.makedirs(os.path.dirname(OUT_NB), exist_ok=True)
    with open(OUT_NB, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Notebook written: {OUT_NB}")
    print(f"  Cells: {len(nb['cells'])}")

    # Verify JSON round-trip
    with open(OUT_NB, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert len(loaded["cells"]) == len(nb["cells"]), "JSON verification failed"
    print("Notebook JSON verification: PASS")


if __name__ == "__main__":
    main()
