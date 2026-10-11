#!/usr/bin/env python3
"""
scripts/gen_kaggle_notebook_replay.py

Generates notebooks/lhrm_dr3_trained_checkpoint_replay_kaggle.ipynb —
Dedicated, reproducible Kaggle GPU notebook for RUN_0_L2_NO_SA_REPLAY_20261011_B:
1. Reproduce 300-step screening of L2 and L2_NO_SA.
2. Save full step-300 checkpoints.
3. Export lightweight model-only checkpoints (L2_STEP300_MODEL_ONLY.pt, L2_NO_SA_STEP300_MODEL_ONLY.pt).
4. Reload and verify models with load_state_dict(strict=True).
5. Execute real trained-checkpoint gradient micro-probe on authentic representations.
6. Package lightweight evidence ZIP and manifest.
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

BUNDLE_NAME = "lhrm_gpu_bundle_l2_no_sa_replay_20261011_b.zip"
OUT_NB = os.path.join(ROOT, "notebooks", "lhrm_dr3_trained_checkpoint_replay_kaggle.ipynb")
RUN_ID = "RUN_0_L2_NO_SA_REPLAY_20261011_B"
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
# LHRM DR3 — Trained-Checkpoint Replay & Gradient Diagnostic ({RUN_ID})

**Controlled Replay & Post-Training Gradient Diagnostic (300 Steps per Variant)**

This Kaggle session replicates the 300-step screening of `L2` and `L2_NO_SA` to recover trained model checkpoints and perform a real post-training gradient diagnostic.

### Research Protocol & Constraints:
1. **Experiment ID:** `{RUN_ID}` (fail-closed if output directory already exists).
2. **Candidates:** `L2` (full CSAM with label SA) vs `L2_NO_SA` (CSAM without label SA).
3. **Training Setup:** Seed 42, Focal Loss (gamma=2.0, alpha=0.25), LR 2e-5, weight decay 0.01, AMP enabled, 300 steps per variant.
4. **Data Isolation:** **Test split is strictly quarantined.** Never read, copied, prepared, predicted, or evaluated. Original AAPD `train.csv` and `dev.csv` source files are required from the verified AAPD dataset and verified against official SHA-256 hashes.
5. **Pretrained BERT:** Official `google-bert/bert-base-uncased` frozen revision `86b5e0934494bd15c9632b12f734a8a67f723594` (matching historical `OFFICIAL_BERT_SOURCE.json`).
6. **Fixed Dev Sample:** Locked 200 validation documents from `experiments/RUN_0/run0_val_sample_ids.json` (frequency prior: exactly 63/200).
7. **Deliverables in `/kaggle/working/`:**
   - `L2_STEP300_MODEL_ONLY.pt`
   - `L2_NO_SA_STEP300_MODEL_ONLY.pt`
   - `DR3_TRAINED_GRADIENT_PROBE_RESULTS.json`
   - `DR3_TRAINED_CHECKPOINT_MANIFEST.json`
   - `DR3_REPLAY_EVIDENCE.zip`
   - `DR3_REPLAY_SHA256SUMS.txt`

**Bundle:** `{bundle_name}`  
**Bundle SHA-256:** `{bundle_sha256}`
""", cell_type="markdown"))

    # ── Cell 1: Environment & Official BERT Setup ────────────────────────────
    cells.append(cell("""\
# Cell 1: Environment check & Official BERT Acquisition
import os, sys, time, json, hashlib, shutil, gc
from pathlib import Path
import torch

print("=== DR3 Replay: Environment & Device Gate ===", flush=True)
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

    # ── Cell 2: Verify & Stage Bundle & AAPD Train/Dev ───────────────────────
    cell2_code = """\
# Cell 2: Verify & Stage Bundle and AAPD Train/Dev Data
from pathlib import Path
import os, json, hashlib, zipfile, shutil, sys

WORK = Path("/kaggle/working")
INP = Path("/kaggle/input")
EXPECTED_BUNDLE_SHA = "@BUNDLE_SHA256@"
EXPECTED_DATA_ZIP_SHA = "e1b418427bcd2662f04c347099deb0d520b4066f782ed40c2a4aed15df694a08"
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

# Locate Bundle ZIP or Directory
bundle_source = None
for base, ds, fs in os.walk(INP):
    root = Path(base)
    for name in fs:
        if name.lower().endswith(".zip") and ("replay" in name.lower() or "ablation" in name.lower() or "lhrm" in name.lower()):
            zp = root / name
            if sha_file(zp) == EXPECTED_BUNDLE_SHA:
                bundle_source = ("zip", zp)
                break
    if bundle_source:
        break
    if "BUNDLE_MANIFEST.json" in fs and "run_baseline.py" in fs:
        try:
            m = json.loads((root / "BUNDLE_MANIFEST.json").read_text("utf-8"))
            if all((root / rel).is_file() and sha_file(root / rel) == d for rel, d in m["files"].items()):
                bundle_source = ("directory", root)
                break
        except Exception:
            pass

if not bundle_source:
    raise FileNotFoundError(
        f"Authentic GPU bundle matching SHA256 {EXPECTED_BUNDLE_SHA} not found in /kaggle/input.\\n"
        "Please attach the fresh replay bundle as a Kaggle dataset."
    )

print("Bundle source VERIFIED:", *bundle_source, flush=True)
if bundle_source[0] == "zip":
    with zipfile.ZipFile(bundle_source[1]) as z:
        z.extractall(WORK)
else:
    for base, ds, fs in os.walk(bundle_source[1]):
        for n in fs:
            src = Path(base) / n
            rel = src.relative_to(bundle_source[1])
            dest = WORK / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if src.resolve() != dest.resolve():
                shutil.copy2(src, dest)

manifest = json.loads((WORK / "BUNDLE_MANIFEST.json").read_text("utf-8"))
for rel, dig in manifest["files"].items():
    assert sha_file(WORK / rel) == dig, f"Bundle component differs: {rel}"
print(f"Bundle integrity PASS: {len(manifest['files'])} files verified.", flush=True)

# Locate AAPD train/dev CSVs
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
    raise RuntimeError("Authentic AAPD train/dev source CSVs not found in /kaggle/input.")

kind, data_path, members, outer_hash = valid[0]
print("AAPD source VERIFIED:", kind, data_path, flush=True)

# Stage Train & Dev Only
STAGED = WORK / "AAPD_REPLAY_TRAIN_DEV_ONLY.zip"
with zipfile.ZipFile(STAGED, "w", compression=zipfile.ZIP_STORED) as out:
    for split in ("train", "dev"):
        with out.open("AAPD/" + split + ".csv", "w") as dest:
            if kind == "zip":
                with zipfile.ZipFile(data_path) as inz, inz.open(members[split]) as src:
                    shutil.copyfileobj(src, dest, 1024*1024)
            else:
                with open(data_path / (split + ".csv"), "rb") as src:
                    shutil.copyfileobj(src, dest, 1024*1024)

# Prepare Processed Data (NO TEST SPLIT)
sys.path.insert(0, str(WORK / "src"))
from lhrm_baseline.data import load_aapd_split, save_processed
from lhrm_baseline.labels import build_all_label_texts

processed = WORK / "data/processed/AAPD"
processed.mkdir(parents=True, exist_ok=True)
label_order = None
for split, sz in [("train", 53840), ("dev", 1000)]:
    texts, labels, cols = load_aapd_split(str(STAGED), split)
    assert len(texts) == sz and labels.shape == (sz, 54)
    if label_order is None:
        label_order = cols
    else:
        assert cols == label_order
    save_processed(str(processed), split, texts, labels, cols)
    print(f"Prepared {split}: {sz} docs, 54 labels (Test strictly quarantined)", flush=True)

label_sha = hashlib.sha256(json.dumps(label_order).encode()).hexdigest()
assert label_sha == EXPECTED_LABEL_SHA, "Label order SHA mismatch"

defs_path = WORK / "Label_Definition/arxiv_labels_definitions.json"
label_texts = build_all_label_texts(label_order, str(defs_path), dataset="aapd")
assert len(label_texts) == 54
(processed / "label_texts.json").write_text(json.dumps(label_texts, indent=2, ensure_ascii=False), encoding="utf-8")
assert not list(processed.glob("test_*")), "Quarantine violation: test files detected!"

# Verify Frozen Validation IDs
val_ids_file = WORK / "experiments/RUN_0/run0_val_sample_ids.json"
assert val_ids_file.is_file(), "Missing frozen validation sample IDs"
frozen_ids = json.loads(val_ids_file.read_text())
assert len(frozen_ids) == 200 and len(set(frozen_ids)) == 200

# Prepare Runtime Config
orig_cfg = WORK / "configs/final/aapd_baseline.yaml"
cfg_text = orig_cfg.read_text(encoding="utf-8")
runtime_cfg = WORK / "configs/final/aapd_replay_runtime.yaml"
runtime_cfg.write_text(cfg_text.replace("bert_model: bert-base-uncased", f"bert_model: {WORK/'google_bert_bert_base_uncased_official'}"), encoding="utf-8")

print("\\n=== DATA & CONFIG GATES PASSED: Train+Dev Staged, Test Quarantined ===", flush=True)
""".replace("@BUNDLE_SHA256@", bundle_sha256)
    cells.append(cell(cell2_code))

    # ── Cell 3: Fail-Closed Check & Run Screening CLI ────────────────────────
    cell3_code = """\
# Cell 3: Execute Controlled 300-Step Screening (L2 vs L2_NO_SA)
import os, sys, subprocess, time
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

cmd = [
    sys.executable, "-u", str(WORK / "run_baseline.py"), "run0",
    "--config", str(WORK / "configs/final/aapd_replay_runtime.yaml"),
    "--run_id", RUN_ID,
    "--steps", str(N_STEPS),
    "--candidates", CANDIDATES,
]

print("=== STARTING SCREENING REPLAY CLI ===", flush=True)
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
    raise RuntimeError(f"Screening CLI failed with exit code {ret_code}")
""".replace("@RUN_ID@", RUN_ID).replace("@CANDIDATES@", CANDIDATES).replace("@N_STEPS@", str(N_STEPS))
    cells.append(cell(cell3_code))

    # ── Cell 4: Audit Diagnostics & Criteria ─────────────────────────────────
    cell4_code = """\
# Cell 4: Verify Full Artifacts, V1-V5 Criteria, and Checkpoint Existence
import os, sys, json, hashlib
from pathlib import Path
import numpy as np

WORK = Path("/kaggle/working")
RUN_ID = "@RUN_ID@"
EXP_DIR = WORK / "experiments" / RUN_ID

# Verify Checkpoint Existence
for cand in ["L2", "L2_NO_SA"]:
    c_dir = EXP_DIR / f"variant_{cand}"
    for f in ["checkpoints/step_150.pt", "checkpoints/step_300.pt", "step0_diagnostics.json", "step300_diagnostics.json", "train.log", "val_sample_predictions.json"]:
        p = c_dir / f
        assert p.is_file(), f"Missing required file: {p}"
        print(f"Verified {cand} {f}: {p.stat().st_size} bytes")

print("All required experiment files and checkpoints present.", flush=True)

# Criteria Audit
print("\\n" + "=" * 65)
print("CRITERIA EVALUATION (V1–V5)")
print("=" * 65)
results = {}
for cand in ["L2", "L2_NO_SA"]:
    s0 = json.load(open(EXP_DIR / f"variant_{cand}/step0_diagnostics.json"))
    s300 = json.load(open(EXP_DIR / f"variant_{cand}/step300_diagnostics.json"))
    losses = [float(l.split("loss=")[1]) for l in (EXP_DIR / f"variant_{cand}/train.log").read_text().splitlines() if l.startswith("step=")]
    
    nan_inf = s300.get("nan_inf_count", 0)
    V1 = (nan_inf == 0)
    mean_first50 = sum(losses[:50]) / max(len(losses[:50]), 1)
    mean_last50 = sum(losses[-50:]) / max(len(losses[-50:]), 1)
    V2 = (mean_last50 < mean_first50)
    alpha_one_hot = float(s300.get("alpha_one_hot_frac", 1.0))
    V3 = (alpha_one_hot < 0.9)
    logit_stds = s300.get("per_label_logit_std", [])
    v4_labels = sum(float(x) > 0.01 for x in logit_stds)
    V4 = (v4_labels >= 28)
    cand_hits = int(s300.get("p_at_1_hits", 0))
    prior_hits = int(s300.get("prior_p_at_1_hits", 63))
    V5 = (cand_hits > prior_hits)
    viable = V1 and V2 and V3 and V4 and V5
    
    results[cand] = {
        "V1": V1, "V2": V2, "V3": V3, "V4": V4, "V5": V5, "viable": viable,
        "p_at_1_hits": cand_hits, "prior_p_at_1_hits": prior_hits,
        "v4_labels_passing": v4_labels, "alpha_one_hot_frac": alpha_one_hot,
        "mean_loss_last50": mean_last50,
    }
    print(f"[{cand}] V1: {V1} | V2: {V2} | V3: {V3} | V4: {V4} ({v4_labels}/54) | V5: {V5} ({cand_hits}/200 vs prior {prior_hits}) | Viable: {viable}")

# Ranking Diversity Analysis
diversity_results = {}
for cand in ["L2", "L2_NO_SA"]:
    records = json.load(open(EXP_DIR / f"variant_{cand}/val_sample_predictions.json"))
    scores = np.array([r["scores"] for r in records], dtype=np.float32)
    top1 = np.argmax(scores, axis=1)
    distinct_top1 = len(set(top1.tolist()))
    top3_tuples = [tuple(row) for row in np.argsort(-scores, axis=1)[:, :3].tolist()]
    distinct_top3 = len(set(top3_tuples))
    counts = np.bincount(top1, minlength=54)
    dominant_idx = int(np.argmax(counts))
    dominant_cnt = int(np.max(counts))
    diversity_results[cand] = {
        "distinct_top1_labels": distinct_top1,
        "distinct_top3_rankings": distinct_top3,
        "dominant_top1_label_index": dominant_idx,
        "dominant_top1_count": dominant_cnt,
        "dominant_top1_fraction": dominant_cnt / 200.0,
    }
    print(f"[{cand}] Distinct Top-1: {distinct_top1} | Distinct Top-3: {distinct_top3} | Dominant Top-1 ({dominant_idx}): {dominant_cnt}/200")
""".replace("@RUN_ID@", RUN_ID)
    cells.append(cell(cell4_code))

    # ── Cell 5: Export & Verify Model-Only Checkpoints ────────────────────────
    cell5_code = """\
# Cell 5: Export Lightweight Model-Only Checkpoints and Strict Reload Verification
import os, sys, json, hashlib
from pathlib import Path
import torch
from transformers import BertModel

WORK = Path("/kaggle/working")
RUN_ID = "@RUN_ID@"
EXP_DIR = WORK / "experiments" / RUN_ID
BERT_DIR = WORK / "google_bert_bert_base_uncased_official"

sys.path.insert(0, str(WORK / "src"))
from lhrm_baseline.model import LHRMModel
from scripts.probe_csam_trained_checkpoint import export_model_only_checkpoint, sha256_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=== EXPORTING MODEL-ONLY CHECKPOINTS ===", flush=True)

bert = BertModel.from_pretrained(str(BERT_DIR), local_files_only=True)
manifest_entries = {}

for cand in ["L2", "L2_NO_SA"]:
    full_ckpt_p = EXP_DIR / f"variant_{cand}/checkpoints/step_300.pt"
    export_name = f"{cand}_STEP300_MODEL_ONLY.pt"
    export_p = WORK / export_name
    
    meta = export_model_only_checkpoint(
        full_ckpt_path=full_ckpt_p,
        out_path=export_p,
        bert_model=bert,
        device=device,
    )
    
    # Verify exact logit agreement between full checkpoint and exported model-only file
    full_ckpt = torch.load(full_ckpt_p, map_location="cpu")
    model_full = LHRMModel(bert, 54, variant=cand).to(device)
    model_full.load_state_dict(full_ckpt["model_state"], strict=True)
    model_full.eval()
    
    model_exp = LHRMModel(bert, 54, variant=cand).to(device)
    model_exp.load_state_dict(torch.load(export_p, map_location="cpu")["model_state"], strict=True)
    model_exp.eval()
    
    # Dummy test batch
    dummy_doc_ids = torch.randint(0, 1000, (2, 64), device=device)
    dummy_doc_mask = torch.ones((2, 64), device=device)
    dummy_lbl_ids = torch.randint(0, 1000, (54, 32), device=device)
    dummy_lbl_mask = torch.ones((54, 32), device=device)
    
    with torch.no_grad():
        out_full, _ = model_full(dummy_doc_ids, dummy_doc_mask, dummy_lbl_ids, dummy_lbl_mask)
        out_exp, _ = model_exp(dummy_doc_ids, dummy_doc_mask, dummy_lbl_ids, dummy_lbl_mask)
    
    max_logit_diff = (out_full - out_exp).abs().max().item()
    assert max_logit_diff < 1e-5, f"Logit mismatch between full and exported checkpoint for {cand}: {max_logit_diff}"
    print(f"[{cand}] Exported: {export_name} ({meta['file_size_mb']} MB) | Logit diff: {max_logit_diff:.2e} | Strict Reload: PASS", flush=True)
    
    manifest_entries[cand] = {
        "candidate": cand,
        "full_checkpoint": {
            "path": str(full_ckpt_p.resolve()),
            "size_mb": round(full_ckpt_p.stat().st_size / (1024 * 1024), 2),
            "sha256": sha256_file(full_ckpt_p),
            "step": full_ckpt["step"],
            "keys": list(full_ckpt.keys()),
        },
        "model_only_checkpoint": meta,
        "logit_agreement_verified": True,
    }

MANIFEST_OUT = WORK / "DR3_TRAINED_CHECKPOINT_MANIFEST.json"
MANIFEST_OUT.write_text(json.dumps(manifest_entries, indent=2), encoding="utf-8")
print(f"Saved Checkpoint Manifest: {MANIFEST_OUT}", flush=True)
""".replace("@RUN_ID@", RUN_ID)
    cells.append(cell(cell5_code))

    # ── Cell 6: Authentic Trained-State Gradient Micro-Probe ─────────────────
    cell6_code = """\
# Cell 6: Authentic Trained-Checkpoint Gradient Micro-Probe Execution
import os, sys, json
from pathlib import Path
import torch

WORK = Path("/kaggle/working")
RUN_ID = "@RUN_ID@"
EXP_DIR = WORK / "experiments" / RUN_ID
BERT_DIR = WORK / "google_bert_bert_base_uncased_official"

sys.path.insert(0, str(WORK / "src"))
from scripts.probe_csam_trained_checkpoint import run_trained_checkpoint_diagnostic_suite

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=== RUNNING AUTHENTIC TRAINED-CHECKPOINT GRADIENT MICRO-PROBE ===", flush=True)

probe_results = run_trained_checkpoint_diagnostic_suite(
    checkpoint_dir=EXP_DIR,
    bert_path=BERT_DIR,
    data_dir=WORK / "data/processed/AAPD",
    label_defs_path=WORK / "Label_Definition/arxiv_labels_definitions.json",
    batch_size=8,
    device=device,
)

PROBE_OUT = WORK / "DR3_TRAINED_GRADIENT_PROBE_RESULTS.json"
PROBE_OUT.write_text(json.dumps(probe_results, indent=2), encoding="utf-8")

print("\\n" + "=" * 65)
print("TRAINED GRADIENT MICRO-PROBE RESULTS (BATCH SIZE = 8)")
print("=" * 65)
for cand in ["L2", "L2_NO_SA"]:
    diag = probe_results["diagnostic_results"][cand]
    fg = diag["focal_gradients"]
    bg = diag["bce_gradients"]
    c = diag["components"]
    att = diag["attention"]
    print(f"\\n--- Variant {cand} ---")
    print(f"  Focal Loss: {diag['losses']['focal_loss_value']:.4f} | BCE Loss: {diag['losses']['bce_loss_value']:.4f}")
    print(f"  Max Recon Error: {diag['max_algebraic_reconstruction_error']:.2e}")
    print(f"  Focal ||grad_h_T||: {fg['h_T_norm']:.4e} | ||grad_h_L||: {fg['h_L_norm']:.4e}")
    print(f"  Focal ||grad_W||:   {fg['W_norm']:.4e} | ||grad_b||:   {fg['b_norm']:.4e}")
    print(f"  BCE   ||grad_h_T||: {bg['h_T_norm']:.4e} | ||grad_W||:   {bg['W_norm']:.4e}")
    print(f"  Max alpha: {att['max_alpha_mean']:.4f} | Entropy: {att['alpha_entropy_mean']:.4f} / {att['uniform_entropy']:.4f}")
    print(f"  Doc Term Abs Mean: {c['doc_term_abs_mean']:.4e} (across docs std: {c['doc_term_std_across_docs']:.4e})")
    if c['sa_to_doc_ratio'] > 0:
        print(f"  SA Term Abs Mean:  {c['sa_term_abs_mean']:.4e} (SA/Doc ratio: {c['sa_to_doc_ratio']:.2f}x)")
print(f"\\nSaved Probe Results to {PROBE_OUT}", flush=True)
""".replace("@RUN_ID@", RUN_ID)
    cells.append(cell(cell6_code))

    # ── Cell 7: Evidence Packaging & Checksums ───────────────────────────────
    cell7_code = """\
# Cell 7: Package Lightweight Evidence ZIP and Master Checksums
import os, sys, json, zipfile, hashlib, time
from pathlib import Path

WORK = Path("/kaggle/working")
RUN_ID = "@RUN_ID@"
EXP_DIR = WORK / "experiments" / RUN_ID

def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

# 1. Package lightweight evidence ZIP (excluding .pt files)
EVIDENCE_ZIP = WORK / "DR3_REPLAY_EVIDENCE.zip"
written_files = []
with zipfile.ZipFile(EVIDENCE_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for path in sorted(EXP_DIR.rglob("*")):
        if path.is_file() and path.suffix.lower() not in (".pt", ".pth", ".ckpt", ".bin", ".safetensors"):
            rel = f"{RUN_ID}/" + path.relative_to(EXP_DIR).as_posix()
            z.write(path, rel)
            written_files.append(rel)
    # Include manifests, configs, and probe results
    for extra in ["DR3_TRAINED_GRADIENT_PROBE_RESULTS.json", "DR3_TRAINED_CHECKPOINT_MANIFEST.json", "OFFICIAL_BERT_SOURCE.json"]:
        if (WORK / extra).is_file():
            z.write(WORK / extra, extra)
            written_files.append(extra)
    if (WORK / "configs/final/aapd_replay_runtime.yaml").is_file():
        z.write(WORK / "configs/final/aapd_replay_runtime.yaml", "aapd_replay_runtime.yaml")
        written_files.append("aapd_replay_runtime.yaml")

# 2. Master Deliverables Checksum Manifest
DELIVERABLES = [
    "L2_STEP300_MODEL_ONLY.pt",
    "L2_NO_SA_STEP300_MODEL_ONLY.pt",
    "DR3_TRAINED_GRADIENT_PROBE_RESULTS.json",
    "DR3_TRAINED_CHECKPOINT_MANIFEST.json",
    "DR3_REPLAY_EVIDENCE.zip",
]

lines = []
for name in DELIVERABLES:
    p = WORK / name
    assert p.is_file(), f"Missing required deliverable: {name}"
    sha = sha_file(p)
    size_mb = p.stat().st_size / (1024 * 1024)
    lines.append(f"{sha}  {name}")
    print(f"Deliverable: {name:<40} | {size_mb:8.2f} MB | {sha}")

SUMS_FILE = WORK / "DR3_REPLAY_SHA256SUMS.txt"
SUMS_FILE.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
print(f"Master Checksums written to {SUMS_FILE}", flush=True)
""".replace("@RUN_ID@", RUN_ID)
    cells.append(cell(cell7_code))

    # ── Cell 8: Final PASS/FAIL Manifest ─────────────────────────────────────
    cell8_code = """\
# Cell 8: Final Verification Manifest
import os, sys, json
from pathlib import Path

WORK = Path("/kaggle/working")
DELIVERABLES = [
    "L2_STEP300_MODEL_ONLY.pt",
    "L2_NO_SA_STEP300_MODEL_ONLY.pt",
    "DR3_TRAINED_GRADIENT_PROBE_RESULTS.json",
    "DR3_TRAINED_CHECKPOINT_MANIFEST.json",
    "DR3_REPLAY_EVIDENCE.zip",
    "DR3_REPLAY_SHA256SUMS.txt",
]

print("=" * 65)
print("FINAL REPLAY DELIVERABLE MANIFEST (PASS/FAIL)")
print("=" * 65)

all_passed = True
for name in DELIVERABLES:
    p = WORK / name
    exists = p.is_file()
    size_mb = p.stat().st_size / (1024 * 1024) if exists else 0.0
    status = "PASS" if exists else "FAIL"
    print(f"[{status}] {name:<42} ({size_mb:8.2f} MB)")
    if not exists:
        all_passed = False

if all_passed:
    print("\\nALL REPLAY & DIAGNOSTIC GATES PASSED.")
    print("Download the deliverables from /kaggle/working/ and preserve via Kaggle Saved Version.")
else:
    raise RuntimeError("Final verification failed: one or more deliverables missing.")
"""
    cells.append(cell(cell8_code))

    nb = {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }
    return nb


def main():
    parser = argparse.ArgumentParser(description="Generate Kaggle notebook and bundle for DR3 Replay")
    parser.add_argument("--build_bundle", action="store_true", help="Build fresh GPU bundle and compute SHA-256")
    parser.add_argument("--bundle_name", default=BUNDLE_NAME)
    args = parser.parse_args()

    bundle_name = args.bundle_name
    bundle_path = os.path.join(ROOT, "bundles", bundle_name)

    # Step 1: Build bundle if requested or missing
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

    # Step 2: Assemble notebook
    nb = build_notebook(bundle_name, bundle_sha)
    os.makedirs(os.path.dirname(OUT_NB), exist_ok=True)
    with open(OUT_NB, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Notebook written: {OUT_NB}")
    print(f"  Cells: {len(nb['cells'])}")

    # Verify JSON
    with open(OUT_NB, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert len(loaded["cells"]) == len(nb["cells"])
    print("Notebook JSON verification: PASS")


if __name__ == "__main__":
    main()
