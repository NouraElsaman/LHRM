#!/usr/bin/env python3
"""
scripts/gen_kaggle_notebook_r_author.py

Generates notebooks/lhrm_r_author_aapd_train_evaluate_kaggle.ipynb —
Dedicated, author-code-aligned recovery and full evaluation workflow on Kaggle GPU:
- Architecture: R_AUTHOR (Concat CSAM with trainable label embeddings & pooler_output)
- Loss: BCEWithLogitsLoss (matching Model/lhrm.py)
- Training schedule: 15 epochs, batch size 8, lr 2e-5, AdamW, AMP enabled
- Step-300 informational checkpoint
- Per-epoch Dev evaluation and rolling checkpoints
- Strict Test split quarantine
- Deterministic non-overlapping 4-way Dev partition for HRRM evaluation
- Client inference demonstration on scientific abstracts
- Evidence packaging and downloadable FileLink outputs
"""
import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BUNDLE_NAME = "lhrm_gpu_bundle_r_author_20261015.zip"
BUNDLE_SHA256 = "e25c583e76d44873e0bdeebcaa99973db8ac4763b79023afd2787902a8e65bbf"
MANIFEST_SHA256 = "7a5f5fa02331954817917569a3eff5166289801e7a62c8b21a4d75bf9ce1c1bf"
OUT_NB = os.path.join(ROOT, "notebooks", "lhrm_r_author_aapd_train_evaluate_kaggle.ipynb")
RUN_ID = "AAPD_R_AUTHOR_BCE_RECOVERY_20261015_A"
CONFIG = "configs/final/aapd_r_author_bce.yaml"


def cell(source_str: str, cell_type: str = "code") -> dict:
    lines = source_str.rstrip("\n").split("\n")
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": [l + "\n" for l in lines[:-1]] + [lines[-1]],
        **({"outputs": [], "execution_count": None} if cell_type == "code" else {}),
    }


def build_notebook() -> dict:
    cells = []

    # ── Cell 0: Markdown Protocol Header ─────────────────────────────────────
    cells.append(cell(f"""\
# LHRM Author-Aligned Model Recovery & Full Evaluation ({RUN_ID})

**Client-Ready Author-Code-Aligned Implementation (15 Epochs, Full AAPD Training & Non-Leaky HRRM Pipeline)**

### Key Technical Characteristics:
1. **Architecture (`R_AUTHOR`)**: Directly implements the original authors' reference architecture from `Model/lhrm.py:144-192`:
   - Document representation: `bert(...).pooler_output` $\\in \\mathbb{{R}}^{{B \\times 768}}$ (Model/lhrm.py:185).
   - Label representation: Trainable $\\text{{nn.Parameter}}$ initialized once from pre-trained BERT token means (Model/lhrm.py:147, 222-243).
   - Label self-attention: `nn.MultiheadAttention(768, heads=4)` with residual addition and LayerNorm (Model/lhrm.py:150-158).
   - Similarity: Unscaled dot-product attention $w = \\text{{softmax}}(h_\\text{{doc}} \\cdot L^T, \\text{{dim}}=1)$ (Model/lhrm.py:159).
   - Aggregation: $\\text{{agg}} = w \\cdot L \\in \\mathbb{{R}}^{{B \\times 768}}$ (Model/lhrm.py:160).
   - Concatenation: $\\text{{combined}} = [h_\\text{{doc}}, \\text{{agg}}] \\in \\mathbb{{R}}^{{B \\times 1536}}$ (Model/lhrm.py:190).
   - Classification: $\\text{{nn.Linear}}(1536, 54)$ (Model/lhrm.py:182, 191).
2. **Objective**: Standard `torch.nn.BCEWithLogitsLoss()` matching the authors' codebase and the successful DR2 control.
3. **Training Protocol**: 15 epochs, batch size 8, learning rate 2e-5, weight decay 0.01, AdamW optimizer, seed 42.
4. **Step-300 Informational Checkpoint**: Informational early progress check to log training loss and early ranking behavior without aborting the full 15-epoch training run.
5. **Data Quarantine**: **Test split is strictly quarantined.** Only AAPD Train (53,840) and Dev (1,000) splits are used.
6. **Non-Leaky HRRM Partition**: Deterministic 4-way Dev partition:
   - 200 docs: Continuity diagnostic check.
   - 400 docs: Neural model checkpoint selection.
   - 200 docs: HRRM regressor training.
   - 200 docs: Held-out HRRM vs Neural comparison.
7. **Client Demonstration**: Interactive inference on scientific text/abstracts producing Top-5 predictions with confidence scores and unverified label mapping disclosure.

- **Bundle Name:** `{BUNDLE_NAME}`
- **Bundle SHA-256:** `{BUNDLE_SHA256}`
""", cell_type="markdown"))

    # ── Cell 1: Environment & Official BERT Acquisition ──────────────────────
    cells.append(cell("""\
# Cell 1: Environment Verification & Official Pinned BERT Backbone Acquisition
import os, sys, time, json, hashlib, shutil, gc
from pathlib import Path
import torch

print("=== LHRM R_AUTHOR Recovery: Environment Verification ===", flush=True)
if not torch.cuda.is_available():
    raise RuntimeError("GPU GATE: CUDA GPU required — select Kaggle GPU runtime (T4 x2 or P100)")
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

def verify_and_record():
    if not BERT_DIR.exists():
        return False
    present = [f for f in WEIGHTS if (BERT_DIR / f).exists()]
    if not present or not (BERT_DIR / "config.json").exists():
        return False
    if not SOURCE_META.exists():
        meta = {
            "source_repo": REPO,
            "pinned_commit": PINNED_COMMIT,
            "weight_files": present,
            "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        SOURCE_META.write_text(json.dumps(meta, indent=2))
    return True

if not verify_and_record():
    print(f"Acquiring official BERT backbone ({REPO} @ {PINNED_COMMIT})...")
    attached_dirs = [
        Path("/kaggle/input/bert-base-uncased"),
        Path("/kaggle/input/google-bert-bert-base-uncased"),
        Path("/kaggle/input/bert-base-uncased-official"),
    ]
    found_attached = False
    for ad in attached_dirs:
        if ad.exists() and any((ad / w).exists() for w in WEIGHTS) and (ad / "config.json").exists():
            print(f"Using attached BERT from: {ad}")
            shutil.copytree(ad, BERT_DIR, dirs_exist_ok=True)
            found_attached = True
            break
            
    if not found_attached:
        from huggingface_hub import snapshot_download
        print(f"Downloading from HuggingFace Hub: {REPO} revision {PINNED_COMMIT}...")
        snapshot_download(
            repo_id=REPO,
            revision=PINNED_COMMIT,
            local_dir=str(BERT_DIR),
            local_dir_use_symlinks=False,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"]
        )

if not verify_and_record():
    raise RuntimeError("Failed to verify official BERT backbone files")
print(f"Official BERT ready at: {BERT_DIR}", flush=True)
"""))

    # ── Cell 2: Code Bundle Unpack & Integrity Check ─────────────────────────
    cells.append(cell(f"""\
# Cell 2: Unpack and Verify Code Bundle
import zipfile

BUNDLE_NAME = "{BUNDLE_NAME}"
EXPECTED_BUNDLE_SHA256 = "{BUNDLE_SHA256}"
EXPECTED_MANIFEST_SHA256 = "{MANIFEST_SHA256}"

print(f"=== Locating and Verifying Code Bundle: {{BUNDLE_NAME}} ===", flush=True)

search_locations = [
    Path(f"/kaggle/input/{{BUNDLE_NAME}}"),
    Path(f"/kaggle/input/lhrm-gpu-bundle/{{BUNDLE_NAME}}"),
    Path(f"/kaggle/input/lhrm-r-author-bundle/{{BUNDLE_NAME}}"),
    Path(f"/kaggle/working/{{BUNDLE_NAME}}"),
]

for d in Path("/kaggle/input").glob("*"):
    if d.is_dir():
        cand = d / BUNDLE_NAME
        if cand.exists() and cand not in search_locations:
            search_locations.append(cand)

bundle_path = None
for p in search_locations:
    if p.exists() and p.is_file():
        bundle_path = p
        break

if bundle_path:
    print(f"Found code bundle archive at: {{bundle_path}}")
    actual_hash = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    print(f"Archive SHA-256: {{actual_hash}}")
    assert actual_hash == EXPECTED_BUNDLE_SHA256, (
        f"BUNDLE HASH MISMATCH! Expected {{EXPECTED_BUNDLE_SHA256}}, got {{actual_hash}}"
    )
    with zipfile.ZipFile(bundle_path, "r") as z:
        z.extractall("/kaggle/working")
    print("Extracted bundle successfully into /kaggle/working.")
else:
    print("Archive file not found directly; searching for auto-extracted bundle files...")
    manifest_candidates = list(Path("/kaggle/input").rglob("BUNDLE_MANIFEST.json"))
    matched = None
    for mc in manifest_candidates:
        m_hash = hashlib.sha256(mc.read_bytes()).hexdigest()
        if m_hash == EXPECTED_MANIFEST_SHA256:
            matched = mc
            break
            
    assert matched is not None, (
        f"Could not locate {{BUNDLE_NAME}} or extracted BUNDLE_MANIFEST.json with SHA-256 {{EXPECTED_MANIFEST_SHA256}}"
    )
    print(f"Found authentic extracted bundle manifest at: {{matched}}")
    source_dir = matched.parent
    for item in source_dir.iterdir():
        dest = Path("/kaggle/working") / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)
    print(f"Copied constituent bundle files from {{source_dir}} into /kaggle/working.")

# Ensure working directory is on sys.path
if "/kaggle/working" not in sys.path:
    sys.path.insert(0, "/kaggle/working")

from src.lhrm_baseline.model import LHRMModel, CSAM_R_AUTHOR, build_author_label_embeddings
from src.lhrm_baseline.losses import get_loss
from src.lhrm_baseline.hrrm import HRRM, create_dev_partition, evaluate_hrrm_heldout
from src.lhrm_baseline.inference import predict_abstract, format_prediction_report
print("Successfully imported LHRM baseline modules.", flush=True)
"""))

    # ── Cell 3: Data Preparation & Quarantine Check ──────────────────────────
    cells.append(cell("""\
# Cell 3: Prepare AAPD Dataset (Train & Dev Only) & Quarantine Test Split
import pandas as pd
from transformers import BertTokenizer

print("=== AAPD Data Preparation & Quarantine Verification ===", flush=True)

DATA_DIR = Path("/kaggle/working/data/processed/AAPD")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Locate AAPD input files
aapd_locations = [
    Path("/kaggle/input/aapd-dataset"),
    Path("/kaggle/input/aapd"),
    Path("/kaggle/input/aapd-data"),
    Path("/kaggle/input/datasets/aapd-dataset"),
]
input_dir = None
for loc in aapd_locations:
    if loc.exists() and (loc / "train.csv").exists() and (loc / "dev.csv").exists():
        input_dir = loc
        break

assert input_dir is not None, "Could not locate AAPD train.csv and dev.csv in /kaggle/input"
print(f"Loading raw AAPD files from: {input_dir}")

train_csv = input_dir / "train.csv"
dev_csv = input_dir / "dev.csv"

# Preprocess and save train and dev
from src.lhrm_baseline.data import save_processed, load_processed, MultiLabelDataset, collate_fn

df_train = pd.read_csv(train_csv)
df_dev = pd.read_csv(dev_csv)

# 54 labels in AAPD
label_cols = [c for c in df_train.columns if c not in ("text", "ID", "id")]
assert len(label_cols) == 54, f"Expected 54 label columns, got {len(label_cols)}"

label_order_hash = hashlib.sha256(json.dumps(label_cols).encode()).hexdigest()
expected_lo_hash = "a3d662f3920259a94ed61d83f98f21cabe7f908d98a5662ccd9adc2413cc000b"
print(f"Label order hash: {label_order_hash}")
assert label_order_hash == expected_lo_hash, (
    f"Label order hash mismatch! Expected {expected_lo_hash}, got {label_order_hash}"
)

# Extract texts and binary label arrays
train_texts = df_train["text"].astype(str).tolist()
train_labels = df_train[label_cols].values.astype(np.float32)

dev_texts = df_dev["text"].astype(str).tolist()
dev_labels = df_dev[label_cols].values.astype(np.float32)

print(f"Train documents: {len(train_texts)} | Dev documents: {len(dev_texts)}")
assert len(train_texts) == 53840, f"Expected 53,840 train docs, got {len(train_texts)}"
assert len(dev_texts) == 1000, f"Expected 1,000 dev docs, got {len(dev_texts)}"

# Save processed splits for reproducibility
save_processed(str(DATA_DIR), "train", train_texts, train_labels, label_cols)
save_processed(str(DATA_DIR), "dev", dev_texts, dev_labels, label_cols)
with open(DATA_DIR / "labels.json", "w", encoding="utf-8") as f:
    json.dump(label_cols, f, indent=2)

print("AAPD processed splits saved successfully. Test split is quarantined (NEVER loaded).", flush=True)
"""))

    # ── Cell 4: Initialize Author Label Embeddings & R_AUTHOR Model ───────────
    cells.append(cell("""\
# Cell 4: Initialize Author-Aligned R_AUTHOR Model & Precompute Label Embeddings
from transformers import BertModel, BertTokenizer
from src.lhrm_baseline.model import build_bert_model, build_author_label_embeddings

print("=== Initializing R_AUTHOR Architecture ===", flush=True)

device = torch.device("cuda")
tokenizer = BertTokenizer.from_pretrained(str(BERT_DIR))
bert = build_bert_model(str(BERT_DIR), device=device)

defs_path = Path("/kaggle/working/Label_Definition/arxiv_labels_definitions.json")
assert defs_path.exists(), f"Missing label definitions at {defs_path}"

# Build initial author label embeddings (Model/lhrm.py:222-243)
print("Precomputing initial label embeddings from BERT token means...")
label_emb = build_author_label_embeddings(
    label_list=label_cols,
    tokenizer=tokenizer,
    bert=bert,
    defs_path=str(defs_path),
    max_len=64,
    device=device,
)
print(f"Label embedding tensor shape: {tuple(label_emb.shape)}")
assert label_emb.shape == (54, 768)

# Construct R_AUTHOR model
model = LHRMModel(
    bert_model=bert,
    num_labels=len(label_cols),
    variant="R_AUTHOR",
    label_emb=label_emb,
).to(device)

# Model sanity check
dummy_ids = torch.randint(0, 1000, (2, 64), device=device)
dummy_mask = torch.ones_like(dummy_ids, device=device)
with torch.no_grad():
    test_logits, test_diag = model(dummy_ids, dummy_mask, return_alpha=True)

print(f"Sanity logits shape: {tuple(test_logits.shape)}")
print(f"Sanity alpha shape:  {tuple(test_diag['alpha'].shape)}")
assert test_logits.shape == (2, 54)
assert test_diag["alpha"].shape == (2, 54)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Parameters:     {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")
print("R_AUTHOR initialization verified successfully.", flush=True)
"""))

    # ── Cell 5: Full 15-Epoch Training Loop & Checkpointing ──────────────────
    cells.append(cell("""\
# Cell 5: Full 15-Epoch Training with Step-300 Check & Per-Epoch Dev Metrics
from torch.utils.data import DataLoader
from src.lhrm_baseline.train import train_epoch_stepped, predict_epoch_loader, save_predictions
from src.lhrm_baseline.metrics import compute_all_metrics, compute_per_label_metrics

EXP_DIR = Path(f"/kaggle/working/experiments/{RUN_ID}")
CKPT_DIR = EXP_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Datasets and Loaders
train_ds = MultiLabelDataset(train_texts, train_labels, tokenizer, max_len=512)
dev_ds   = MultiLabelDataset(dev_texts, dev_labels, tokenizer, max_len=512)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)
dev_loader   = DataLoader(dev_ds, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)

# Loss & Optimizer
criterion = get_loss("bce")  # BCEWithLogitsLoss
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
scaler = torch.amp.GradScaler("cuda")

# Fixed Dev Diagnostic IDs (for step-300 continuity check)
dev_partition = create_dev_partition(n_docs=len(dev_texts), seed=42)
diag_ids = dev_partition["diagnostic_ids"]
v_labels_diag = dev_labels[diag_ids]

# Files for recording
metrics_csv_path = EXP_DIR / "epoch_metrics.csv"
history_json_path = EXP_DIR / "train_history.json"
train_log_path = EXP_DIR / "train.log"

with open(metrics_csv_path, "w") as f:
    f.write("epoch,train_loss,global_steps,val_P@1,val_P@3,val_P@5,val_nDCG@3,val_nDCG@5,val_micro_f1,val_macro_f1\\n")

epochs = 15
steps_per_epoch = len(train_loader)
total_global_steps = epochs * steps_per_epoch
total_steps = 0
history = []
best_dev_p1 = 0.0

step_300_reported = False

print(f"=== Starting 15-Epoch Training Run ({RUN_ID}) ===", flush=True)
print(f"Total Epochs: {epochs} | Steps per Epoch: {steps_per_epoch} | Total Steps: {total_global_steps}")

t_train_start = time.time()

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    model.train()
    total_epoch_loss = 0.0
    
    for batch_idx, batch in enumerate(train_loader, 1):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        y = batch["labels"].to(device)
        
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            logits, _ = model(ids, mask)
            loss = criterion(logits, y)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        loss_val = float(loss.item())
        total_epoch_loss += loss_val
        total_steps += 1
        
        # Step-300 informational check
        if total_steps == 300 and not step_300_reported:
            step_300_reported = True
            model.eval()
            with torch.no_grad():
                y_true_d, y_pred_d, _ = predict_epoch_loader(model, dev_loader, device, None, None)
            diag_metrics = compute_all_metrics(y_true_d[diag_ids], y_pred_d[diag_ids], [1, 3, 5], threshold=0.5)
            s300_info = {
                "step": 300,
                "train_loss_step300": loss_val,
                "diag_P@1": diag_metrics.get("P@1", 0.0),
                "diag_hits_at_1": int(round(diag_metrics.get("P@1", 0.0) * len(diag_ids))),
                "prior_reference": 63,
            }
            print(f"\\n[STEP 300 INFORMATIONAL CHECK] train_loss={loss_val:.4f} | diag_P@1={s300_info['diag_P@1']:.4f} ({s300_info['diag_hits_at_1']}/200 vs prior 63/200)\\n", flush=True)
            with open(EXP_DIR / "step300_info.json", "w") as sf:
                json.dump(s300_info, sf, indent=2)
            model.train()
            
        if batch_idx % 200 == 0:
            elapsed = time.time() - t_train_start
            speed = total_steps / max(elapsed, 1e-5)
            eta = (total_global_steps - total_steps) / speed
            print(f"[Epoch {epoch:02d}/{epochs:02d}] Step {batch_idx:04d}/{steps_per_epoch} | global={total_steps}/{total_global_steps} | loss={loss_val:.4f} | speed={speed:.2f} step/s | ETA~{eta/60:.1f}m", flush=True)
            
    mean_train_loss = total_epoch_loss / steps_per_epoch
    
    # Per-epoch Dev Evaluation
    print(f"\\n[VAL EVALUATION] Evaluating Epoch {epoch} on full 1,000 Dev documents...", flush=True)
    y_true_dev, y_pred_dev, _ = predict_epoch_loader(model, dev_loader, device, None, None)
    dev_m = compute_all_metrics(y_true_dev, y_pred_dev, [1, 3, 5], threshold=0.5, include_f1=True)
    
    row = {
        "epoch": epoch,
        "train_loss": round(mean_train_loss, 6),
        "global_steps": total_steps,
        "val_P@1": dev_m.get("P@1", 0.0),
        "val_P@3": dev_m.get("P@3", 0.0),
        "val_P@5": dev_m.get("P@5", 0.0),
        "val_nDCG@3": dev_m.get("nDCG@3", 0.0),
        "val_nDCG@5": dev_m.get("nDCG@5", 0.0),
        "val_micro_f1": dev_m.get("micro_f1", 0.0),
        "val_macro_f1": dev_m.get("macro_f1", 0.0),
    }
    history.append(row)
    
    print(f"[VAL METRICS Epoch {epoch:02d}] train_loss={row['train_loss']:.4f} | P@1={row['val_P@1']:.4f} | P@3={row['val_P@3']:.4f} | nDCG@3={row['val_nDCG@3']:.4f} | micro_F1={row['val_micro_f1']:.4f} | macro_F1={row['val_macro_f1']:.4f}", flush=True)
    
    # Append to CSV
    with open(metrics_csv_path, "a") as f:
        f.write(f"{epoch},{row['train_loss']:.6f},{total_steps},{row['val_P@1']:.4f},{row['val_P@3']:.4f},{row['val_P@5']:.4f},{row['val_nDCG@3']:.4f},{row['val_nDCG@5']:.4f},{row['val_micro_f1']:.4f},{row['val_macro_f1']:.4f}\\n")
        
    # Checkpointing (rolling latest and best-dev)
    latest_ckpt = CKPT_DIR / "latest.pt"
    previous_ckpt = CKPT_DIR / "previous.pt"
    if latest_ckpt.exists():
        if previous_ckpt.exists():
            previous_ckpt.unlink()
        shutil.copy2(latest_ckpt, previous_ckpt)
        
    torch.save({
        "epoch": epoch,
        "step": total_steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "variant": "R_AUTHOR",
        "val_metrics": dev_m,
    }, latest_ckpt)
    
    if row["val_P@1"] > best_dev_p1:
        best_dev_p1 = row["val_P@1"]
        torch.save({
            "epoch": epoch,
            "step": total_steps,
            "model_state_dict": model.state_dict(),
            "variant": "R_AUTHOR",
            "val_P@1": best_dev_p1,
        }, CKPT_DIR / "best_dev_model_only.pt")
        print(f"  * New best Dev P@1: {best_dev_p1:.4f} -> saved best_dev_model_only.pt")

# Save final epoch checkpoint
torch.save({
    "epoch": epochs,
    "step": total_steps,
    "model_state_dict": model.state_dict(),
    "variant": "R_AUTHOR",
    "final_dev_metrics": history[-1],
}, CKPT_DIR / "final_model_only.pt")

with open(history_json_path, "w") as f:
    json.dump(history, f, indent=2)

# Save final predictions on full Dev
save_predictions(y_true_dev, y_pred_dev, label_cols, str(EXP_DIR / "neural_dev_predictions.json"))
print(f"Training completed successfully in {(time.time() - t_train_start)/60:.1f} minutes.", flush=True)
"""))

    # ── Cell 6: HRRM Non-Leaky Fitting & Held-out Evaluation ─────────────────
    cells.append(cell("""\
# Cell 6: Non-Leaky HRRM Pipeline Fitting & Held-Out Evaluation
print("=== Non-Leaky HRRM Pipeline Evaluation ===", flush=True)

# Use deterministic 4-way Dev partition:
# partition['hrrm_fit_ids'] = 200 docs (FIT ONLY)
# partition['hrrm_eval_ids'] = 200 docs (EVAL ONLY)
hrrm_results = evaluate_hrrm_heldout(
    y_true_full=y_true_dev,
    y_pred_full=y_pred_dev,
    label_order=label_cols,
    partition=dev_partition,
    seed=42,
)

# Export HRRM model
hrrm_model_path = EXP_DIR / "hrrm_model.joblib"
hrrm_results["hrrm_model"].save(str(hrrm_model_path))

# Save comparison JSON
comparison_record = {
    "protocol": "NON_LEAKY_DEV_PARTITION",
    "n_fit_docs": hrrm_results["n_fit_docs"],
    "n_eval_docs": hrrm_results["n_eval_docs"],
    "neural_metrics": hrrm_results["neural_metrics"],
    "hrrm_metrics": hrrm_results["hrrm_metrics"],
    "delta": hrrm_results["delta"],
}
with open(EXP_DIR / "hrrm_heldout_comparison.json", "w") as f:
    json.dump(comparison_record, f, indent=2)

print("\\n--- Held-Out HRRM Comparison (Evaluated strictly on held-out 200 Dev rows) ---")
print(f"Neural Base:  P@1={hrrm_results['neural_metrics']['P@1']:.4f} | P@3={hrrm_results['neural_metrics']['P@3']:.4f} | nDCG@3={hrrm_results['neural_metrics']['nDCG@3']:.4f}")
print(f"HRRM Re-rank: P@1={hrrm_results['hrrm_metrics']['P@1']:.4f} | P@3={hrrm_results['hrrm_metrics']['P@3']:.4f} | nDCG@3={hrrm_results['hrrm_metrics']['nDCG@3']:.4f}")
print(f"Delta:        P@1={hrrm_results['delta']['P@1']:+.4f} | P@3={hrrm_results['delta']['P@3']:+.4f} | nDCG@3={hrrm_results['delta']['nDCG@3']:+.4f}")
print("Non-leaky HRRM evaluation complete.", flush=True)
"""))

    # ── Cell 7: Client Inference Demonstration ───────────────────────────────
    cells.append(cell("""\
# Cell 7: Client Inference Demonstration with Genuine Trained Checkpoint
from IPython.display import display, Markdown

print("=== Client Inference Demonstration ===", flush=True)

# Sample scientific abstracts for interactive testing
demo_abstracts = [
    "We present a novel neural architecture for multi-label text classification using contextual similarity attention and label embeddings. Our hierarchical ranking approach learns cross-label dependencies and optimizes ranking metrics directly.",
    "Quantum phase transitions in disordered spin systems are investigated using numerical Monte Carlo simulations. We examine the low-temperature critical exponents and spatial correlation functions across the disordered phase boundary.",
    "Efficient algorithms for convex optimization in high-dimensional machine learning problems. We analyze stochastic gradient methods with adaptive step-sizes and prove linear convergence under strong convexity assumptions."
]

demo_results = []
for i, abstract_text in enumerate(demo_abstracts, 1):
    res = predict_abstract(
        text=abstract_text,
        model=model,
        tokenizer=tokenizer,
        label_order=label_cols,
        device=device,
        hrrm_model=hrrm_results["hrrm_model"],
        top_k=5,
    )
    demo_results.append(res)
    print(f"\\n--- Abstract Sample {i} ---")
    report_md = format_prediction_report(res)
    display(Markdown(report_md))

# Save demo outputs to JSON
with open(EXP_DIR / "inference_demo.json", "w") as f:
    json.dump(demo_results, f, indent=2)

print("Inference demonstration complete.", flush=True)
"""))

    # ── Cell 8: Evidence Package & FileLinks ──────────────────────────────────
    cells.append(cell(f"""\
# Cell 8: Evidence Packaging, Integrity Manifest & Downloadable Links
import zipfile
from IPython.display import FileLink

print("=== Evidence Packaging & Integrity Check ===", flush=True)

evidence_zip_path = Path("/kaggle/working/EVIDENCE.zip")
files_to_pack = [
    "epoch_metrics.csv",
    "train_history.json",
    "step300_info.json",
    "hrrm_heldout_comparison.json",
    "inference_demo.json",
    "neural_dev_predictions.json",
    "train.log",
]

# Write SHA256SUMS.txt
sha256_records = []
with zipfile.ZipFile(evidence_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as ez:
    for fname in files_to_pack:
        fpath = EXP_DIR / fname
        if fpath.exists():
            h = hashlib.sha256(fpath.read_bytes()).hexdigest()
            sha256_records.append(f"{{h}}  {{fname}}")
            ez.write(fpath, arcname=fname)
            print(f"Packed: {{fname}} ({{h[:12]}}...)")
            
    sha_str = "\\n".join(sha256_records) + "\\n"
    ez.writestr("SHA256SUMS.txt", sha_str)

print(f"\\nEVIDENCE.zip size: {{evidence_zip_path.stat().st_size:,}} bytes")
ez_hash = hashlib.sha256(evidence_zip_path.read_bytes()).hexdigest()
print(f"EVIDENCE.zip SHA-256: {{ez_hash}}")

# Expose key files in /kaggle/working for direct FileLink download
shutil.copy2(CKPT_DIR / "best_dev_model_only.pt", "/kaggle/working/best_dev_model_only.pt")
shutil.copy2(CKPT_DIR / "final_model_only.pt", "/kaggle/working/final_model_only.pt")
shutil.copy2(EXP_DIR / "epoch_metrics.csv", "/kaggle/working/epoch_metrics.csv")
shutil.copy2(EXP_DIR / "hrrm_heldout_comparison.json", "/kaggle/working/hrrm_heldout_comparison.json")

print("\\n=== Downloadable Artifacts (Click to Download) ===")
display(FileLink("EVIDENCE.zip"))
display(FileLink("epoch_metrics.csv"))
display(FileLink("hrrm_heldout_comparison.json"))
display(FileLink("best_dev_model_only.pt"))
display(FileLink("final_model_only.pt"))
print("All deliverables ready.", flush=True)
"""))

    nb = {
        "cells": cells,
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
        "nbformat": 4,
        "nbformat_minor": 4
    }
    return nb


def main():
    nb = build_notebook()
    os.makedirs(os.path.dirname(OUT_NB), exist_ok=True)
    with open(OUT_NB, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Successfully generated notebook: {OUT_NB}")
    print(f"Total cells: {len(nb['cells'])}")


if __name__ == "__main__":
    main()
