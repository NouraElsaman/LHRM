"""
Local CPU step-0 cross-check (Phase 1F, Section 4).

Computes step-0 diagnostics for both variants (L and L2) on the saved
200-doc val sample IDs, using fp32 on CPU (no GPU required).

Preconditions:
  - bert-base-uncased is cached locally (set BERT_CACHE below or env var)
  - data/processed/AAPD/ processed splits exist
  - experiments/RUN_0/run0_val_sample_ids.json exists (from first Kaggle run)

Output: docs/phase1/run0_local_step0_crosscheck.json

After the corrected Kaggle rerun, the GPU-fp32 step-0 numbers must agree:
  - mean_max_alpha:     ±0.001
  - alpha_one_hot_frac: exact (both fp32)
  - prior_p_at_1_hits:  exact (integer, deterministic)
"""
import json, os, sys, hashlib, time
import numpy as np
import torch

try:
    _SCRIPT_FILE = __file__
except NameError:
    _SCRIPT_FILE = os.path.join(os.getcwd(), "_crosscheck.py")

# scripts/ is one level below the project root
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(_SCRIPT_FILE))
ROOT = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, os.path.join(ROOT, "src"))

# ── Config ────────────────────────────────────────────────────────────────────
BERT_CACHE = os.environ.get("BERT_CACHE", "bert-base-uncased")
DATA_DIR   = os.path.join(ROOT, "data", "processed", "AAPD")
VAL_IDS_PATH = os.path.join(ROOT, "experiments", "RUN_0", "run0_val_sample_ids.json")
DEFS_PATH    = os.path.join(ROOT, "Label_Definition", "arxiv_labels_definitions.json")
FREEZE_PATH  = os.path.join(ROOT, "configs", "final", "CONFIG_FREEZE.json")
OUT_PATH     = os.path.join(ROOT, "docs", "phase1", "run0_local_step0_crosscheck.json")

SEED     = 42
MAX_LEN  = 512
LABEL_MAX_LEN = 64
BATCH_SIZE = 4  # small for CPU
VARIANTS = ["L", "L2", "L2_NO_SA"]

# ── Imports ───────────────────────────────────────────────────────────────────
from transformers import BertTokenizer
from lhrm_baseline.data import MultiLabelDataset, collate_fn
from lhrm_baseline.model import LHRMModel, build_bert_model
from lhrm_baseline.labels import build_all_label_texts
from lhrm_baseline.data import tokenize_labels
from lhrm_baseline.diagnostics import (
    compute_diagnostics, frequency_prior_hits_at_1, DiagnosticError
)
from torch.utils.data import DataLoader


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def main():
    device = torch.device("cpu")
    print("Device: CPU (local cross-check)")
    print("BERT:", BERT_CACHE)

    # Load config freeze for label hash verification
    with open(FREEZE_PATH) as f:
        freeze = json.load(f)
    expected_label_hash = freeze.get("label_order_sha256")

    # Load processed data
    with open(os.path.join(DATA_DIR, "labels.json")) as f:
        label_order = json.load(f)
    with open(os.path.join(DATA_DIR, "dev_texts.json")) as f:
        dev_texts = json.load(f)
    dev_labels = np.load(os.path.join(DATA_DIR, "dev_labels.npy"))
    train_labels = np.load(os.path.join(DATA_DIR, "train_labels.npy"))

    # Verify label hash
    actual_hash = hashlib.sha256(json.dumps(label_order).encode()).hexdigest()
    if expected_label_hash:
        assert actual_hash == expected_label_hash, (
            f"Label hash mismatch: {actual_hash} != {expected_label_hash}"
        )
        print(f"Label hash: OK ({actual_hash[:12]}...)")
    else:
        print(f"Label hash (no freeze reference): {actual_hash[:12]}...")

    # Load the exact 200 val doc IDs from the first Kaggle run
    if not os.path.exists(VAL_IDS_PATH):
        raise FileNotFoundError(
            f"Val sample IDs not found: {VAL_IDS_PATH}\n"
            "Run the Kaggle notebook first and import artifacts."
        )
    with open(VAL_IDS_PATH) as f:
        val_sample_ids = json.load(f)
    print(f"Val sample: {len(val_sample_ids)} doc IDs loaded from first Kaggle run")
    assert len(val_sample_ids) == 200, f"Expected 200 val IDs; got {len(val_sample_ids)}"

    v_texts  = [dev_texts[i] for i in val_sample_ids]
    v_labels = dev_labels[val_sample_ids]

    # Tokenizer
    try:
        tokenizer = BertTokenizer.from_pretrained(BERT_CACHE, local_files_only=True)
        print("Tokenizer: loaded from local cache")
    except Exception:
        tokenizer = BertTokenizer.from_pretrained(BERT_CACHE)
        print("Tokenizer: downloaded")

    # Label encodings
    label_texts = build_all_label_texts(label_order, DEFS_PATH, dataset="aapd")
    label_ids, label_mask = tokenize_labels(
        label_texts, tokenizer, max_len=LABEL_MAX_LEN, device=device
    )
    print(f"Label tokens: {label_ids.shape}")

    # Prior (integer, seeded tie-break)
    prior_hits = frequency_prior_hits_at_1(train_labels, v_labels, seed=SEED)
    print(f"Prior P@1 hits: {prior_hits}/200 = {prior_hits/200:.4f}")

    # Val dataset
    val_ds = MultiLabelDataset(v_texts, v_labels, tokenizer, MAX_LEN)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    results = {
        "description": "Local CPU step-0 cross-check (Phase 1F, Section 4)",
        "device": "cpu",
        "bert_cache": BERT_CACHE,
        "seed": SEED,
        "n_val_docs": len(val_sample_ids),
        "val_sample_ids_path": VAL_IDS_PATH,
        "label_order_sha256": actual_hash,
        "prior_p_at_1_hits": prior_hits,
        "prior_p_at_1": prior_hits / 200,
        "tolerance": {
            "mean_max_alpha":     0.001,
            "alpha_one_hot_frac": 0.0,   # exact
            "prior_p_at_1_hits":  0,     # exact (integer)
        },
        "variants": {}
    }

    for variant in VARIANTS:
        print(f"\n{'='*40}\nVariant {variant} — step-0 forward pass (CPU fp32)\n{'='*40}")
        set_seed(SEED)
        try:
            bert = build_bert_model(BERT_CACHE, device)
        except Exception:
            bert = build_bert_model("bert-base-uncased", device)
        model = LHRMModel(bert, len(label_order), variant=variant).to(device)
        model.eval()

        t0 = time.time()
        try:
            diag = compute_diagnostics(
                model=model,
                loader=val_loader,
                label_input_ids=label_ids,
                label_attention_mask=label_mask,
                val_labels=v_labels,
                device=device,
                train_labels=train_labels,
                n_labels=len(label_order),
                seed=SEED,
                amp_for_training=True,  # informational only; no effect here
            )
            elapsed = time.time() - t0
            diag["elapsed_seconds"] = elapsed
            diag["status"] = "OK"
            print(f"  mean_max_alpha:    {diag['mean_max_alpha']:.6f}")
            print(f"  alpha_one_hot_frac:{diag['alpha_one_hot_frac']:.6f}")
            print(f"  n_distinct_argmax: {diag['n_distinct_argmax']}")
            print(f"  logit_std_frac:    {diag['logit_std_frac']:.6f}")
            print(f"  P@1 hits:          {diag['p_at_1_hits']}/200")
            print(f"  prior hits:        {diag['prior_p_at_1_hits']}/200")
            print(f"  Elapsed:           {elapsed:.1f}s")
        except DiagnosticError as exc:
            print(f"  DiagnosticError: {exc}")
            diag = {"status": "ERROR", "error": str(exc)}

        # Save per_label_logit_std separately (it's a list of 54 floats)
        std_vec = diag.pop("per_label_logit_std", None)
        results["variants"][variant] = diag
        if std_vec is not None:
            results["variants"][variant]["per_label_logit_std"] = std_vec

        del model, bert
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Write output
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWritten: {OUT_PATH}")
    print("\n=== LOCAL STEP-0 CROSS-CHECK SUMMARY ===")
    for v in VARIANTS:
        r = results["variants"].get(v, {})
        if r.get("status") == "OK":
            print(f"  {v}: mean_max_alpha={r['mean_max_alpha']:.4f}"
                  f"  one_hot={r['alpha_one_hot_frac']:.4f}"
                  f"  P@1={r['p_at_1_hits']}/200"
                  f"  prior={r['prior_p_at_1_hits']}/200"
                  f"  V5={'PASS' if r['p_at_1_hits'] > r['prior_p_at_1_hits'] else 'FAIL/TIE'}")
        else:
            print(f"  {v}: ERROR — {r.get('error', 'unknown')}")


if __name__ == "__main__":
    main()
