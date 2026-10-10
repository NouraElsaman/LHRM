"""
LHRM Baseline CLI -- paper-faithful reproduction.

Commands:
  prepare          -- Prepare dataset (LEVEL 1: CPU)
  run0             -- Architecture screening smoke run (LEVEL 3: GPU required)
  train            -- Full training run (LEVEL 3: GPU required)
  evaluate         -- Compute metrics from saved predictions (LEVEL 1: CPU)
  hrrm_fit         -- Fit HRRM on validation predictions (LEVEL 1: CPU)
  hrrm_apply       -- Apply fitted HRRM to predictions (LEVEL 1: CPU)
  import_artifacts -- Verify and register GPU-run artifacts (LEVEL 1: CPU)
  freeze           -- Write CONFIG_FREEZE.json (LEVEL 1: CPU)

Usage examples:
  python run_baseline.py prepare --config configs/reuters_baseline.yaml
  python run_baseline.py freeze  --config configs/reuters_baseline.yaml --out configs/final/CONFIG_FREEZE.json
  python run_baseline.py run0    --config configs/reuters_baseline.yaml --run_id RUN_0 --steps 300
  python run_baseline.py train   --config configs/reuters_baseline.yaml --run_id RUN_A
  python run_baseline.py train   --config configs/reuters_baseline.yaml --run_id RUN_A --resume_from experiments/RUN_A/checkpoints/epoch_5.pt
  python run_baseline.py evaluate --predictions experiments/RUN_A/predictions_test_raw.json
  python run_baseline.py hrrm_fit --predictions experiments/RUN_A/predictions_val.json
  python run_baseline.py hrrm_apply --hrrm experiments/RUN_A/hrrm.joblib --predictions experiments/RUN_A/predictions_test_raw.json
  python run_baseline.py import_artifacts --path /downloads/artifacts.zip

GPU gate: run0 and train require CUDA. All other commands run on CPU.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_yaml_config(path):
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except ImportError:
        cfg = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    k, _, v = line.partition(":")
                    v = v.strip().split("#")[0].strip()
                    if v in ("true", "True"):
                        v = True
                    elif v in ("false", "False"):
                        v = False
                    elif v in ("null", "None", ""):
                        v = None
                    else:
                        try:
                            v = int(v)
                        except ValueError:
                            try:
                                v = float(v)
                            except ValueError:
                                pass
                    cfg[k.strip()] = v
        return cfg


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        dirty_out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()
        return {"commit": commit, "dirty": bool(dirty_out)}
    except Exception as e:
        return {"commit": "UNKNOWN", "dirty": True, "error": str(e)}


def require_gpu():
    import torch
    if not torch.cuda.is_available():
        print("=" * 70)
        print("GPU GATE: No CUDA GPU detected.")
        print("This command requires a GPU. Options:")
        print("  1. Run on Kaggle: notebooks/lhrm_run0_reuters_gpu.ipynb")
        print("  2. Run on Colab with GPU runtime")
        print("  3. SSH to a CUDA-capable remote machine")
        print("=" * 70)
        sys.exit(1)


def write_sha256sums(out_base):
    sums = []
    for dirpath, _, fnames in os.walk(out_base):
        for fname in sorted(fnames):
            if fname == "SHA256SUMS.txt":
                continue
            full = os.path.join(dirpath, fname)
            rel = os.path.relpath(full, out_base).replace("\\", "/")
            sums.append(sha256_file(full) + "  " + rel)
    with open(os.path.join(out_base, "SHA256SUMS.txt"), "w") as f:
        f.write("\n".join(sums) + "\n")


def load_freeze(path=None):
    if path is None:
        path = os.path.join(ROOT, "configs", "final", "CONFIG_FREEZE.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# freeze: write CONFIG_FREEZE.json (P3)
# ---------------------------------------------------------------------------

def cmd_freeze(args):
    """Write CONFIG_FREEZE.json from current configs and data manifests."""
    cfg = load_yaml_config(args.config)
    git = get_git_commit()

    if git["dirty"]:
        print("WARNING: Working tree is dirty. CONFIG_FREEZE.json will record dirty=true.")
        if getattr(args, "require_clean", False):
            print("ERROR: --require_clean set; refusing to freeze dirty tree.")
            sys.exit(1)

    data_dir = cfg.get("data_dir", "data/processed/Reuters")
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(ROOT, data_dir)

    manifest_path = os.path.join(data_dir, "DATA_MANIFEST.json")
    label_hash_path = os.path.join(data_dir, "label_order.hash")
    manifest_hash = sha256_file(manifest_path) if os.path.exists(manifest_path) else "MISSING"
    label_order_hash = (open(label_hash_path).read().strip()
                        if os.path.exists(label_hash_path) else "MISSING")
    config_hash = sha256_file(args.config)

    freeze = {
        "_meta": {
            "format": "CONFIG_FREEZE_v1",
            "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git": git,
            "config_file": os.path.relpath(args.config, ROOT),
            "config_file_sha256": config_hash,
        },
        "dataset": cfg.get("dataset", "UNKNOWN"),
        "data_dir": data_dir,
        "data_manifest_sha256": manifest_hash,
        "label_order_sha256": label_order_hash,
        "bert_model": cfg.get("bert_model", "bert-base-uncased"),
        "bert_model_source": "REPOSITORY-DEFINED (paper says BERT without variant)",
        "max_len": cfg.get("max_len", 512),
        "max_len_note": "A10: head truncation; 5.88% Reuters train docs truncated at BERT limit",
        "batch_size": cfg.get("batch_size", 8),
        "num_epochs": cfg.get("num_epochs", 10),
        "learning_rate": cfg.get("learning_rate", 2e-5),
        "weight_decay": cfg.get("weight_decay", 0.01),
        "seed": cfg.get("seed", 42),
        "dynamic_padding": True,
        "tie_breaking": "seeded_random (seed XOR doc_idx) -- A16/A18",
        "architecture": cfg.get("architecture", "TBD"),
        "architecture_note": "Set after RUN_0 screening by V1-V5; NEVER chosen by performance",
        "loss": "focal",
        "focal_gamma": cfg.get("focal_loss_gamma", 2.0),
        "focal_alpha": cfg.get("focal_loss_alpha", 0.25),
        "focal_note": "ASSUMED (A9): gamma/alpha not stated in paper; Lin et al. 2017 defaults",
        "amp": cfg.get("amp", True),
        "amp_note": "NEA (A11): AMP not mentioned in paper",
        "hrrm_n_estimators": 100,
        "hrrm_learning_rate": 0.1,
        "hrrm_max_depth": 3,
        "hrrm_subsample": 1.0,
        "hrrm_criterion": "friedman_mse",
        "hrrm_n_iter_no_change": 10,
        "hrrm_tol": 0.0001,
        "hrrm_validation_fraction": 0.1,
        "hrrm_loss": "squared_error",
        "hrrm_note": "A13: PAPER-EXPLICIT Section IV-D verbatim",
        "hrrm_fitting_data": "validation_split non-leaky A17",
        "evaluation_model": "final_epoch (A19); best.pt is DIAGNOSTIC ONLY",
        "train_split_note": "TRAIN_SPLIT_DEVIATION: see reuters_resplit_seed42.json",
        "registry_version": "v4",
        "registry_path": "docs/phase1/decision_registry_v4.md",
        "experiment_ids": {
            "run0_L":   "lhrm_run0_reuters_L_focal_seed42_v1",
            "run0_L2":  "lhrm_run0_reuters_L2_focal_seed42_v1",
            "run_A":    "lhrm_baseline_reuters_{variant}_focal_seed42_v1",
            "run_A_bce": "lhrm_baseline_reuters_{variant}_bce_seed42_v1",
        },
    }

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(freeze, f, indent=2)
    print("CONFIG_FREEZE.json written:", out_path)
    print("  SHA-256:", sha256_file(out_path))
    print("  Git:", git["commit"], "dirty=" + str(git["dirty"]))
    if git["dirty"]:
        print("  WARNING: dirty=true. Commit all changes before final freeze.")


# ---------------------------------------------------------------------------
# run0: architecture-viability screening (P1)
# ---------------------------------------------------------------------------

# GPU-vs-CPU step-0 cross-check reference (from docs/phase1/run0_local_step0_crosscheck.json)
_CROSSCHECK_REFS = {
    "L": {
        "mean_max_alpha":     (0.760762, 0.001),
        "alpha_one_hot_frac": (0.41,     0.0),    # exact in fp32
        "alpha_entropy_mean": (0.714277, 0.01),
        "S_mean":             (167.2796, 0.5),
        "S_std":              (14.4664,  0.1),
        "n_distinct_argmax":  (19,       2),
        "prior_p_at_1_hits":  (63,       0),      # exact
    },
    "L2": {
        "mean_max_alpha":     (0.029699, 0.001),
        "alpha_one_hot_frac": (0.0,      0.0),
        "alpha_entropy_mean": (3.964148, 0.01),
        "S_mean":             (6.036183, 0.5),
        "S_std":              (0.522010, 0.1),
        "n_distinct_argmax":  (19,       2),
        "prior_p_at_1_hits":  (63,       0),
    },
}


def _run0_step0_crosscheck(s0, variant, out_base):
    """
    Compare GPU fp32 step-0 diagnostics to preregistered CPU fp32 reference.
    Returns True (pass) or False (fail). Writes preflight_fail_{variant}.json on failure.

    Tolerances are from docs/phase1/run0_audit.md Section 4 / run0_local_step0_crosscheck.json.
    If the reference file exists locally it overrides the hardcoded values.
    """
    # Try to load from the reference file (takes priority over hardcoded)
    ref_path = os.path.join(ROOT, "docs", "phase1", "run0_local_step0_crosscheck.json")
    refs = _CROSSCHECK_REFS.copy()
    if os.path.exists(ref_path):
        try:
            _ref_data = json.load(open(ref_path))
            for v in ("L", "L2"):
                if v in _ref_data.get("variants", {}):
                    vr = _ref_data["variants"][v]
                    tol = _ref_data.get("tolerance", {})
                    refs[v] = {
                        "mean_max_alpha":     (vr["mean_max_alpha"],    tol.get("mean_max_alpha", 0.001)),
                        "alpha_one_hot_frac": (vr["alpha_one_hot_frac"], tol.get("alpha_one_hot_frac", 0.0)),
                        "alpha_entropy_mean": (vr.get("alpha_entropy_mean", 0.0), 0.01),
                        "S_mean":             (vr.get("S_mean", 0.0), 0.5),
                        "S_std":              (vr.get("S_std", 0.0), 0.1),
                        "n_distinct_argmax":  (vr.get("n_distinct_argmax", 0), 2),
                        "prior_p_at_1_hits":  (vr["prior_p_at_1_hits"], 0),
                    }
        except Exception as exc:
            print("  WARNING: could not load cross-check reference file: {}".format(exc))

    if variant not in refs:
        print("  Preflight cross-check: variant {} not in reference table (skipped).".format(variant))
        return True

    if s0.get("diagnostic_error"):
        report = {
            "status": "FAIL",
            "reason": "step-0 DiagnosticError: {}".format(s0["diagnostic_error"]),
            "variant": variant, "s0": s0,
        }
        with open(os.path.join(out_base, "preflight_fail_{}.json".format(variant)), "w") as f:
            json.dump(report, f, indent=2)
        return False

    ref = refs[variant]
    failures = []
    for key, (expected, tol) in ref.items():
        measured = s0.get(key)
        if measured is None:
            failures.append({"field": key, "expected": expected,
                             "measured": None, "tolerance": tol,
                             "reason": "field missing from step-0 diagnostics"})
            continue
        diff = abs(float(measured) - float(expected))
        if diff > tol:
            failures.append({"field": key, "expected": expected,
                             "measured": measured, "diff": diff, "tolerance": tol})

    if failures:
        report = {
            "status": "FAIL",
            "variant": variant,
            "failures": failures,
            "s0": {k: v for k, v in s0.items() if k != "per_label_logit_std"},
            "reference_file": ref_path if os.path.exists(ref_path) else "HARDCODED",
            "message": (
                "GPU step-0 values do not match CPU fp32 reference within tolerance. "
                "This indicates a BERT checkpoint mismatch, wrong dataset, wrong val IDs, "
                "or a code path difference. Investigate before trusting step-300 results."
            ),
        }
        with open(os.path.join(out_base, "preflight_fail_{}.json".format(variant)), "w") as f:
            json.dump(report, f, indent=2)
        print("  PREFLIGHT FAIL ({} checks failed):".format(len(failures)))
        for fl in failures:
            print("    {}: measured={} expected={} tol={} diff={}".format(
                fl["field"], fl.get("measured"), fl["expected"],
                fl["tolerance"], fl.get("diff", "N/A")))
        return False

    print("  Preflight cross-check PASSED ({} fields within tolerance).".format(len(ref)))
    return True


def cmd_run0(args):
    """
    RUN_0: Architecture-viability screening (LEVEL 3 -- GPU required).
    All outputs: SMOKE_ONLY / DIAGNOSTIC_ONLY.
    Test split is NOT predicted.
    See docs/phase1/run0_preregistration.md.
    """
    require_gpu()
    import torch
    import numpy as np
    import contextlib

    cfg = load_yaml_config(args.config)
    run_id  = args.run_id
    n_steps = args.steps if not args.extended else 1000
    candidates = [c.strip() for c in args.candidates.split(",")]
    seed = cfg.get("seed", 42)

    out_base = os.path.join(ROOT, "experiments", run_id)
    os.makedirs(out_base, exist_ok=True)

    freeze = load_freeze()
    if freeze is None:
        print("WARNING: CONFIG_FREEZE.json not found. Run freeze first.")

    from lhrm_baseline.data import load_processed, MultiLabelDataset, collate_fn, tokenize_labels
    from lhrm_baseline.labels import build_all_label_texts
    from lhrm_baseline.model import LHRMModel, build_bert_model
    from lhrm_baseline.losses import get_loss
    from lhrm_baseline.metrics import precision_at_k
    from lhrm_baseline.train import forward_pass, predict_epoch_loader, save_predictions
    from transformers import BertTokenizer
    from torch.utils.data import DataLoader

    def set_seed(s):
        random.seed(s); np.random.seed(s); torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

    set_seed(seed)
    device = torch.device("cuda")

    data_dir = cfg.get("data_dir", "data/processed/AAPD")
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(ROOT, data_dir)

    dataset_name = cfg.get("dataset", "aapd")

    print("=" * 60)
    print("RUN_0 -- {} Architecture Screening".format(dataset_name.upper()))
    print("Candidates:", candidates, "  Steps:", n_steps, "  Seed:", seed)
    if args.extended:
        print("EXTENDED run (1000 steps, pre-declared)")
    print("=" * 60)

    train_texts, train_labels, label_order = load_processed(data_dir, "train")
    val_texts, val_labels, _ = load_processed(data_dir, "dev")

    # Fixed 200-doc val sample: load from prior run if file exists; otherwise generate and save.
    # If the file exists AND was pre-registered, the IDs are locked — we must reuse exactly.
    _saved_ids_path = os.path.join(out_base, "run0_val_sample_ids.json")
    _freeze_ids_path = os.path.join(ROOT, "experiments", "RUN_0", "run0_val_sample_ids.json")
    if os.path.exists(_saved_ids_path):
        with open(_saved_ids_path) as f:
            val_sample_ids = json.load(f)
        print("Validation sample: {} docs (IDs loaded from existing file)".format(len(val_sample_ids)))
    elif os.path.exists(_freeze_ids_path) and _freeze_ids_path != _saved_ids_path:
        # Reuse the pre-registered RUN_0 IDs (cross-check reference)
        with open(_freeze_ids_path) as f:
            val_sample_ids = json.load(f)
        with open(_saved_ids_path, "w") as f:
            json.dump(val_sample_ids, f)
        print("Validation sample: {} docs (IDs copied from registered RUN_0)".format(len(val_sample_ids)))
    else:
        val_rng = random.Random(seed)
        val_n = min(200, len(val_texts))
        val_sample_ids = sorted(val_rng.sample(range(len(val_texts)), val_n))
        with open(_saved_ids_path, "w") as f:
            json.dump(val_sample_ids, f)
        print("Validation sample: {} docs (IDs generated and saved)".format(len(val_sample_ids)))
    assert len(val_sample_ids) == 200, "Expected 200 val IDs; got {}".format(len(val_sample_ids))
    v_texts  = [val_texts[i]  for i in val_sample_ids]
    v_labels = val_labels[val_sample_ids]

    bert_name = cfg.get("bert_model", "bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    _default_defs = (
        os.path.join(ROOT, "Label_Definition", "arxiv_labels_definitions.json")
        if dataset_name == "aapd" else
        os.path.join(ROOT, "Label_Definition", "reuters_labels_definitions.json")
    )
    defs_path = cfg.get("defs_path", _default_defs)
    if not os.path.isabs(defs_path):
        defs_path = os.path.join(ROOT, defs_path)
    label_texts_list = build_all_label_texts(label_order, defs_path, dataset=dataset_name)

    # Verify label-order hash against preregistered value
    import hashlib as _hl
    _label_hash = _hl.sha256(json.dumps(label_order).encode()).hexdigest()
    _expected_hash = "a3d662f3920259a94ed61d83f98f21cabe7f908d98a5662ccd9adc2413cc000b"
    if freeze and freeze.get("label_order_sha256") not in (None, "MISSING"):
        _expected_hash = freeze["label_order_sha256"]
    assert _label_hash == _expected_hash, (
        "PREFLIGHT FAIL: label_order SHA256 mismatch!\n"
        "  got:      {}\n  expected: {}".format(_label_hash, _expected_hash)
    )
    print("Label-order hash: OK ({}...)".format(_label_hash[:12]))
    print("Labels:", len(label_order))
    assert len(label_texts_list) == len(label_order), (
        "PREFLIGHT FAIL: label_texts count {} != label_order count {}".format(
            len(label_texts_list), len(label_order)))
    _empty = [i for i, t in enumerate(label_texts_list) if not t.strip()]
    assert not _empty, "PREFLIGHT FAIL: empty label texts at indices {}".format(_empty)
    label_ids, label_mask = tokenize_labels(label_texts_list, tokenizer, max_len=64, device=device)

    max_len  = cfg.get("max_len", 512)
    batch_sz = cfg.get("batch_size", 8)
    use_amp  = cfg.get("amp", True)
    criterion = get_loss("focal", focal_gamma=cfg.get("focal_loss_gamma", 2.0),
                         focal_alpha=cfg.get("focal_loss_alpha", 0.25))

    train_ds = MultiLabelDataset(train_texts, train_labels, tokenizer, max_len)
    val_ds   = MultiLabelDataset(v_texts, v_labels, tokenizer, max_len)

    # F2 fix: prior hit count computed by diagnostics module (integer, seeded tie-break)
    from lhrm_baseline.diagnostics import frequency_prior_hits_at_1, DiagnosticError
    prior_hits_on_val = frequency_prior_hits_at_1(train_labels, v_labels, seed=seed)
    print("  Prior P@1 on 200 val docs (seeded tie-break): {}/{} = {:.4f}".format(
        prior_hits_on_val, len(v_labels), prior_hits_on_val / max(len(v_labels), 1)))
    assert prior_hits_on_val == 63, (
        "PREFLIGHT FAIL: prior hits = {} (expected 63 for AAPD dev, seed=42). "
        "Wrong dataset, IDs or seed?".format(prior_hits_on_val))

    results = {}

    for variant in candidates:
        print("\n" + "-" * 50)
        print("Candidate: Variant", variant)
        print("-" * 50)
        cand_dir = os.path.join(out_base, "variant_" + variant)
        ckpt_dir = os.path.join(cand_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        set_seed(seed)
        train_loader = DataLoader(train_ds, batch_size=batch_sz, shuffle=True,
                                  collate_fn=collate_fn, num_workers=0, pin_memory=True)
        val_loader   = DataLoader(val_ds, batch_size=batch_sz, shuffle=False,
                                  collate_fn=collate_fn, num_workers=0, pin_memory=True)

        bert = build_bert_model(bert_name, device)
        model = LHRMModel(bert, len(label_order), variant=variant).to(device)
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=cfg.get("learning_rate", 2e-5),
                                      weight_decay=cfg.get("weight_decay", 0.01))
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        # Step-0 diagnostics (F1/F2/F3 fix: fp32, alpha tensor, integer prior)
        model.eval()
        try:
            s0 = _val_diagnostics_v2(model, val_loader, label_ids, label_mask,
                                      device, v_labels, train_labels, n_labels=len(label_order),
                                      seed=seed, amp_for_training=use_amp)
            s0["step"] = 0
            print("  Step 0: mean_max_alpha={:.6f} one_hot_frac={:.4f} P@1={}/{} S_mean={:.2f}".format(
                s0["mean_max_alpha"], s0["alpha_one_hot_frac"],
                s0["p_at_1_hits"], s0["n_docs"],
                s0.get("S_mean", float("nan"))))
        except DiagnosticError as exc:
            print("  Step 0 DiagnosticError:", exc)
            s0 = {"step": 0, "diagnostic_error": str(exc)}

        # GPU-vs-CPU step-0 preflight cross-check (fail-closed)
        _preflight_ok = _run0_step0_crosscheck(s0, variant, out_base)
        if not _preflight_ok:
            print("PREFLIGHT CROSS-CHECK FAILED for variant {}. Stopping.".format(variant))
            print("See {}/preflight_fail_{}.json for details.".format(out_base, variant))
            sys.exit(2)

        # Training loop
        model.train()
        losses, step_times = [], []
        nan_inf_count, amp_skips = 0, 0
        set_seed(seed)
        train_iter = iter(train_loader)

        for step in range(1, n_steps + 1):
            try:
                batch = next(train_iter)
            except StopIteration:
                set_seed(seed + step)
                train_iter = iter(train_loader)
                batch = next(train_iter)

            t0 = time.time()
            input_ids  = batch["input_ids"].to(device)
            attn_mask  = batch["attention_mask"].to(device)
            targets    = batch["labels"].to(device)

            optimizer.zero_grad()
            ctx = torch.amp.autocast("cuda") if use_amp else contextlib.nullcontext()
            with ctx:
                logits, _ = forward_pass(model, input_ids, attn_mask, label_ids, label_mask)
                loss = criterion(logits, targets)

            if torch.isnan(loss) or torch.isinf(loss):
                nan_inf_count += 1
            else:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() < old_scale:
                        amp_skips += 1
                else:
                    loss.backward()
                    optimizer.step()

            losses.append(float(loss.detach()))
            step_times.append(time.time() - t0)

            if step % 50 == 0:
                print("  Step {:4d}/{}: loss={:.4f}".format(step, n_steps, losses[-1]))

            if step == 150:
                ckpt = {"step": step, "variant": variant,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scaler_state": scaler.state_dict() if scaler else None,
                        "config": cfg}
                torch.save(ckpt, os.path.join(ckpt_dir, "step_150.pt"))
                print("  Checkpoint saved: step_150.pt")

        # Final checkpoint
        torch.save({"step": n_steps, "variant": variant,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "losses": losses, "config": cfg},
                   os.path.join(ckpt_dir, "step_{}.pt".format(n_steps)))

        # Step-300 diagnostics (F1/F2/F3 fix: fp32, alpha tensor, integer prior)
        model.eval()
        try:
            s300 = _val_diagnostics_v2(model, val_loader, label_ids, label_mask,
                                        device, v_labels, train_labels,
                                        n_labels=len(label_order), seed=seed,
                                        amp_for_training=use_amp)
            s300["step"] = n_steps
            s300["nan_inf_count"] = nan_inf_count
        except DiagnosticError as exc:
            print("  Step-300 DiagnosticError:", exc)
            s300 = {"step": n_steps, "nan_inf_count": nan_inf_count,
                    "diagnostic_error": str(exc)}

        # V1-V5 (via diagnostics.evaluate_criteria)
        from lhrm_baseline.diagnostics import evaluate_criteria
        crit = evaluate_criteria(s0, s300, losses, n_labels=len(label_order))
        v1, v2, v3, v4, v5 = crit["V1"], crit["V2"], crit["V3"], crit["V4"], crit["V5"]
        viable   = crit["viable"]
        escalate = crit["escalate_v3"]

        tput     = len(losses[100:]) / max(sum(step_times[100:]), 1e-6)
        peak_mem = torch.cuda.max_memory_allocated(device) / 1024 ** 3 if torch.cuda.is_available() else 0.0

        print("\n  V1(no NaN): {} ({} bad, {} AMP skips)".format(v1, nan_inf_count, amp_skips))
        print("  V2(loss down): {} (first50={:.4f} last50={:.4f})".format(
            v2, crit["mean_loss_first50"], crit["mean_loss_last50"]))
        print("  V3(alpha): {} (one_hot_frac={} < 0.9?)".format(v3, crit["V3_value"]))
        print("  V4(logit var): {} (logit_std_frac={} > 0.5?)".format(v4, crit["V4_value"]))
        print("  V5(P@1 hits): {} (cand={} vs prior={} [integer, strict>])".format(
            v5, crit["p_at_1_hits"], crit["prior_p_at_1_hits"]))
        print("  VIABLE:", viable)
        if escalate:
            print("  ESCALATE: L fails V3 but argmax-alpha is true label for"
                  " {:.2f} of val docs.".format(crit.get("argmax_alpha_is_true_frac", 0)))
        print("  Throughput: {:.2f} steps/s  Peak GPU: {:.2f} GB".format(tput, peak_mem))

        cand_result = {
            "variant": variant,
            **crit,
            "nan_inf_count": nan_inf_count, "amp_skips": amp_skips,
            "prior_hits_on_val": prior_hits_on_val,
            "throughput_steps_per_s": tput, "peak_gpu_mem_gb": peak_mem,
            "step0_diagnostics":   {k: v for k, v in s0.items()
                                    if k not in ("losses", "per_label_logit_std")},
            "step300_diagnostics": {k: v for k, v in s300.items()
                                    if k not in ("losses", "per_label_logit_std")},
            "per_label_logit_std_step300": s300.get("per_label_logit_std", []),
            "losses": losses,
        }
        results[variant] = cand_result

        # Save per-candidate files
        with open(os.path.join(cand_dir, "step0_diagnostics.json"), "w") as f:
            json.dump({k: v for k, v in s0.items() if k != "losses"}, f, indent=2)
        with open(os.path.join(cand_dir, "step300_diagnostics.json"), "w") as f:
            json.dump({k: v for k, v in s300.items() if k != "losses"}, f, indent=2)
        with open(os.path.join(cand_dir, "train.log"), "w") as f:
            for i, loss_val in enumerate(losses, 1):
                f.write("step={} loss={:.6f}\n".format(i, loss_val))

        y_true, y_pred, _ = predict_epoch_loader(
            model, val_loader, device, label_ids, label_mask)
        save_predictions(y_true, y_pred, label_order,
                         os.path.join(cand_dir, "val_sample_predictions.json"))
        torch.cuda.reset_peak_memory_stats(device)

    # Scripted decision (via diagnostics.run0_decision — single source of truth)
    from lhrm_baseline.diagnostics import run0_decision as _diag_decision
    decision = _diag_decision({v: results[v] for v in results}, candidates=candidates)
    print("\n" + "=" * 60)
    print("SCRIPTED DECISION:", decision["chosen_variant"], "--", decision["reason"])
    print("=" * 60)

    with open(os.path.join(out_base, "run0_decision.json"), "w") as f:
        json.dump(decision, f, indent=2)

    # env.json
    import torch
    env = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "python_version": sys.version,
    }
    try:
        import transformers
        env["transformers_version"] = transformers.__version__
        env["bert_tokenizer_vocab_size"] = tokenizer.vocab_size
        env["bert_name"] = bert_name
    except (ImportError, NameError):
        pass
    with open(os.path.join(out_base, "env.json"), "w") as f:
        json.dump(env, f, indent=2)

    # resolved_config.json: full provenance record
    import hashlib as _hl2
    _val_ids_hash = _hl2.sha256(json.dumps(val_sample_ids).encode()).hexdigest()
    resolved_cfg = {
        "config_file": os.path.relpath(args.config, ROOT),
        "run_id": run_id,
        "dataset": dataset_name,
        "data_dir": data_dir,
        "defs_path": defs_path,
        "label_order_sha256": _hl2.sha256(json.dumps(label_order).encode()).hexdigest(),
        "n_labels": len(label_order),
        "val_sample_ids_sha256": _val_ids_hash,
        "n_val_docs": len(val_sample_ids),
        "prior_p_at_1_hits": prior_hits_on_val,
        "bert_model": bert_name,
        "max_len": max_len,
        "batch_size": batch_sz,
        "n_steps": n_steps,
        "seed": seed,
        "amp": use_amp,
        "candidates": candidates,
        "git": get_git_commit(),
        "crosscheck_ref_file": os.path.join(ROOT, "docs", "phase1", "run0_local_step0_crosscheck.json"),
    }
    with open(os.path.join(out_base, "resolved_config.json"), "w") as f:
        json.dump(resolved_cfg, f, indent=2)

    write_sha256sums(out_base)
    print("RUN_0 complete. Artifacts:", out_base)
    print("Next: python run_baseline.py import_artifacts --path", out_base)


def _val_diagnostics_v2(model, val_loader, label_ids, label_mask, device,
                         val_labels_np, train_labels_np, n_labels, seed=42,
                         amp_for_training=True):
    """
    Thin wrapper around diagnostics.compute_diagnostics.
    F1/F2/F3 fix: fp32 (no autocast), alpha tensor required, integer prior.
    Raises DiagnosticError if alpha is missing or malformed.
    The old _val_diagnostics is REMOVED; this is the single implementation.
    """
    from lhrm_baseline.diagnostics import compute_diagnostics
    return compute_diagnostics(
        model=model,
        loader=val_loader,
        label_input_ids=label_ids,
        label_attention_mask=label_mask,
        val_labels=val_labels_np,
        device=device,
        train_labels=train_labels_np,
        n_labels=n_labels,
        seed=seed,
        amp_for_training=amp_for_training,
    )


# ---------------------------------------------------------------------------
# import_artifacts (P1)
# ---------------------------------------------------------------------------

def cmd_import_artifacts(args):
    """Verify and register artifacts from a GPU run (LEVEL 1 -- CPU)."""
    src = args.path
    freeze = load_freeze()
    errors = []

    if src.endswith(".zip"):
        with zipfile.ZipFile(src) as z:
            names = z.namelist()
            top_dirs = {n.split("/")[0] for n in names if "/" in n}
            exp_id = top_dirs.pop() if len(top_dirs) == 1 else os.path.basename(src).replace(".zip", "")
            extract_to = os.path.join(ROOT, "experiments")
            os.makedirs(extract_to, exist_ok=True)
            z.extractall(extract_to)
            artifact_dir = os.path.join(extract_to, exp_id)
            print("Extracted to:", artifact_dir)
    elif os.path.isdir(src):
        artifact_dir = src
        exp_id = os.path.basename(src.rstrip("/\\"))
    else:
        print("ERROR: --path must be a .zip or directory. Got:", src)
        sys.exit(1)

    # 1. Verify SHA256SUMS
    sums_path = os.path.join(artifact_dir, "SHA256SUMS.txt")
    if not os.path.exists(sums_path):
        errors.append("MISSING: SHA256SUMS.txt")
    else:
        print("Verifying SHA256SUMS...")
        with open(sums_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        for line in lines:
            exp_hash, rel_path = line.split("  ", 1)
            full = os.path.join(artifact_dir, rel_path.replace("/", os.sep))
            if not os.path.exists(full):
                errors.append("MISSING file: " + rel_path)
                continue
            act = sha256_file(full)
            if act != exp_hash:
                errors.append("HASH MISMATCH: " + rel_path)
        if not [e for e in errors if "HASH" in e or "MISSING" in e]:
            print("  SHA256SUMS:", len(lines), "files OK")

    # 2. Verify resolved_config.json
    cfg_path = os.path.join(artifact_dir, "resolved_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            run_cfg = json.load(f)
        if freeze:
            freeze_path = os.path.join(ROOT, "configs", "final", "CONFIG_FREEZE.json")
            local_hash = sha256_file(freeze_path) if os.path.exists(freeze_path) else ""
            run_hash = run_cfg.get("_config_freeze_sha256", "")
            if run_hash and local_hash and run_hash != local_hash:
                errors.append("CONFIG_FREEZE MISMATCH: artifact vs local")
            run_lh = run_cfg.get("_label_order_sha256", "")
            if freeze and run_lh and run_lh != freeze.get("label_order_sha256", ""):
                errors.append("LABEL_ORDER HASH MISMATCH: artifact vs local")
        print("  Config: seed={}, dataset={}, variant={}".format(
            run_cfg.get("seed"), run_cfg.get("dataset"), run_cfg.get("architecture")))
    else:
        errors.append("MISSING: resolved_config.json")

    # 3. Required files
    # train.log: may be at root (legacy) or under variant_* dirs (run0 multi-candidate)
    _has_root_trainlog  = os.path.exists(os.path.join(artifact_dir, "train.log"))
    _has_variant_trainlog = bool([
        f for f in os.listdir(artifact_dir)
        if f.startswith("variant_") and os.path.exists(
            os.path.join(artifact_dir, f, "train.log"))
    ]) if os.path.isdir(artifact_dir) else False
    if not _has_root_trainlog and not _has_variant_trainlog:
        errors.append("MISSING: train.log (checked root and variant_*/train.log)")
    elif not _has_root_trainlog:
        print("  NOTE: no root train.log; found variant-level train logs (run0 multi-candidate format)")
    # results_summary.json is optional (not generated by run0)
    if not os.path.exists(os.path.join(artifact_dir, "results_summary.json")):
        print("  NOTE: results_summary.json absent (optional; not generated by run_baseline run0)")
    # env.json is required
    if not os.path.exists(os.path.join(artifact_dir, "env.json")):
        errors.append("MISSING: env.json")

    if errors:
        print("\n" + "=" * 60)
        print("IMPORT FAILED ({} errors):".format(len(errors)))
        for e in errors:
            print("  ERROR:", e)
        print("=" * 60)
        sys.exit(1)

    reg = {"experiment_id": exp_id, "artifact_dir": artifact_dir,
           "import_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "sha256sums_verified": True, "config_verified": freeze is not None,
           "status": "REGISTERED"}
    reg_path = os.path.join(artifact_dir, "import_registration.json")
    with open(reg_path, "w") as f:
        json.dump(reg, f, indent=2)
    print("\nIMPORT SUCCESS:", exp_id)
    print("  Registered:", reg_path)

    summary_path = os.path.join(artifact_dir, "results_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            print("\nResults summary:")
            print(json.dumps(json.load(f), indent=2))


# ---------------------------------------------------------------------------
# Existing commands
# ---------------------------------------------------------------------------

def cmd_prepare(args):
    cfg = load_yaml_config(args.config)
    dataset = cfg.get("dataset", "aapd")
    if dataset == "reuters":
        from lhrm_baseline.data import load_processed
        data_dir = cfg.get("data_dir", "data/processed/Reuters")
        if not os.path.isabs(data_dir):
            data_dir = os.path.join(ROOT, data_dir)
        for split in ["train", "dev", "test"]:
            texts, labels, lo = load_processed(data_dir, split)
            print("{}: {} docs, {} labels".format(split, len(texts), labels.shape[1]))
        print("Use scripts/prepare_reuters.py to re-generate.")
    else:
        from lhrm_baseline.data import load_aapd_split, save_processed, validate_splits
        from lhrm_baseline.labels import build_all_label_texts
        zip_path = cfg.get("zip_path", "Data/AAPD.zip")
        out_dir  = cfg.get("data_dir", "data/processed/AAPD")
        defs_path = cfg.get("defs_path", "Label_Definition/arxiv_labels_definitions.json")
        splits = {}
        label_order = None
        for split in ["train", "dev", "test"]:
            texts, labels, label_cols = load_aapd_split(zip_path, split)
            splits[split] = (texts, labels)
            if label_order is None:
                label_order = label_cols
            print("{}: {} docs, {} labels".format(split, len(texts), labels.shape[1]))
        label_texts = build_all_label_texts(label_order, defs_path, dataset="aapd")
        report = validate_splits(splits["train"][1], splits["dev"][1], splits["test"][1], label_order)
        print(json.dumps(report, indent=2))
        os.makedirs(out_dir, exist_ok=True)
        from lhrm_baseline.data import save_processed as _sp
        for split, (texts, labels) in splits.items():
            _sp(out_dir, split, texts, labels, label_order)
        with open(os.path.join(out_dir, "label_texts.json"), "w", encoding="utf-8") as f:
            json.dump(label_texts, f, indent=2, ensure_ascii=False)
        print("Saved to:", out_dir)
        print("WARNING: UNVERIFIED_LABEL_MAPPING (D3) -- AAPD training blocked.")


def cmd_train(args):
    """
    Full training command — TRAIN and DEV splits only.

    A1: Test split is NEVER loaded or predicted during training.
        Use `cmd_test_eval` (a separately authorized command) for final test evaluation.
    A5: data_dir is resolved case-sensitively; absolute path is logged and verified.
    A9: Rolling checkpoints only: latest.pt and previous.pt (one swap per epoch).
        Optional best_dev.pt (model state only, no optimizer) when dev P@1 improves.
        Per-epoch full checkpoints are NOT saved by default (avoids 15 × 1.3 GB).
    A4: Resume from epoch boundary only. Exact mid-epoch resume not supported;
        limitation is documented in save_checkpoint_v2.
    """
    require_gpu()
    import torch
    import numpy as np
    import sys, time as _time

    cfg = load_yaml_config(args.config)
    run_id      = getattr(args, "run_id", "RUN_unnamed")
    resume_from = getattr(args, "resume_from", None)
    variant     = getattr(args, "variant",
                          cfg.get("architecture", cfg.get("variant", "L2")))
    if variant == "TBD":
        print("ERROR: architecture=TBD in config — set --variant L or L2 explicitly.")
        sys.exit(1)

    from lhrm_baseline.data import load_processed, MultiLabelDataset, collate_fn, tokenize_labels
    from lhrm_baseline.labels import build_all_label_texts
    from lhrm_baseline.model import LHRMModel, build_bert_model
    from lhrm_baseline.losses import get_loss
    from lhrm_baseline.train import (train_epoch_stepped, predict_epoch_loader,
                                      save_predictions, log_metrics, save_checkpoint_v2)
    from lhrm_baseline.metrics import compute_all_metrics, compute_per_label_metrics
    from transformers import BertTokenizer
    from torch.utils.data import DataLoader

    def set_seed(s):
        import random as _rnd
        _rnd.seed(s); np.random.seed(s); torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

    device = torch.device("cuda")
    seed   = cfg.get("seed", 42)
    set_seed(seed)

    # --- A5: resolve data_dir case-sensitively; log absolute path ---
    data_dir = cfg.get("data_dir", "data/processed/AAPD")
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(ROOT, data_dir)
    # Try case variants for Kaggle Linux (Data/ vs data/)
    if not os.path.isdir(data_dir):
        alt = os.path.join(ROOT, "Data", "processed", "AAPD")
        if os.path.isdir(alt):
            print("NOTE: data_dir resolved to case-alternative:", alt)
            data_dir = alt
    if not os.path.isdir(data_dir):
        print("ERROR: data_dir not found:", data_dir)
        sys.exit(1)
    print("data_dir (absolute):", data_dir)

    # Verify label hash before loading
    import hashlib as _hl
    labels_json = os.path.join(data_dir, "labels.json")
    if os.path.exists(labels_json):
        label_order_raw = open(labels_json).read()
        label_order = json.loads(label_order_raw)
        lo_hash = _hl.sha256(json.dumps(label_order).encode()).hexdigest()
        expected_lo_hash = "a3d662f3920259a94ed61d83f98f21cabe7f908d98a5662ccd9adc2413cc000b"
        print("label_order SHA256:", lo_hash)
        if lo_hash != expected_lo_hash:
            print("WARNING: label_order hash mismatch! expected:", expected_lo_hash)

    # A1: Load ONLY train and dev — NEVER test
    train_texts, train_labels, label_order = load_processed(data_dir, "train")
    val_texts,   val_labels,   _           = load_processed(data_dir, "dev")
    print("Train docs:", len(train_texts), " Dev docs:", len(val_texts),
          " Labels:", len(label_order))
    print("TEST SPLIT: not loaded (A1 isolation). Use cmd_test_eval after training.")

    bert_name = cfg.get("bert_model", "bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    print("BERT tokenizer vocab_size:", tokenizer.vocab_size)

    defs_path = cfg.get("defs_path", "Label_Definition/arxiv_labels_definitions.json")
    if not os.path.isabs(defs_path):
        defs_path = os.path.join(ROOT, defs_path)
    print("defs_path:", defs_path)
    dataset_name = cfg.get("dataset", "aapd")
    label_texts_list = build_all_label_texts(label_order, defs_path, dataset=dataset_name)

    max_len    = int(cfg.get("max_len", 512))
    batch_size = int(cfg.get("batch_size", 8))
    use_amp    = bool(cfg.get("amp", True))
    epochs     = int(cfg.get("num_epochs", 15))
    lr         = float(cfg.get("learning_rate", 2e-5))
    wd         = float(cfg.get("weight_decay", 0.01))

    train_ds = MultiLabelDataset(train_texts, train_labels, tokenizer, max_len)
    val_ds   = MultiLabelDataset(val_texts, val_labels, tokenizer, max_len)
    set_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=0, pin_memory=True)

    label_ids, label_mask = tokenize_labels(label_texts_list, tokenizer, max_len=64, device=device)

    bert  = build_bert_model(bert_name, device)
    model = LHRMModel(bert, len(label_order), variant=variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scaler    = torch.amp.GradScaler("cuda") if use_amp else None
    criterion = get_loss(cfg.get("loss", "focal"),
                         focal_gamma=float(cfg.get("focal_loss_gamma", 2.0)),
                         focal_alpha=float(cfg.get("focal_loss_alpha", 0.25)))

    out_dir  = os.path.join(ROOT, "experiments", run_id)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    log_file          = os.path.join(out_dir, "train.log")
    progress_jsonl    = os.path.join(out_dir, "train_progress.jsonl")
    epoch_metrics_csv = os.path.join(out_dir, "epoch_metrics.csv")

    # Rolling checkpoint paths
    latest_ckpt   = os.path.join(ckpt_dir, "latest.pt")
    previous_ckpt = os.path.join(ckpt_dir, "previous.pt")
    best_dev_ckpt = os.path.join(ckpt_dir, "best_dev_model_only.pt")  # model state only

    freeze_path = os.path.join(ROOT, "configs", "final", "CONFIG_FREEZE.json")
    resolved = dict(cfg)
    resolved["_run_id"]   = run_id
    resolved["_variant"]  = variant
    resolved["_git"]      = get_git_commit()
    resolved["_config_freeze_sha256"] = sha256_file(freeze_path) if os.path.exists(freeze_path) else "MISSING"
    import hashlib as _hl2
    resolved["_label_order_sha256"] = _hl2.sha256(
        json.dumps(label_order).encode()).hexdigest()
    resolved["_data_dir"] = data_dir
    resolved["_bert_model"] = bert_name
    resolved["_test_split_loaded"] = False   # A1 provenance
    with open(os.path.join(out_dir, "resolved_config.json"), "w") as f:
        json.dump(resolved, f, indent=2)

    steps_per_epoch   = len(train_loader)
    total_global_steps = epochs * steps_per_epoch

    start_epoch = 1
    total_steps = 0
    if resume_from and os.path.exists(resume_from):
        print("Resuming from:", resume_from)
        from lhrm_baseline.train import load_checkpoint as _lc
        start_epoch = _lc(resume_from, model, optimizer, scaler, device) + 1
        # Recover history from checkpoint
        ckpt_data = torch.load(resume_from, map_location="cpu")
        history = list(ckpt_data.get("history", []))
        total_steps = ckpt_data.get("step", (start_epoch - 1) * steps_per_epoch)
        print("Resumed at epoch {} / global step {}".format(start_epoch, total_steps))
        print("NOTE: epoch-boundary resume only; batch order within epoch not guaranteed (A4).")
    else:
        history = []

    best_dev_p1 = max((row.get("val_P@1", 0.0) for row in history), default=0.0)

    # --- Provenance banner ---
    print("\n" + "="*70)
    print("TRAINING START")
    print("  run_id:       ", run_id)
    print("  variant:      ", variant)
    print("  dataset:      ", dataset_name, " train=", len(train_texts), " dev=", len(val_texts))
    print("  label_hash:   ", resolved["_label_order_sha256"][:16], "...")
    print("  bert_model:   ", bert_name)
    print("  epochs:       ", epochs, "  steps/epoch:", steps_per_epoch)
    print("  total_steps:  ", total_global_steps)
    print("  batch_size:   ", batch_size, "  lr:", lr, "  wd:", wd)
    print("  amp:          ", use_amp, "  seed:", seed)
    print("  loss:         ", cfg.get("loss"), "  gamma:", cfg.get("focal_loss_gamma"),
          "  alpha:", cfg.get("focal_loss_alpha"))
    print("  test_split:   NOT LOADED (A1)")
    print("  GPU:          ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    print("="*70, "\n", flush=True)

    # Write CSV header if new run
    if not os.path.exists(epoch_metrics_csv):
        with open(epoch_metrics_csv, "w") as f:
            f.write("epoch,train_loss,global_steps,"
                    "val_P@1,val_P@3,val_P@5,val_nDCG@3,val_nDCG@5,"
                    "val_micro_f1,val_macro_f1,val_threshold\n")

    for epoch in range(start_epoch, epochs + 1):
        print("\n=== Epoch {}/{} ===".format(epoch, epochs), flush=True)
        epoch_global_offset = total_steps

        train_loss, steps_ep, amp_skips = train_epoch_stepped(
            model, train_loader, criterion, optimizer, scaler,
            device, label_ids, label_mask, use_amp=use_amp,
            log_every=100,
            global_step_offset=epoch_global_offset,
            epoch=epoch, total_epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            total_global_steps=total_global_steps,
            run_id=run_id, variant=variant,
            progress_jsonl_path=progress_jsonl,
        )
        total_steps += steps_ep

        # --- Full dev evaluation ---
        y_true_v, y_pred_v, _ = predict_epoch_loader(
            model, val_loader, device, label_ids, label_mask)
        val_m = compute_all_metrics(y_true_v, y_pred_v, [1, 3, 5],
                                    threshold=0.5, include_f1=True)

        row = {"epoch": epoch, "train_loss": train_loss, "global_steps": total_steps,
               "amp_skips": amp_skips,
               **{"val_" + k: v for k, v in val_m.items()}}
        history.append(row)

        # Epoch-level log
        print("[VAL_FULL] epoch={} | train_loss={:.4f} | P@1={:.4f} | P@3={:.4f} | P@5={:.4f} "
              "| nDCG@3={:.4f} | nDCG@5={:.4f} | micro_f1={:.4f} (thr=0.5) "
              "| macro_f1={:.4f} | amp_skips={}".format(
            epoch, train_loss,
            val_m.get("P@1", 0), val_m.get("P@3", 0), val_m.get("P@5", 0),
            val_m.get("nDCG@3", 0), val_m.get("nDCG@5", 0),
            val_m.get("micro_f1", 0), val_m.get("macro_f1", 0),
            amp_skips), flush=True)

        if val_m.get("P@1", 0) <= val_m.get("prior_p_at_1", 0):
            print("WARNING: val P@1 has not exceeded prior — check training progress.")

        log_line = json.dumps({"type": "EPOCH_COMPLETE", **row,
                               "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime())})
        with open(log_file, "a") as f:
            f.write(log_line + "\n")
        with open(progress_jsonl, "a") as f:
            f.write(log_line + "\n")

        # CSV row
        with open(epoch_metrics_csv, "a") as f:
            f.write("{},{:.6f},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},0.5\n".format(
                epoch, train_loss, total_steps,
                val_m.get("P@1",0), val_m.get("P@3",0), val_m.get("P@5",0),
                val_m.get("nDCG@3",0), val_m.get("nDCG@5",0),
                val_m.get("micro_f1",0), val_m.get("macro_f1",0)))

        # A9: Rolling checkpoints — latest.pt and previous.pt only
        # Rotate: previous <- latest, then save new latest
        ckpt_payload = {
            "epoch": epoch, "step": total_steps, "variant": variant,
            "model_state": model.state_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if scaler else None,
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": (torch.cuda.get_rng_state()
                               if torch.cuda.is_available() else None),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": __import__("random").getstate(),
            "config": resolved, "history": history,
        }
        # Rotate
        if os.path.exists(latest_ckpt):
            tmp_prev = previous_ckpt + ".tmp"
            import shutil
            shutil.copy2(latest_ckpt, tmp_prev)
            if os.path.exists(previous_ckpt):
                os.remove(previous_ckpt)
            os.rename(tmp_prev, previous_ckpt)
        # Save latest atomically
        tmp_latest = latest_ckpt + ".tmp"
        torch.save(ckpt_payload, tmp_latest)
        if os.path.exists(latest_ckpt):
            os.remove(latest_ckpt)
        os.rename(tmp_latest, latest_ckpt)
        print("Checkpoint saved: latest.pt (epoch {}, global_step {})".format(
            epoch, total_steps), flush=True)

        # Optional: best_dev model-only checkpoint (does not contain optimizer)
        val_p1 = val_m.get("P@1", 0.0)
        if val_p1 > best_dev_p1:
            best_dev_p1 = val_p1
            torch.save({"epoch": epoch, "step": total_steps, "variant": variant,
                        "model_state": model.state_dict(),
                        "val_P@1": val_p1,
                        "note": "DIAGNOSTIC ONLY — best dev P@1. Not for final reporting."},
                       best_dev_ckpt)
            print("  best_dev_model_only.pt updated (P@1={:.4f})".format(val_p1))

    # Final dev predictions
    print("\nSaving final dev predictions...", flush=True)
    y_true_v, y_pred_v, _ = predict_epoch_loader(
        model, val_loader, device, label_ids, label_mask)
    save_predictions(y_true_v, y_pred_v, label_order,
                     os.path.join(out_dir, "predictions_val.json"))

    # Save per-label metrics on dev
    from lhrm_baseline.metrics import compute_per_label_metrics as _cpl
    per_label = _cpl(y_true_v, y_pred_v, label_names=label_order, threshold=0.5)
    with open(os.path.join(out_dir, "val_per_label_metrics.json"), "w") as f:
        json.dump(per_label, f, indent=2)

    with open(os.path.join(out_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Final epoch checkpoint (kept for HRRM fitting; labelled explicitly)
    final_ckpt = os.path.join(ckpt_dir, "final_epoch.pt")
    torch.save(ckpt_payload, final_ckpt)  # ckpt_payload still in scope from last epoch

    write_sha256sums(out_dir)
    print("Training complete. Output:", out_dir)
    print("Next: python run_baseline.py hrrm_fit --predictions",
          os.path.join(out_dir, "predictions_val.json"))
    print("Final test evaluation requires explicit authorization via cmd_test_eval.")


def cmd_test_eval(args):
    """
    A1: Final test evaluation — SEPARATE, EXPLICITLY AUTHORIZED command.
    Requires --checkpoint pointing to the final_epoch.pt from a completed RUN_A.
    Loads test split ONCE, predicts, saves predictions_test_raw.json.
    This command must NOT be run during training, hyperparameter selection, or HRRM fitting.
    """
    require_gpu()
    import torch, sys
    cfg = load_yaml_config(args.config)
    run_id = getattr(args, "run_id", "RUN_unnamed")
    ckpt_path = getattr(args, "checkpoint", None)
    if not ckpt_path or not os.path.exists(ckpt_path):
        print("ERROR: --checkpoint must point to a valid final_epoch.pt file.")
        print("Run training to completion first, then run test_eval once.")
        sys.exit(1)

    from lhrm_baseline.data import load_processed, MultiLabelDataset, collate_fn, tokenize_labels
    from lhrm_baseline.labels import build_all_label_texts
    from lhrm_baseline.model import LHRMModel, build_bert_model
    from lhrm_baseline.train import predict_epoch_loader, save_predictions
    from lhrm_baseline.metrics import compute_all_metrics
    from transformers import BertTokenizer
    from torch.utils.data import DataLoader

    device = torch.device("cuda")
    data_dir = cfg.get("data_dir", "data/processed/AAPD")
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(ROOT, data_dir)

    _, _, label_order = load_processed(data_dir, "train")  # for label_order only
    test_texts, test_labels, _ = load_processed(data_dir, "test")
    print("Test docs:", len(test_texts), "(loading ONCE for final eval)")

    bert_name = cfg.get("bert_model", "bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    defs_path = cfg.get("defs_path", "Label_Definition/arxiv_labels_definitions.json")
    if not os.path.isabs(defs_path):
        defs_path = os.path.join(ROOT, defs_path)
    label_texts_list = build_all_label_texts(
        label_order, defs_path, dataset=cfg.get("dataset", "aapd"))

    max_len = int(cfg.get("max_len", 512))
    batch_sz = int(cfg.get("batch_size", 8))
    test_ds = MultiLabelDataset(test_texts, test_labels, tokenizer, max_len)
    test_loader = DataLoader(test_ds, batch_sz, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    label_ids, label_mask = tokenize_labels(label_texts_list, tokenizer, max_len=64, device=device)
    bert  = build_bert_model(bert_name, device)
    model = LHRMModel(bert, len(label_order), variant=cfg.get("architecture", "L2")).to(device)
    ckpt_data = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt_data.get("model_state", ckpt_data.get("model_state_dict")))
    model.eval()

    out_dir = os.path.join(ROOT, "experiments", run_id)
    os.makedirs(out_dir, exist_ok=True)
    pred_path = os.path.join(out_dir, "predictions_test_raw.json")
    if os.path.exists(pred_path):
        print("ERROR: predictions_test_raw.json already exists. Refusing to overwrite.")
        print("Test evaluation must run exactly once.")
        sys.exit(1)

    y_true_t, y_pred_t, _ = predict_epoch_loader(model, test_loader, device, label_ids, label_mask)
    save_predictions(y_true_t, y_pred_t, label_order, pred_path)
    test_m = compute_all_metrics(y_true_t, y_pred_t, [1, 3, 5], threshold=0.5)
    print("TEST METRICS (preliminary, threshold=0.5):")
    for k, v in test_m.items():
        print("  {}={:.4f}".format(k, v))
    with open(os.path.join(out_dir, "test_metrics_preliminary.json"), "w") as f:
        json.dump(test_m, f, indent=2)
    print("Test predictions saved:", pred_path)


def _resolve_label_order(args):
    """
    A7: Dataset-aware label order resolution for evaluate/hrrm_fit/hrrm_apply.
    Priority:
      1. labels.json next to the predictions file
      2. labels.json in data_dir from resolved_config.json in the experiment dir
      3. AAPD canonical locations
      Never silently falls back to Reuters.
    """
    pred_dir = os.path.dirname(getattr(args, "predictions", ""))
    lj = os.path.join(pred_dir, "labels.json")
    if os.path.exists(lj):
        with open(lj) as f:
            return json.load(f)
    rc = os.path.join(pred_dir, "resolved_config.json")
    if os.path.exists(rc):
        rc_data = json.load(open(rc))
        data_dir = rc_data.get("_data_dir") or rc_data.get("data_dir", "")
        if data_dir:
            if not os.path.isabs(data_dir):
                data_dir = os.path.join(ROOT, data_dir)
            lj2 = os.path.join(data_dir, "labels.json")
            if os.path.exists(lj2):
                with open(lj2) as f:
                    return json.load(f)
    for candidate in [
        os.path.join(ROOT, "data", "processed", "AAPD", "labels.json"),
        os.path.join(ROOT, "Data", "processed", "AAPD", "labels.json"),
    ]:
        if os.path.exists(candidate):
            print("NOTE: using AAPD canonical label order:", candidate)
            with open(candidate) as f:
                return json.load(f)
    raise FileNotFoundError(
        "Cannot resolve label order. Provide labels.json next to predictions "
        "or ensure data_dir is configured in resolved_config.json.")


def cmd_evaluate(args):
    from lhrm_baseline.evaluate import evaluate_predictions
    label_order = _resolve_label_order(args)
    result = evaluate_predictions(args.predictions, label_order, k_values=[1, 3, 5],
                                  n_bootstrap=1000, seed=42, verbose=True)
    out = args.predictions.replace(".json", "_eval.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print("Evaluation saved:", out)


def cmd_hrrm_fit(args):
    from lhrm_baseline.hrrm import HRRM, build_hrrm_features
    # A7: dataset-aware label order resolution
    label_order = _resolve_label_order(args)
    X, y, _, n_docs = build_hrrm_features(args.predictions, label_order)
    print("Docs: {}, Feature rows: {}".format(n_docs, len(X)))
    hrrm = HRRM(seed=42)
    hrrm.fit(X, y)
    hrrm_path = os.path.join(os.path.dirname(args.predictions), "hrrm.joblib")
    hrrm.save(hrrm_path)
    print("HRRM saved:", hrrm_path)


def cmd_hrrm_apply(args):
    from lhrm_baseline.hrrm import HRRM
    from lhrm_baseline.evaluate import evaluate_predictions
    # A7: dataset-aware label order resolution
    label_order = _resolve_label_order(args)
    hrrm = HRRM.load(args.hrrm)
    out_path = args.predictions.replace("_raw.json", "_hrrm.json")
    hrrm.apply_to_predictions(args.predictions, label_order, out_path)
    print("=== BASE MODEL ===")
    evaluate_predictions(args.predictions, label_order, verbose=True)
    print("=== HRRM ===")
    evaluate_predictions(out_path, label_order, verbose=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LHRM Baseline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("prepare", help="Prepare data (LEVEL 1: CPU)")
    p.add_argument("--config", default="configs/reuters_baseline.yaml")

    p = sub.add_parser("run0", help="Architecture screening smoke run (LEVEL 3: GPU)")
    p.add_argument("--config", default="configs/reuters_baseline.yaml")
    p.add_argument("--run_id", default="RUN_0")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--candidates", default="L,L2")
    p.add_argument("--extended", action="store_true")

    p = sub.add_parser("train", help="Train LHRM (LEVEL 3: GPU)")
    p.add_argument("--config", default="configs/reuters_baseline.yaml")
    p.add_argument("--run_id", default="RUN_A")
    p.add_argument("--variant", type=str, default=None,
                   help="L or L2. Overrides architecture in config. Required if config has TBD.")
    p.add_argument("--resume_from", type=str, default=None)

    p = sub.add_parser("test_eval",
                       help="Final test evaluation (A1: separate, explicitly authorized)")
    p.add_argument("--config", required=True)
    p.add_argument("--run_id", required=True)
    p.add_argument("--checkpoint", required=True,
                   help="Path to final_epoch.pt from completed training run")

    p = sub.add_parser("evaluate", help="Evaluate predictions (LEVEL 1: CPU)")
    p.add_argument("--predictions", required=True)

    p = sub.add_parser("hrrm_fit", help="Fit HRRM (LEVEL 1: CPU)")
    p.add_argument("--predictions", required=True)

    p = sub.add_parser("hrrm_apply", help="Apply HRRM (LEVEL 1: CPU)")
    p.add_argument("--predictions", required=True)
    p.add_argument("--hrrm", required=True)

    p = sub.add_parser("import_artifacts", help="Verify and register GPU-run artifacts (LEVEL 1: CPU)")
    p.add_argument("--path", required=True)

    p = sub.add_parser("freeze", help="Write CONFIG_FREEZE.json (LEVEL 1: CPU)")
    p.add_argument("--config", default="configs/reuters_baseline.yaml")
    p.add_argument("--out", default="configs/final/CONFIG_FREEZE.json")
    p.add_argument("--require_clean", action="store_true")

    args = parser.parse_args()
    dispatch = {
        "prepare": cmd_prepare,
        "run0": cmd_run0,
        "train": cmd_train,
        "test_eval": cmd_test_eval,
        "evaluate": cmd_evaluate,
        "hrrm_fit": cmd_hrrm_fit,
        "hrrm_apply": cmd_hrrm_apply,
        "import_artifacts": cmd_import_artifacts,
        "freeze": cmd_freeze,
    }
    if args.command in dispatch:
        dispatch[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
