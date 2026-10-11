"""
scripts/probe_csam_trained_checkpoint.py

Authentic Trained-Checkpoint CSAM Gradient & Architectural Micro-Probe:
Evaluates real post-training representations (h_T, h_L) and gradient flows
from saved step-300 checkpoint states.

Key Guarantees:
1. Strictly requires real checkpoint file (step_300.pt) and real BERT model weights.
   Never silently falls back to random/synthetic representations.
2. Validates checkpoint schema and loads with strict state-dict enforcement (strict=True).
3. Evaluates in eval mode (dropout=0) with gradients enabled on authentic AAPD Train data.
4. Performs algebraic logit decomposition into document-dependent, static SA, and bias terms.
5. Computes Focal Loss and BCE gradients on the SAME computational graph and logits.
6. Does NOT perform any optimizer updates (parameters remain frozen).
7. Exports and validates lightweight model-only checkpoints (stripping optimizer state).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lhrm_baseline.model import LHRMModel, CSAM_L, get_cls_embedding
from src.lhrm_baseline.losses import FocalLoss, BCELoss
from scripts.probe_csam_gradient_paths import decompose_csam_logits, compute_analytical_attention_grad_factor


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def export_model_only_checkpoint(
    full_ckpt_path: str | Path,
    out_path: str | Path,
    bert_model: Optional[BertModel] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Extracts model_state from full checkpoint (omitting ~900MB optimizer state)
    and verifies strict round-trip loadability.
    """
    full_p = Path(full_ckpt_path)
    out_p = Path(out_path)
    if not full_p.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {full_p}")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(full_p, map_location="cpu")
    if "model_state" not in ckpt and "model_state_dict" not in ckpt:
        raise KeyError(f"Checkpoint at {full_p} missing 'model_state' key")

    model_state = ckpt.get("model_state", ckpt.get("model_state_dict"))
    step = ckpt.get("step", 300)
    variant = ckpt.get("variant", "unknown")
    cfg = ckpt.get("config", {})

    export_payload = {
        "step": step,
        "variant": variant,
        "model_state": model_state,
        "config": cfg,
        "export_type": "MODEL_ONLY",
        "description": "Full model state (BERT + CSAM), optimizer state stripped for lightweight download",
    }

    out_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(export_payload, out_p)
    file_sha = sha256_file(out_p)
    file_size_mb = out_p.stat().st_size / (1024 * 1024)

    # Round-trip verification if bert_model is provided
    roundtrip_ok = False
    if bert_model is not None:
        num_labels = 54
        test_model = LHRMModel(bert_model, num_labels, variant=variant).to(device)
        loaded_ckpt = torch.load(out_p, map_location=device)
        test_model.load_state_dict(loaded_ckpt["model_state"], strict=True)
        roundtrip_ok = True

    return {
        "file_name": out_p.name,
        "file_path": str(out_p.resolve()),
        "file_size_mb": round(file_size_mb, 2),
        "sha256": file_sha,
        "step": step,
        "variant": variant,
        "roundtrip_verified": roundtrip_ok,
    }


def probe_single_trained_variant(
    model: LHRMModel,
    variant_name: str,
    doc_input_ids: torch.Tensor,
    doc_attention_mask: torch.Tensor,
    label_input_ids: torch.Tensor,
    label_attention_mask: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Executes post-training gradient and logit decomposition probe on an authentic trained model.
    Runs both Focal Loss and BCE on the SAME representations and computational graph.
    No optimizer updates are performed.
    """
    model.eval()  # Disable dropout for deterministic representations
    B, seq_len = doc_input_ids.shape
    k = label_input_ids.shape[0]
    H = model.bert.config.hidden_size

    # 1. Forward through BERT encoders with gradient tracking
    doc_input_ids = doc_input_ids.to(device)
    doc_attention_mask = doc_attention_mask.to(device)
    label_input_ids = label_input_ids.to(device)
    label_attention_mask = label_attention_mask.to(device)
    targets = targets.to(device)

    # Document encoding
    doc_output = model.bert(doc_input_ids, attention_mask=doc_attention_mask)
    h_T = get_cls_embedding(doc_output)  # [B, H]
    h_T.retain_grad()

    # Label encoding
    label_encoder = model.separate_label_encoder if model.separate_label_encoder else model.bert
    label_output = label_encoder(label_input_ids, attention_mask=label_attention_mask)
    h_L = get_cls_embedding(label_output)  # [k, H]
    h_L.retain_grad()

    # 2. Decompose CSAM logits algebraically
    logits, doc_term, label_sa_term, bias_val, alpha = decompose_csam_logits(model.csam, h_T, h_L)

    # Algebraic reconstruction validation
    recon = doc_term + (label_sa_term if label_sa_term is not None else 0.0) + bias_val
    max_recon_error = (logits - recon).abs().max().item()
    if max_recon_error > 1e-4:
        raise ValueError(f"Algebraic reconstruction error too high: {max_recon_error}")

    # Attention distribution statistics
    max_alpha = alpha.max(dim=1).values.mean().item()
    alpha_entropy = -(alpha * (alpha + 1e-12).log()).sum(dim=1).mean().item()
    uniform_entropy = math.log(k)
    alpha_deriv_factor = compute_analytical_attention_grad_factor(alpha, H)

    # Logit component magnitudes and variance
    doc_term_abs_mean = doc_term.abs().mean().item()
    doc_term_std_docs = doc_term.std(dim=0).mean().item()  # Variation across documents
    doc_term_std_labels = doc_term.std(dim=1).mean().item()  # Variation across labels

    if label_sa_term is not None:
        sa_term_abs_mean = label_sa_term.abs().mean().item()
        sa_term_std_labels = label_sa_term[0].std().item()
        sa_to_doc_ratio = sa_term_abs_mean / (doc_term_abs_mean + 1e-12)
    else:
        sa_term_abs_mean = 0.0
        sa_term_std_labels = 0.0
        sa_to_doc_ratio = 0.0

    # 3. Controlled Loss Backward Pass: Focal Loss
    model.zero_grad()
    if h_T.grad is not None:
        h_T.grad.zero_()
    if h_L.grad is not None:
        h_L.grad.zero_()

    focal_fn = FocalLoss(gamma=2.0, alpha=0.25, reduction="mean")
    loss_focal = focal_fn(logits, targets)
    loss_focal.backward(retain_graph=True)

    focal_grads = {
        "h_T_norm": torch.norm(h_T.grad).item(),
        "h_T_sample_norm_mean": float(torch.norm(h_T.grad, dim=1).mean().item()),
        "h_T_sample_norm_std": float(torch.norm(h_T.grad, dim=1).std().item()),
        "h_L_norm": torch.norm(h_L.grad).item(),
        "h_L_label_norm_mean": float(torch.norm(h_L.grad, dim=1).mean().item()),
        "h_L_label_norm_std": float(torch.norm(h_L.grad, dim=1).std().item()),
        "W_norm": torch.norm(model.csam.W.weight.grad).item(),
        "b_norm": torch.norm(model.csam.W.bias.grad).item(),
        "is_finite": bool(
            torch.isfinite(h_T.grad).all().item()
            and torch.isfinite(h_L.grad).all().item()
            and torch.isfinite(model.csam.W.weight.grad).all().item()
            and torch.isfinite(model.csam.W.bias.grad).all().item()
        ),
    }

    # 4. Controlled Loss Backward Pass: BCE on SAME logits and representations
    model.zero_grad()
    h_T.grad.zero_()
    h_L.grad.zero_()

    bce_fn = BCELoss()
    loss_bce = bce_fn(logits, targets)
    loss_bce.backward()

    bce_grads = {
        "h_T_norm": torch.norm(h_T.grad).item(),
        "h_T_sample_norm_mean": float(torch.norm(h_T.grad, dim=1).mean().item()),
        "h_T_sample_norm_std": float(torch.norm(h_T.grad, dim=1).std().item()),
        "h_L_norm": torch.norm(h_L.grad).item(),
        "h_L_label_norm_mean": float(torch.norm(h_L.grad, dim=1).mean().item()),
        "h_L_label_norm_std": float(torch.norm(h_L.grad, dim=1).std().item()),
        "W_norm": torch.norm(model.csam.W.weight.grad).item(),
        "b_norm": torch.norm(model.csam.W.bias.grad).item(),
        "is_finite": bool(
            torch.isfinite(h_T.grad).all().item()
            and torch.isfinite(h_L.grad).all().item()
            and torch.isfinite(model.csam.W.weight.grad).all().item()
            and torch.isfinite(model.csam.W.bias.grad).all().item()
        ),
    }

    # Clear gradients to leave model unmodified
    model.zero_grad()

    return {
        "variant": variant_name,
        "batch_size": B,
        "hidden_dim": H,
        "num_labels": k,
        "max_algebraic_reconstruction_error": max_recon_error,
        "components": {
            "doc_term_abs_mean": doc_term_abs_mean,
            "doc_term_std_across_docs": doc_term_std_docs,
            "doc_term_std_across_labels": doc_term_std_labels,
            "sa_term_abs_mean": sa_term_abs_mean,
            "sa_term_std_across_labels": sa_term_std_labels,
            "sa_to_doc_ratio": sa_to_doc_ratio,
            "bias_value": bias_val,
        },
        "attention": {
            "max_alpha_mean": max_alpha,
            "alpha_entropy_mean": alpha_entropy,
            "uniform_entropy": uniform_entropy,
            "entropy_ratio_to_uniform": alpha_entropy / uniform_entropy,
            "alpha_deriv_factor": alpha_deriv_factor,
        },
        "losses": {
            "focal_loss_value": loss_focal.item(),
            "bce_loss_value": loss_bce.item(),
        },
        "focal_gradients": focal_grads,
        "bce_gradients": bce_grads,
    }


def run_trained_checkpoint_diagnostic_suite(
    checkpoint_dir: str | Path,
    bert_path: str | Path,
    data_dir: str | Path,
    label_defs_path: str | Path,
    batch_size: int = 8,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Orchestrates the complete post-training diagnostic probe across L2 and L2_NO_SA.
    Fails closed if checkpoints are missing or unreadable.
    """
    ckpt_base = Path(checkpoint_dir)
    data_base = Path(data_dir)
    defs_file = Path(label_defs_path)
    bert_p = Path(bert_path)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Verify input paths
    if not bert_p.is_dir():
        raise FileNotFoundError(f"BERT weights directory not found: {bert_p}")
    if not defs_file.is_file():
        raise FileNotFoundError(f"Label definitions file not found: {defs_file}")
    train_texts_file = data_base / "train_texts.json"
    train_labels_file = data_base / "train_labels.npy"
    labels_file = data_base / "labels.json"
    if not (train_texts_file.is_file() and train_labels_file.is_file() and labels_file.is_file()):
        raise FileNotFoundError(f"Missing processed AAPD data files in {data_base}")

    # Load canonical data batch
    with open(train_texts_file, "r", encoding="utf-8") as f:
        all_train_texts = json.load(f)
    train_labels_matrix = np.load(train_labels_file)
    with open(labels_file, "r", encoding="utf-8") as f:
        canonical_labels = json.load(f)

    batch_texts = all_train_texts[:batch_size]
    batch_targets = torch.tensor(train_labels_matrix[:batch_size, :len(canonical_labels)], dtype=torch.float32)

    # Load tokenizer and label definitions
    tokenizer = BertTokenizer.from_pretrained(str(bert_p), local_files_only=True)
    from src.lhrm_baseline.labels import build_all_label_texts
    from src.lhrm_baseline.data import tokenize_labels
    label_texts = build_all_label_texts(canonical_labels, str(defs_file), dataset="aapd")

    # Tokenize inputs - labels use identical tokenize_labels(..., max_len=64) as run_baseline.py::cmd_run0
    doc_enc = tokenizer(batch_texts, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
    lbl_ids, lbl_mask = tokenize_labels(label_texts, tokenizer, max_len=64, device=torch.device("cpu"))

    doc_ids = doc_enc["input_ids"]
    doc_mask = doc_enc["attention_mask"]

    candidates = ["L2", "L2_NO_SA"]
    probe_results = {}
    checkpoint_manifest = {}

    for cand in candidates:
        ckpt_path = ckpt_base / f"variant_{cand}" / "checkpoints" / "step_300.pt"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"CRITICAL: Required trained checkpoint step_300.pt missing for {cand}: {ckpt_path}")

        ckpt_sha = sha256_file(ckpt_path)
        ckpt_size_mb = ckpt_path.stat().st_size / (1024 * 1024)

        # 1. Load full checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu")
        assert ckpt.get("step") == 300, f"Expected step 300, got {ckpt.get('step')}"
        assert ckpt.get("variant") == cand, f"Expected variant {cand}, got {ckpt.get('variant')}"

        checkpoint_manifest[cand] = {
            "candidate": cand,
            "full_checkpoint_path": str(ckpt_path.resolve()),
            "full_checkpoint_size_mb": round(ckpt_size_mb, 2),
            "full_checkpoint_sha256": ckpt_sha,
            "step": ckpt["step"],
            "keys_present": list(ckpt.keys()),
        }

        # 2. Build model and load trained state with strict verification
        bert = BertModel.from_pretrained(str(bert_p), local_files_only=True)
        model = LHRMModel(bert, len(canonical_labels), variant=cand).to(device)
        model.load_state_dict(ckpt["model_state"], strict=True)
        print(f"Loaded trained state for variant {cand} (strict=True): PASS", flush=True)

        # 3. Execute diagnostic probe
        res = probe_single_trained_variant(
            model=model,
            variant_name=cand,
            doc_input_ids=doc_ids,
            doc_attention_mask=doc_mask,
            label_input_ids=lbl_ids,
            label_attention_mask=lbl_mask,
            targets=batch_targets,
            device=device,
        )
        probe_results[cand] = res

    suite_output = {
        "experiment_purpose": "DR3_TRAINED_CHECKPOINT_GRADIENT_MICRO_PROBE",
        "device": str(device),
        "batch_size": batch_size,
        "input_provenance": {
            "train_texts_file": str(train_texts_file),
            "train_labels_file": str(train_labels_file),
            "bert_path": str(bert_p),
            "active_labels_mean_in_batch": float(batch_targets.sum(dim=1).mean().item()),
        },
        "checkpoint_manifest": checkpoint_manifest,
        "diagnostic_results": probe_results,
    }
    return suite_output


def main():
    parser = argparse.ArgumentParser(description="Trained-Checkpoint CSAM Gradient Probe")
    parser.add_argument("--checkpoint_dir", required=True, help="Base directory containing variant_* folders")
    parser.add_argument("--bert_path", required=True, help="Path to local official BERT weights")
    parser.add_argument("--data_dir", default="data/processed/AAPD")
    parser.add_argument("--label_defs", default="Label_Definition/arxiv_labels_definitions.json")
    parser.add_argument("--output_json", default="DR3_TRAINED_GRADIENT_PROBE_RESULTS.json")
    parser.add_argument("--manifest_json", default="DR3_TRAINED_CHECKPOINT_MANIFEST.json")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    results = run_trained_checkpoint_diagnostic_suite(
        checkpoint_dir=args.checkpoint_dir,
        bert_path=args.bert_path,
        data_dir=args.data_dir,
        label_defs_path=args.label_defs,
        batch_size=args.batch_size,
    )

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(args.manifest_json, "w", encoding="utf-8") as f:
        json.dump(results["checkpoint_manifest"], f, indent=2)

    print("\n" + "=" * 65)
    print("TRAINED CHECKPOINT GRADIENT MICRO-PROBE COMPLETE")
    print("=" * 65)
    for cand in ["L2", "L2_NO_SA"]:
        r = results["diagnostic_results"][cand]
        fg = r["focal_gradients"]
        c = r["components"]
        print(f"[{cand}] Max recon error: {r['max_algebraic_reconstruction_error']:.2e}")
        print(f"  Focal ||grad_h_T||: {fg['h_T_norm']:.4e} | ||grad_W||: {fg['W_norm']:.4e} | ||grad_b||: {fg['b_norm']:.4e}")
        print(f"  Doc term abs mean: {c['doc_term_abs_mean']:.4e} (across docs std: {c['doc_term_std_across_docs']:.4e})")
        if c["sa_to_doc_ratio"] > 0:
            print(f"  SA term abs mean:  {c['sa_term_abs_mean']:.4e} (SA/Doc ratio: {c['sa_to_doc_ratio']:.2f}x)")
    print(f"\nSaved results to {args.output_json} and manifest to {args.manifest_json}")


if __name__ == "__main__":
    main()
