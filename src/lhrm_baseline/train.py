"""
Training loop for LHRM baseline.
Implements paper training config (A15):
  - AdamW, lr=2e-5, weight_decay=0.01
  - batch_size=8, epochs=15 (AAPD) / 10 (Reuters)
  - Per-epoch checkpointing (model + optimizer + RNG state + epoch)
  - Train/val logging
  - AMP if CUDA available (ASSUMED A11)

GPU gate: This module is designed to run on GPU. On CPU it will work
but training will be extremely slow (Level 3 compute).
"""
from __future__ import annotations
import json
import math
import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from .config import LHRMConfig
from .data import make_dataloader, AAPDDataset, tokenize_labels, set_seed
from .losses import get_loss
from .metrics import compute_all_metrics


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scaler,
    epoch: int,
    config: LHRMConfig,
    out_dir: str,
    label_input_ids: torch.Tensor,
    label_attention_mask: torch.Tensor,
    filename: str = None,
):
    """Save model checkpoint with full state for resume support."""
    os.makedirs(out_dir, exist_ok=True)
    fname = filename or f"epoch_{epoch:02d}.pt"
    path = os.path.join(out_dir, fname)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "config": config.to_dict(),
        "label_input_ids": label_input_ids,
        "label_attention_mask": label_attention_mask,
    }, path)
    return path


def load_checkpoint(path: str, model: nn.Module, optimizer, scaler, device: torch.device):
    """Load checkpoint and restore full state. Returns epoch number.
    
    A4: Restores model, optimizer, scaler, torch CPU RNG, torch CUDA RNG,
    numpy RNG, and python random RNG from the checkpoint.
    Resume is epoch-boundary only: exact mid-epoch batch order cannot be
    guaranteed without saving DataLoader permutation state (not implemented).
    The test `test_epoch_boundary_resume_equivalence` verifies loss continuity
    across an epoch boundary (not mid-epoch).
    """
    import random as _py_random
    # weights_only=False: checkpoint contains numpy RNG state (non-trivial global);
    # safe because checkpoints are produced by our own trusted save_checkpoint_v2.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("model_state", {})))
    optimizer.load_state_dict(ckpt.get("optimizer_state_dict", ckpt.get("optimizer_state", {})))
    if scaler is not None:
        sd = ckpt.get("scaler_state_dict") or ckpt.get("scaler_state")
        if sd:
            scaler.load_state_dict(sd)
    if "rng_state" in ckpt:
        torch.set_rng_state(ckpt["rng_state"])
    if "cuda_rng_state" in ckpt and ckpt["cuda_rng_state"] is not None:
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(ckpt["cuda_rng_state"])
    if "numpy_rng_state" in ckpt:
        np.random.set_state(ckpt["numpy_rng_state"])
    if "python_rng_state" in ckpt:
        _py_random.setstate(ckpt["python_rng_state"])
    return ckpt["epoch"]


def train_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer,
    scaler,
    device: torch.device,
    label_input_ids: torch.Tensor,
    label_attention_mask: torch.Tensor,
    use_amp: bool = False,
) -> float:
    """Run one training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    autocast_ctx = torch.amp.autocast("cuda") if (use_amp and device.type == "cuda") else torch.no_grad().__class__()
    
    for batch in tqdm(loader, desc="train", leave=False):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        y = batch["labels"].to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits, _ = model(ids, mask, label_input_ids, label_attention_mask)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, diag = model(ids, mask, label_input_ids, label_attention_mask)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        
        total_loss += float(loss.item())
        n_batches += 1
    
    return total_loss / max(1, n_batches)


@torch.no_grad()
def predict_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    label_input_ids: torch.Tensor,
    label_attention_mask: torch.Tensor,
) -> tuple:
    """Run prediction on a DataLoader. Returns (y_true, y_pred_scores) ndarrays."""
    model.eval()
    all_preds = []
    all_true = []
    all_diags = []
    
    for batch in tqdm(loader, desc="predict", leave=False):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        y = batch["labels"].numpy()
        
        logits, diag = model(ids, mask, label_input_ids, label_attention_mask)
        probs = torch.sigmoid(logits).cpu().numpy()
        
        all_preds.append(probs)
        all_true.append(y)
        all_diags.append(diag)
    
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_true)
    return y_true, y_pred, all_diags


def save_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_order: list,
    out_path: str,
):
    """Save predictions in the standard format for HRRM input."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    preds = []
    for i in range(len(y_pred)):
        true_labels = [label_order[j] for j in range(len(label_order)) if y_true[i, j] > 0.5]
        preds.append({
            "scores": [float(s) for s in y_pred[i]],
            "labels": label_order,
            "true_labels": true_labels,
        })
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2)
    return out_path


def log_metrics(metrics: dict, prefix: str = "", log_file: str = None):
    """Print and optionally save metrics."""
    line = prefix + " | ".join(f"{k}: {v*100:.2f}%" for k, v in metrics.items())
    print(line)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ------------------------------------------------------------------
# Adapter functions required by run_baseline.py CLI
# ------------------------------------------------------------------

def forward_pass(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    label_input_ids: torch.Tensor,
    label_attention_mask: torch.Tensor,
):
    """
    Thin wrapper around model.forward that returns (logits, diag).
    Allows run_baseline.py to call forward_pass(model, ...) without
    knowing the model's internal call signature.
    C11 fix: exported from train.py.
    """
    logits, diag = model(input_ids, attention_mask,
                         label_input_ids, label_attention_mask)
    return logits, diag


def train_epoch_stepped(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer,
    scaler,
    device: torch.device,
    label_input_ids: torch.Tensor,
    label_attention_mask: torch.Tensor,
    use_amp: bool = False,
    log_every: int = 100,
    global_step_offset: int = 0,
    epoch: int = 0,
    total_epochs: int = 0,
    steps_per_epoch: int = 0,
    total_global_steps: int = 0,
    run_id: str = "",
    variant: str = "",
    progress_jsonl_path: str = None,
):
    """
    Run one training epoch with periodic progress logging (A2).
    Returns (mean_loss, n_steps, amp_skips).

    Every `log_every` optimizer steps emits one line to stdout (sys.stdout.flush)
    and appends one JSON record to `progress_jsonl_path` atomically (if given).
    The JSON record is the canonical source for screenshot evidence.

    AMP skips: when GradScaler skips an optimizer step (NaN/Inf gradient),
    the batch is counted in n_steps but the optimizer does not update;
    amp_skips tracks these.
    """
    import contextlib, sys, time as _time
    model.train()
    total_loss = 0.0
    n_steps = 0
    amp_skips = 0
    window_loss = 0.0
    window_steps = 0
    t_epoch_start = _time.time()
    t_window_start = _time.time()

    for batch in loader:
        ids  = batch["input_ids"].to(device)
        msk  = batch["attention_mask"].to(device)
        y    = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        prev_scale = scaler.get_scale() if (scaler is not None) else 1.0

        ctx = (torch.amp.autocast("cuda")
               if (use_amp and device.type == "cuda")
               else contextlib.nullcontext())
        with ctx:
            logits, _ = model(ids, msk, label_input_ids, label_attention_mask)
            loss = criterion(logits, y)

        if use_amp and device.type == "cuda" and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            new_scale = scaler.get_scale()
            if new_scale < prev_scale:  # scale was reduced = step was skipped
                amp_skips += 1
        else:
            loss.backward()
            optimizer.step()

        loss_val = float(loss.item())
        total_loss  += loss_val
        window_loss += loss_val
        n_steps      += 1
        window_steps += 1
        global_step   = global_step_offset + n_steps

        if log_every > 0 and n_steps % log_every == 0:
            t_now   = _time.time()
            elapsed = t_now - t_epoch_start
            w_elapsed = t_now - t_window_start
            speed   = window_steps / max(w_elapsed, 1e-6)
            mean_loss_window = window_loss / max(window_steps, 1)
            remaining_steps  = total_global_steps - global_step if total_global_steps > 0 else 0
            eta_s = remaining_steps / max(speed, 1e-6)
            eta_str = "{:.0f}m{:.0f}s".format(eta_s // 60, eta_s % 60)

            line = (
                "[TRAIN] {} | variant={} | epoch {}/{} | step {}/{} "
                "| global {}/{} | loss_mean_last{}={:.4f} "
                "| elapsed={:.0f}s | speed={:.2f} steps/s "
                "| amp_skips={} | ETA~{} [ESTIMATED]"
            ).format(
                run_id, variant, epoch, total_epochs,
                n_steps, steps_per_epoch,
                global_step, total_global_steps,
                window_steps, mean_loss_window,
                int(elapsed), speed, amp_skips, eta_str
            )
            print(line, flush=True)

            if progress_jsonl_path:
                record = {
                    "type": "TRAIN_PROGRESS",
                    "run_id": run_id, "variant": variant,
                    "epoch": epoch, "total_epochs": total_epochs,
                    "step_in_epoch": n_steps, "steps_per_epoch": steps_per_epoch,
                    "global_step": global_step, "total_global_steps": total_global_steps,
                    "loss_mean_last_window": round(mean_loss_window, 6),
                    "window_steps": window_steps,
                    "elapsed_s": round(elapsed, 2),
                    "speed_steps_per_s": round(speed, 3),
                    "amp_skips": amp_skips,
                    "eta_s_estimated": round(eta_s, 1),
                    "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
                }
                try:
                    with open(progress_jsonl_path, "a", encoding="utf-8") as pf:
                        pf.write(json.dumps(record) + "\n")
                        pf.flush()
                except Exception:
                    pass  # never let logging break training

            # reset window
            window_loss = 0.0
            window_steps = 0
            t_window_start = _time.time()

    return total_loss / max(1, n_steps), n_steps, amp_skips


@torch.no_grad()
def predict_epoch_loader(
    model: nn.Module,
    loader,
    device: torch.device,
    label_input_ids: torch.Tensor,
    label_attention_mask: torch.Tensor,
):
    """
    Alias of predict_epoch() for consistency with run_baseline.py CLI.
    Returns (y_true, y_pred_scores, diagnostics).
    C11 fix: exported from train.py.
    """
    return predict_epoch(model, loader, device, label_input_ids, label_attention_mask)


def save_checkpoint_v2(
    model: nn.Module,
    optimizer,
    scaler,
    epoch: int,
    step: int,
    config: dict,
    out_dir: str,
    label_input_ids: torch.Tensor,
    label_attention_mask: torch.Tensor,
    history: list = None,
    filename: str = None,
):
    """
    Extended checkpoint with step count, training history, and full RNG state (A4).
    Saves: torch CPU RNG, torch CUDA RNG, numpy RNG, python random RNG.
    Written atomically (tmp + rename) so a crash during save never corrupts the checkpoint.
    Resume is epoch-boundary only; exact mid-epoch batch order is NOT guaranteed.
    """
    import numpy as _np
    import random as _py_random
    os.makedirs(out_dir, exist_ok=True)
    fname = filename or "epoch_{:02d}.pt".format(epoch)
    path = os.path.join(out_dir, fname)
    tmp_path = path + ".tmp"
    torch.save({
        "epoch": epoch,
        "step": step,
        "model_state": model.state_dict(),
        "model_state_dict": model.state_dict(),          # legacy key
        "optimizer_state": optimizer.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),  # legacy key
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": (torch.cuda.get_rng_state()
                           if torch.cuda.is_available() else None),
        "numpy_rng_state": _np.random.get_state(),
        "python_rng_state": _py_random.getstate(),  # A4
        "config": config,
        "label_input_ids": label_input_ids,
        "label_attention_mask": label_attention_mask,
        "history": history or [],
    }, tmp_path)
    # Atomic rename
    if os.path.exists(path):
        os.remove(path)
    os.rename(tmp_path, path)
    return path
