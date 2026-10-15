"""
src/lhrm_baseline/inference.py

Client-facing inference demonstration module for LHRM / CSAM models.
Provides:
- predict_abstract: predicts Top-k labels, sigmoid scores, and optional HRRM re-ranking
  for an unlabelled input text (e.g. abstract).
- format_prediction_report: prints a clean client-ready markdown table of predictions.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer

from .hrrm import HRRM, _seeded_rank_order


@torch.no_grad()
def predict_abstract(
    text: str,
    model: nn.Module,
    tokenizer: BertTokenizer,
    label_order: List[str],
    device: torch.device,
    hrrm_model: Optional[HRRM] = None,
    top_k: int = 5,
    max_len: int = 512,
) -> Dict[str, Any]:
    """
    Run inference on a single raw text (e.g., scientific abstract) using a genuine trained checkpoint.
    Does NOT require ground-truth labels.
    
    Returns structured dict with:
    - text_preview: first 150 chars of input text
    - neural_top_k: list of dicts with label, score, and rank
    - hrrm_top_k: list of dicts if hrrm_model is provided
    - metadata: model variant, device, and unverified dataset mapping disclaimer
    """
    model.eval()
    enc = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_len,
        padding="max_length",
        truncation=True,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    
    # Model forward pass (returns logits, diag)
    logits, _ = model(input_ids, attention_mask)
    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # [k]
    
    # Top-k neural predictions
    top_indices = (-probs).argsort()[:top_k]
    neural_top = [
        {
            "rank": rank + 1,
            "label": label_order[idx],
            "score": round(float(probs[idx]), 6),
        }
        for rank, idx in enumerate(top_indices)
    ]
    
    res = {
        "text_preview": text[:150] + ("..." if len(text) > 150 else ""),
        "neural_top_k": neural_top,
        "metadata": {
            "variant": getattr(model, "variant", "UNKNOWN"),
            "num_labels_total": len(label_order),
            "disclaimer": (
                "NOTE: Predictions are mapped to raw AAPD dataset column names. "
                "Semantic validity of the underlying category definitions remains "
                "unverified (UNRESOLVED_LABEL_MAPPING_ACCEPTED_BY_CLIENT_DECISION)."
            ),
        },
    }
    
    if hrrm_model is not None and getattr(hrrm_model, "is_fitted", False):
        k = len(label_order)
        # Compute 1-based ranks
        rank_order = _seeded_rank_order(probs, seed=42)
        ranks = np.empty_like(rank_order)
        ranks[rank_order] = np.arange(1, k + 1)
        X_doc = np.column_stack([probs, ranks.astype(float)])
        adj_scores = hrrm_model.predict(X_doc)
        hrrm_indices = (-adj_scores).argsort()[:top_k]
        hrrm_top = [
            {
                "rank": rank + 1,
                "label": label_order[idx],
                "score": round(float(adj_scores[idx]), 6),
                "original_neural_score": round(float(probs[idx]), 6),
            }
            for rank, idx in enumerate(hrrm_indices)
        ]
        res["hrrm_top_k"] = hrrm_top
        
    return res


def format_prediction_report(pred_dict: Dict[str, Any]) -> str:
    """Format predict_abstract output into clean Markdown for display or export."""
    lines = [
        "### Client Inference Demonstration Report",
        f"**Input Preview**: {pred_dict['text_preview']}",
        f"**Model Variant**: {pred_dict['metadata']['variant']}",
        "",
        "#### Neural Model Predictions (Top 5)",
        "| Rank | Label | Probability |",
        "|:---:|---|---:|",
    ]
    for item in pred_dict["neural_top_k"]:
        lines.append(f"| {item['rank']} | `{item['label']}` | {item['score']:.4f} |")
        
    if "hrrm_top_k" in pred_dict:
        lines.extend([
            "",
            "#### HRRM Re-Ranked Predictions (Top 5)",
            "| Rank | Label | Adjusted Score | Base Score |",
            "|:---:|---|---:|---:|",
        ])
        for item in pred_dict["hrrm_top_k"]:
            lines.append(
                f"| {item['rank']} | `{item['label']}` | {item['score']:.4f} | {item['original_neural_score']:.4f} |"
            )
            
    lines.extend([
        "",
        f"> *Disclaimer*: {pred_dict['metadata']['disclaimer']}",
    ])
    return "\n".join(lines)
