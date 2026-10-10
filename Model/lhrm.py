
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LHRM: Label-Aware Hierarchical Ranking Model for Multi-Label Text Classification
- Variants: bert, bert+label, csam_nosa, csam_sa
- BERT-based doc & label representations + CSAM
- Optional HRRM (meta re-ranking) with Gradient Boosting on saved predictions

Usage examples:
---------------
# Train + evaluate and save predictions
python lhrm.py train \
  --train_json /path/processed_train.json \
  --test_json /path/processed_test.json \
  --labels_txt /path/labels.txt \
  --context_json /path/arxiv_labels_definitions_full.json \
  --variant csam_sa \
  --epochs 3 \
  --batch_size 16 \
  --bert_name bert-base-uncased \
  --max_len 128 \
  --out_preds /tmp/base_preds.json

# Run HRRM re-ranking on saved predictions (and compute P@k, nDCG@k)
python lhrm.py rerank \
  --preds_json /tmp/base_preds.json \
  --out_json /tmp/reranked.json

"""
import os
import json
import math
import argparse
import random
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# AMP / Optimizer
try:
    from torch.cuda.amp import autocast, GradScaler
except Exception:
    # Fallback if CUDA/AMP unavailable
    class autocast:
        def __init__(self, enabled=False):
            self.enabled = False
        def __enter__(self): pass
        def __exit__(self, exc_type, exc, tb): pass
    class GradScaler:
        def __init__(self, enabled=False): pass
        def scale(self, loss): return loss
        def step(self, opt): opt.step()
        def update(self): pass

from transformers import BertTokenizer, BertModel
from torch.optim import AdamW

# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_list(x):
    if isinstance(x, list):
        return x
    return [x]

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ----------------------------
# Dataset
# ----------------------------
class CustomDS(Dataset):
    def __init__(self, df: pd.DataFrame, text_key: str, labels_key: str, tokenizer, max_len: int, mlb: MultiLabelBinarizer):
        self.df = df.reset_index(drop=True)
        self.text_key = text_key
        self.labels_key = labels_key
        self.tok = tokenizer
        self.max_len = max_len
        self.mlb = mlb

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        text = str(row[self.text_key])
        labels = ensure_list(row[self.labels_key])
        y = self.mlb.transform([labels])[0]
        enc = self.tok(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attn_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(y, dtype=torch.float),
        }

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = batch[0].keys()
    out = {}
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out

# ----------------------------
# Model components
# ----------------------------
class SimilarityAttention(nn.Module):
    def __init__(self, hidden: int, label_emb: torch.Tensor):
        super().__init__()
        # label_emb: [L, H]
        self.label_emb = nn.Parameter(label_emb.clone(), requires_grad=True)

    def forward(self, doc_emb: torch.Tensor) -> torch.Tensor:
        # doc_emb: [B, H], label_emb: [L, H]
        sim = doc_emb @ self.label_emb.t()        # [B, L]
        w = sim.softmax(dim=1)                    # [B, L]
        return w @ self.label_emb                 # [B, H]

class CSAM(nn.Module):
    def __init__(self, hidden: int, label_emb: torch.Tensor, use_sa: bool = True, heads: int = 4):
        super().__init__()
        self.label_emb = nn.Parameter(label_emb.clone(), requires_grad=True)  # [L, H]
        self.use_sa = use_sa
        if use_sa:
            self.sa = nn.MultiheadAttention(hidden, heads, batch_first=True)
            self.ln = nn.LayerNorm(hidden)

    def forward(self, doc_emb: torch.Tensor) -> torch.Tensor:
        # doc_emb: [B, H]
        L = self.label_emb                         # [L, H]
        if self.use_sa:
            att, _ = self.sa(L.unsqueeze(0), L.unsqueeze(0), L.unsqueeze(0))  # [1, L, H]
            L = self.ln(L + att.squeeze(0))                                   # [L, H]
        w = (doc_emb @ L.t()).softmax(dim=1)        # [B, L]
        return w @ L                                 # [B, H]

class LHRMModel(nn.Module):
    def __init__(self, bert: BertModel, num_labels: int, variant: str, label_emb: torch.Tensor | None):
        super().__init__()
        self.bert = bert
        H = self.bert.config.hidden_size
        self.variant = variant
        if variant == "bert":
            self.agg = None
            in_dim = H
        elif variant == "bert+label":
            assert label_emb is not None, "label_emb is required for bert+label"
            self.agg = SimilarityAttention(H, label_emb)
            in_dim = 2 * H
        elif variant in ("csam_nosa", "csam_sa"):
            assert label_emb is not None, "label_emb is required for CSAM variants"
            use_sa = (variant == "csam_sa")
            self.agg = CSAM(H, label_emb, use_sa=use_sa)
            in_dim = 2 * H
        else:
            raise ValueError(f"Unknown variant: {variant}")
        self.classifier = nn.Linear(in_dim, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.bert(input_ids, attention_mask=attention_mask).pooler_output  # [B,H]
        if getattr(self, "agg", None) is None:
            emb = out
        else:
            agg = self.agg(out)   # [B,H]
            emb = torch.cat([out, agg], dim=1)
        return self.classifier(emb)   # [B, L]

# ----------------------------
# Metrics
# ----------------------------
def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    # y_true: [N, L] (0/1), y_score: [N, L] (scores)
    topk = np.argsort(-y_score, axis=1)[:, :k]  # indices of top-k scores
    hits = np.array([y_true[i, topk[i]].sum() for i in range(len(y_true))], dtype=float)
    # Normalize per-sample by min(k, #true) to avoid over-crediting when few true labels
    denom = np.array([max(1, min(k, int(y_true[i].sum()))) for i in range(len(y_true))], dtype=float)
    return float(np.mean(hits / denom))

def _dcg_at_k(rel: np.ndarray, k: int) -> float:
    rel = rel[:k]
    if len(rel) == 0:
        return 0.0
    denom = np.log2(np.arange(2, k + 2))
    return float((rel / denom).sum())

def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    idx = np.argsort(-y_score, axis=1)
    rel = np.take_along_axis(y_true, idx, axis=1)
    dcg = np.array([_dcg_at_k(rel[i], k) for i in range(len(y_true))], dtype=float)
    ideal = np.array([_dcg_at_k(np.sort(y_true[i])[::-1], k) for i in range(len(y_true))], dtype=float)
    ideal[ideal == 0] = 1.0
    return float(np.mean(dcg / ideal))

# ----------------------------
# Label Embeddings
# ----------------------------
@torch.no_grad()
def build_label_embeddings(label_list: List[str], tokenizer, bert: BertModel, context: Dict[str, str] | None, max_len: int, device: torch.device) -> torch.Tensor:
    """
    Returns: Tensor [L, H]
    Each label embedding = mean over token embeddings for: "label: definition"
    """
    bert.eval()
    embs = []
    for lab in label_list:
        enriched = lab
        if context is not None:
            enriched = f"{lab}: {context.get(lab, '')}".strip()
        enc = tokenizer(
            enriched,
            return_tensors="pt",
            max_length=max_len,
            padding="max_length",
            truncation=True
        ).to(device)
        out = bert(**enc).last_hidden_state.mean(dim=1)  # [1, H]
        embs.append(out.squeeze(0))
    return torch.stack(embs, dim=0)  # [L, H]

# ----------------------------
# Training & Evaluation
# ----------------------------
def train_loop(model, loader, criterion, optimizer, scaler, device, use_amp=True):
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        ids = batch["input_ids"].to(device)
        mask = batch["attn_mask"].to(device)
        y = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=(use_amp and device.type == "cuda")):
            logits = model(ids, mask)
            loss = criterion(logits, y)

        if isinstance(scaler, GradScaler) and (use_amp and device.type == "cuda"):
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total += float(loss.item())
    return total / max(1, len(loader))

@torch.no_grad()
def eval_loop(model, loader, device):
    model.eval()
    preds, trues = [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        ids = batch["input_ids"].to(device)
        mask = batch["attn_mask"].to(device)
        logits = model(ids, mask)
        preds.append(torch.sigmoid(logits).cpu().numpy())
        trues.append(batch["labels"].numpy())
    y_pred = np.vstack(preds)
    y_true = np.vstack(trues)
    return y_true, y_pred

def run_train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_df = pd.read_json(args.train_json)
    test_df = pd.read_json(args.test_json)
    # Normalize label column into list
    for df in (train_df, test_df):
        df[args.labels_key] = df[args.labels_key].apply(ensure_list)

    # Labels
    label_list = [l.strip() for l in open(args.labels_txt, "r", encoding="utf-8") if l.strip()]
    mlb = MultiLabelBinarizer(classes=label_list)
    mlb.fit(train_df[args.labels_key])

    # Tokenizer & BERT
    tokenizer = BertTokenizer.from_pretrained(args.bert_name)
    bert = BertModel.from_pretrained(args.bert_name).to(device)

    # Label context (optional)
    context = None
    if args.context_json and os.path.isfile(args.context_json):
        context = load_json(args.context_json)

    # Build label embeddings for variants that require them
    label_emb = None
    if args.variant != "bert":
        label_emb = build_label_embeddings(label_list, tokenizer, bert, context, args.max_len, device)

    # Datasets / Loaders
    train_loader = DataLoader(
        CustomDS(train_df, args.text_key, args.labels_key, tokenizer, args.max_len, mlb),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        CustomDS(test_df, args.text_key, args.labels_key, tokenizer, args.max_len, mlb),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # Model
    model = LHRMModel(bert, num_labels=len(label_list), variant=args.variant, label_emb=label_emb).to(device)

    # Optim & Loss
    optimizer = AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # Train epochs
    for ep in range(1, args.epochs + 1):
        loss = train_loop(model, train_loader, criterion, optimizer, scaler, device, use_amp=not args.no_amp)
        print(f"Epoch {ep}/{args.epochs} | loss={loss:.4f}")

    # Evaluate
    y_true, y_pred = eval_loop(model, test_loader, device)
    for k in (1, 3, 5):
        print(f"P@{k}: {precision_at_k(y_true, y_pred, k):.4f}")
    for k in (3, 5):
        print(f"nDCG@{k}: {ndcg_at_k(y_true, y_pred, k):.4f}")

    # Save predictions for HRRM
    if args.out_preds:
        # Save per-sample predicted labels & scores + true labels
        data_out = []
        for i in range(y_pred.shape[0]):
            scores = y_pred[i].tolist()
            # Sort labels by score desc
            idx = np.argsort(-y_pred[i])
            pre_label = [label_list[j] for j in idx]
            pre_score = [float(y_pred[i][j]) for j in idx]
            # Gather true labels for the sample
            true_indices = np.where(y_true[i] > 0.5)[0].tolist()
            ture_label = [label_list[j] for j in true_indices]  # keep "ture_label" for backward compatibility
            data_out.append({
                "pre_label": pre_label,
                "pre_score": pre_score,
                "ture_label": ture_label
            })
        save_json(data_out, args.out_preds)
        print(f"Saved predictions to: {args.out_preds}")

def run_rerank(args):
    """
    HRRM: Gradient Boosting regressor to adjust scores and re-rank.
    Input JSON schema per sample supports either keys:
      - 'ture_label' (typo in older dumps) OR 'true_label'
      - 'pre_label', 'pre_score'
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    data = load_json(args.preds_json)

    # Prepare meta-features
    feat_rows = []
    true_labels_list = []
    for sample in data:
        true_labels = sample.get("ture_label", sample.get("true_label", []))
        pre_labels = sample["pre_label"]
        pre_scores = sample["pre_score"]
        true_labels_list.append(true_labels)

        for i, lab in enumerate(pre_labels):
            feat_rows.append({
                "predicted_label": lab,
                "predicted_score": float(pre_scores[i]),
                "rank": i + 1,
                "is_true": 1 if lab in true_labels else 0
            })
    meta_df = pd.DataFrame(feat_rows)

    # Feature matrix
    X = meta_df[["predicted_score", "rank"]].copy()
    y = meta_df["is_true"].astype(float).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=args.seed)
    meta_model = GradientBoostingRegressor(random_state=args.seed)
    meta_model.fit(X_train, y_train)
    y_pred = meta_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Meta-Model MSE: {mse:.4f}")

    # Adjust predictions
    adjusted = []
    for sample in data:
        pre_labels = sample["pre_label"]
        pre_scores = sample["pre_score"]
        features = pd.DataFrame({
            "predicted_score": pre_scores,
            "rank": list(range(1, len(pre_scores) + 1))
        })
        adj_scores = meta_model.predict(features).tolist()
        idx = np.argsort(-np.array(adj_scores))
        adjusted.append({
            "pre_label": [pre_labels[i] for i in idx],
            "pre_score": [float(adj_scores[i]) for i in idx]
        })

    # Compute metrics after re-ranking
    # Build y_true/y_pred matrices in label order
    # First collect full label space
    all_labels = sorted(list({lab for s in data for lab in s["pre_label"]}))
    label2idx = {l: i for i, l in enumerate(all_labels)}

    def build_matrix(samples, key_scores="pre_score", key_labels="pre_label"):
        N = len(samples)
        L = len(all_labels)
        Y = np.zeros((N, L), dtype=float)
        for i, s in enumerate(samples):
            labs = s[key_labels]
            scs = s[key_scores]
            for lab, sc in zip(labs, scs):
                j = label2idx[lab]
                Y[i, j] = float(sc)
        return Y

    Y_pred_base = build_matrix(data)
    Y_pred_adj = build_matrix(adjusted)
    # True matrix
    Y_true = np.zeros_like(Y_pred_base)
    for i, s in enumerate(data):
        true_labs = s.get("ture_label", s.get("true_label", []))
        for lab in true_labs:
            if lab in label2idx:
                Y_true[i, label2idx[lab]] = 1.0

    for name, mat in [("BASE", Y_pred_base), ("HRRM", Y_pred_adj)]:
        print(f"== {name} ==")
        for k in (1, 3, 5):
            print(f"P@{k}: {precision_at_k(Y_true, mat, k):.4f}")
        for k in (3, 5):
            print(f"nDCG@{k}: {ndcg_at_k(Y_true, mat, k):.4f}")

    if args.out_json:
        out = {
            "metrics": {
                "meta_model_mse": float(mse)
            },
            "reranked": adjusted
        }
        save_json(out, args.out_json)
        print(f"Saved reranked results to: {args.out_json}")

# ----------------------------
# CLI
# ----------------------------
def build_parser():
    p = argparse.ArgumentParser(description="LHRM training/eval and HRRM re-ranking")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Train/Eval
    t = sub.add_parser("train", help="Train + evaluate LHRM and (optionally) save predictions")
    t.add_argument("--train_json", type=str, required=True)
    t.add_argument("--test_json", type=str, required=True)
    t.add_argument("--labels_txt", type=str, required=True)
    t.add_argument("--context_json", type=str, default="", help="Optional: label definitions json")
    t.add_argument("--text_key", type=str, default="text")
    t.add_argument("--labels_key", type=str, default="label")
    t.add_argument("--variant", type=str, default="csam_nosa", choices=["bert", "bert+label", "csam_nosa", "csam_sa"])
    t.add_argument("--bert_name", type=str, default="bert-base-uncased")
    t.add_argument("--max_len", type=int, default=128)
    t.add_argument("--batch_size", type=int, default=16)
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--lr", type=float, default=2e-5)
    t.add_argument("--num_workers", type=int, default=2)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--no_amp", action="store_true")
    t.add_argument("--out_preds", type=str, default="", help="Path to save predictions JSON for HRRM")

    # Rerank
    r = sub.add_parser("rerank", help="Run HRRM re-ranking on saved predictions")
    r.add_argument("--preds_json", type=str, required=True)
    r.add_argument("--out_json", type=str, default="")
    r.add_argument("--seed", type=int, default=42)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == "train":
        run_train(args)
    elif args.cmd == "rerank":
        run_rerank(args)
    else:
        raise ValueError("Unknown command")

if __name__ == "__main__":
    main()
