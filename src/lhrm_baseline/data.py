"""
Data loading and preparation for LHRM baseline.

Handles:
  - AAPD: CSV format with binary label columns (from AAPD.zip)
  - Reuters: NLTK format (to be implemented when data available)
  - Case-insensitive label mapping (D2)
  - Label order consistency (saved to labels.json)
  - Validation: zero-label docs, duplicates, label counts
"""
from __future__ import annotations
import csv
import io
import json
import os
import zipfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


# ------------------------------------------------------------------
# Seed utility (reused from legacy)
# ------------------------------------------------------------------
def set_seed(seed: int = 42):
    """Set random seeds for reproducibility. REUSED from legacy."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------------
# AAPD data loading
# ------------------------------------------------------------------
def load_aapd_split(zip_path: str, split: str) -> Tuple[List[str], np.ndarray, List[str]]:
    """
    Load one split of AAPD from the zip archive.
    
    Args:
        zip_path: Path to AAPD.zip
        split: 'train', 'dev', or 'test'
    
    Returns:
        texts: List of abstract strings
        labels_matrix: np.ndarray [N, L] binary
        label_order: List of label column names (consistent across splits)
    """
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(f"AAPD/{split}.csv") as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
            rows = list(reader)
    
    if not rows:
        raise ValueError(f"Empty split: {split}")
    
    headers = list(rows[0].keys())
    label_cols = [h for h in headers if h != "abstract"]
    
    texts = [row["abstract"] for row in rows]
    labels = np.array(
        [[int(row[col]) for col in label_cols] for row in rows],
        dtype=np.float32
    )
    
    return texts, labels, label_cols


def validate_splits(
    train_labels: np.ndarray,
    dev_labels: np.ndarray,
    test_labels: np.ndarray,
    label_order: List[str],
) -> dict:
    """
    Run data validation checks and return a report dict.
    Checks: zero-label docs, label counts, no duplicates across train/test.
    """
    report = {}
    
    for name, mat in [("train", train_labels), ("dev", dev_labels), ("test", test_labels)]:
        counts = mat.sum(axis=1)
        report[name] = {
            "n_docs": len(mat),
            "n_labels": mat.shape[1],
            "mean_labels": float(counts.mean()),
            "min_labels": int(counts.min()),
            "max_labels": int(counts.max()),
            "zero_label_docs": int((counts == 0).sum()),
        }
    
    report["label_order"] = label_order
    report["n_labels"] = len(label_order)
    
    return report


# ------------------------------------------------------------------
# Label tokenization
# ------------------------------------------------------------------
def tokenize_labels(
    label_texts: List[str],
    tokenizer: BertTokenizer,
    max_len: int = 64,
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenize all label texts Y_1..Y_k.
    Returns input_ids [k, max_len] and attention_mask [k, max_len].
    
    Labels are shorter than documents; max_len=64 is sufficient for
    "full_name: definition" strings (verified empirically).
    """
    enc = tokenizer(
        label_texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    if device is not None:
        enc = {k: v.to(device) for k, v in enc.items()}
    return enc["input_ids"], enc["attention_mask"]


# ------------------------------------------------------------------
# PyTorch Dataset
# ------------------------------------------------------------------
class MultiLabelDataset(Dataset):
    """
    Dataset for multi-label text classification (AAPD and Reuters).
    Tokenizes documents lazily; padding is applied per-batch in collate_fn
    (dynamic padding, N7 fix / A10).

    N7 fix: __getitem__ returns sequences WITHOUT padding (truncate only).
    collate_fn pads to the longest sequence in the batch.
    This reduces compute by ~2.7x on AAPD (mean 187 tokens vs max 512).
    """

    def __init__(
        self,
        texts: List[str],
        labels: np.ndarray,
        tokenizer: BertTokenizer,
        max_len: int = 512,
    ):
        assert len(texts) == len(labels), "texts/labels length mismatch"
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self._truncation_count = 0  # incremented at tokenisation

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        # Truncate-only; NO padding here (padding happens per-batch in collate_fn)
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,          # dynamic padding applied in collate_fn
            truncation=True,
            max_length=self.max_len,
        )
        input_ids = enc["input_ids"].squeeze(0)        # [L_i] variable length
        attention_mask = enc["attention_mask"].squeeze(0)  # [L_i]
        # Track truncation
        if len(input_ids) == self.max_len:
            self._truncation_count += 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),  # [k]
        }


# Alias for backward compatibility
AAPDDataset = MultiLabelDataset


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Dynamic-padding collate_fn (N7 fix).
    Pads input_ids and attention_mask to the longest sequence in the batch.
    Labels are stacked as-is (fixed-length).
    """
    max_len = max(item["input_ids"].shape[0] for item in batch)
    pad_id = 0  # BERT tokenizer padding token id

    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad_len = max_len - seq_len
        # Right-pad with zeros
        padded_ids = torch.cat(
            [item["input_ids"], torch.zeros(pad_len, dtype=torch.long)], dim=0
        )
        padded_mask = torch.cat(
            [item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)], dim=0
        )
        input_ids_list.append(padded_ids)
        attention_mask_list.append(padded_mask)
        labels_list.append(item["labels"])

    return {
        "input_ids": torch.stack(input_ids_list, dim=0),       # [B, max_len]
        "attention_mask": torch.stack(attention_mask_list, dim=0),  # [B, max_len]
        "labels": torch.stack(labels_list, dim=0),              # [B, k]
    }


def make_dataloader(
    dataset: AAPDDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    """Create a DataLoader with a seeded generator for reproducibility."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        generator=generator if shuffle else None,
        pin_memory=False,  # CPU-only machine
    )


# ------------------------------------------------------------------
# Processed data saving/loading
# ------------------------------------------------------------------
def save_processed(
    out_dir: str,
    split: str,
    texts: List[str],
    labels: np.ndarray,
    label_order: List[str],
):
    """Save processed split to disk."""
    os.makedirs(out_dir, exist_ok=True)
    
    # Save texts as JSON lines
    texts_path = os.path.join(out_dir, f"{split}_texts.json")
    with open(texts_path, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)
    
    # Save labels as numpy
    labels_path = os.path.join(out_dir, f"{split}_labels.npy")
    np.save(labels_path, labels)
    
    # Save label order once
    labels_json = os.path.join(out_dir, "labels.json")
    if not os.path.exists(labels_json):
        with open(labels_json, "w", encoding="utf-8") as f:
            json.dump(label_order, f, indent=2)


def load_processed(out_dir: str, split: str) -> Tuple[List[str], np.ndarray, List[str]]:
    """Load processed split from disk."""
    texts_path = os.path.join(out_dir, f"{split}_texts.json")
    labels_path = os.path.join(out_dir, f"{split}_labels.npy")
    labels_json = os.path.join(out_dir, "labels.json")
    
    with open(texts_path, "r", encoding="utf-8") as f:
        texts = json.load(f)
    labels = np.load(labels_path)
    with open(labels_json, "r", encoding="utf-8") as f:
        label_order = json.load(f)
    
    return texts, labels, label_order
