"""
AAPD Data Preparation Script.
Extracts AAPD.zip, applies case-insensitive label mapping,
builds processed train/dev/test splits, saves DATA_MANIFEST.json.

Usage:
  python scripts/prepare_aapd.py
  python scripts/prepare_aapd.py --label_mapping_file data/label_mapping.json

F1 fix: sys imported before use; unterminated f-string fixed.

LEVEL 1 (LOCAL): This script runs on CPU, no GPU required.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import sys
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from lhrm_baseline.labels import build_all_label_texts


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Prepare AAPD data")
    parser.add_argument("--zip_path", default=os.path.join(ROOT, "Data", "AAPD.zip"))
    parser.add_argument("--out_dir", default=os.path.join(ROOT, "data", "processed", "AAPD"))
    parser.add_argument(
        "--defs_path",
        default=os.path.join(ROOT, "Label_Definition", "arxiv_labels_definitions.json"),
    )
    parser.add_argument(
        "--label_mapping_file",
        default=None,
        help="Optional JSON mapping CSV col -> authoritative label code (fixes D3)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load label mapping (D3 fix if available)
    label_mapping_status = "UNVERIFIED_LABEL_MAPPING"
    col_to_label = None
    if args.label_mapping_file and os.path.exists(args.label_mapping_file):
        with open(args.label_mapping_file) as f:
            col_to_label = json.load(f)
        label_mapping_status = col_to_label.get("_mapping_status", "VERIFIED_LABEL_MAPPING")
        print("Using label mapping from: " + args.label_mapping_file)
        print("  Mapping status: " + label_mapping_status)
    else:
        print("WARNING: No label_mapping_file provided.")
        print("  D3 risk: label column headers may not match column contents.")
        print("  Results will be marked UNVERIFIED_LABEL_MAPPING.")

    # Extract and load splits
    print("Loading from: " + args.zip_path)
    splits_data = {}
    header_label_cols = None

    with zipfile.ZipFile(args.zip_path, "r") as z:
        for split in ["train", "dev", "test"]:
            with z.open("AAPD/" + split + ".csv") as f:
                reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
                rows = list(reader)

            headers = list(rows[0].keys())
            label_cols = [h for h in headers if h != "abstract"]

            if header_label_cols is None:
                header_label_cols = label_cols
            else:
                assert header_label_cols == label_cols, (
                    "Label cols differ between splits!"
                )

            texts = [row["abstract"] for row in rows]
            labels = np.array(
                [[int(row[col]) for col in label_cols] for row in rows],
                dtype=np.float32,
            )
            splits_data[split] = (texts, labels)
            print("  " + split + ": " + str(len(texts)) + " docs, " + str(labels.shape[1]) + " labels")

    # Determine label order
    if col_to_label is not None:
        label_order = [
            col_to_label.get(col, col)
            for col in header_label_cols
        ]
    else:
        label_order = list(header_label_cols)

    # Build label texts
    print("Building label texts...")
    try:
        label_texts = build_all_label_texts(label_order, args.defs_path, dataset="aapd")
        print("  All " + str(len(label_texts)) + " label texts built successfully.")
        for i, (label, text) in enumerate(zip(label_order, label_texts)):
            print("  [" + str(i).zfill(2) + "] " + label + ": " + text[:60] + "...")
    except AssertionError as e:
        print("ERROR building label texts: " + str(e))
        raise

    # Validate
    for split, (texts, labels) in splits_data.items():
        counts = labels.sum(axis=1)
        zero = int((counts == 0).sum())
        print("  " + split + " validation: mean=" + f"{counts.mean():.3f}" +
              ", min=" + str(int(counts.min())) + ", zero-label=" + str(zero))
        assert zero == 0, split + " has " + str(zero) + " zero-label documents"

    # Save processed data
    for split, (texts, labels) in splits_data.items():
        texts_path = os.path.join(args.out_dir, split + "_texts.json")
        labels_path = os.path.join(args.out_dir, split + "_labels.npy")
        with open(texts_path, "w", encoding="utf-8") as f:
            json.dump(texts, f, ensure_ascii=False)
        np.save(labels_path, labels)
        print("  Saved " + split + ": " + os.path.relpath(texts_path, ROOT))

    labels_json = os.path.join(args.out_dir, "labels.json")
    with open(labels_json, "w") as f:
        json.dump(label_order, f, indent=2)

    label_texts_json = os.path.join(args.out_dir, "label_texts.json")
    with open(label_texts_json, "w", encoding="utf-8") as f:
        json.dump(label_texts, f, indent=2, ensure_ascii=False)

    # DATA_MANIFEST.json
    manifest = {
        "dataset": "AAPD",
        "label_mapping_status": label_mapping_status,
        "splits": {},
        "label_order": label_order,
        "n_labels": len(label_order),
        "zip_sha256": sha256(args.zip_path),
        "defs_path": args.defs_path,
        "note_D1": "train=53840+dev=1000 matches paper training 54840; test=1000",
        "note_D2": "Labels case-insensitive mapped; JSON keys are original-case",
        "note_D3": (
            "Label mapping status: " + label_mapping_status
            + ". cmp-lg and cond-mat.dis-nn are byte-identical columns (freq=17152). "
            "math.co and math.it are byte-identical (freq=875). "
            "See docs/phase1/aapd_label_mapping_verification.md"
        ),
        "note_D4": (
            "A10 max_len decided from BERT token-length percentiles on train only. "
            "See docs/phase1/token_length_stats.json."
        ),
    }

    for split, (texts, labels) in splits_data.items():
        counts = labels.sum(axis=1)
        manifest["splits"][split] = {
            "n_docs": len(texts),
            "n_labels": labels.shape[1],
            "mean_labels": float(counts.mean()),
            "min_labels": int(counts.min()),
            "max_labels": int(counts.max()),
            "zero_label_docs": 0,
        }

    manifest_path = os.path.join(ROOT, "data", "DATA_MANIFEST.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print("DATA_MANIFEST.json saved to: " + os.path.relpath(manifest_path, ROOT))
    print("Label mapping status: " + label_mapping_status)
    if label_mapping_status == "UNVERIFIED_LABEL_MAPPING":
        print("ACTION REQUIRED: Confirm AAPD label order.")
        print("  See docs/phase1/aapd_label_mapping_verification.md")


if __name__ == "__main__":
    main()
