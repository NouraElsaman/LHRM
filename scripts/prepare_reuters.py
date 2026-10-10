"""
Reuters-21578 Data Preparation Script.
Extracts Reuters-21578.zip (Zenodo 10.5281/zenodo.6344750),
builds processed train/dev/test splits, saves DATA_MANIFEST.json.

Usage:
  python scripts/prepare_reuters.py
  python scripts/prepare_reuters.py --zip_path Data/Reuters-21578.zip

LEVEL 1 (LOCAL): This script runs on CPU, no GPU required.

Source data notes (A_R1b):
  Zenodo splits: train=5,827  dev=1,943  test=3,019  (total=10,789)
  Paper reports train=7,769, test=3,019 (ModApte).

  Zenodo data quality issue (D8):
    train/dev overlap: 31 docs
    train/test overlap: 4 docs
    dev/test overlap: 2 docs
  Fix (A_R1b): merge train+dev, remove any doc also in test (by text),
  deduplicate within traindev, then seeded 90/10 resplit.
  Test set is kept unmodified (ground truth holdout).
  Result: traindev_unique ~ 7,683 -> 90/10: ~6,914 train / ~769 val.

  Text column: 'text' (not 'abstract' as in AAPD).
  90 label categories, exact-case match with reuters_labels_definitions.json.
  0 zero-label documents in train and test; may be non-zero in dev (checked).
"""
import argparse
import csv
import hashlib
import io
import json
import os
import random
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
    parser = argparse.ArgumentParser(description="Prepare Reuters-21578 data (A_R1b)")
    parser.add_argument(
        "--zip_path", default=os.path.join(ROOT, "Data", "Reuters-21578.zip")
    )
    parser.add_argument(
        "--out_dir",
        default=os.path.join(ROOT, "data", "processed", "Reuters"),
    )
    parser.add_argument(
        "--defs_path",
        default=os.path.join(ROOT, "Label_Definition", "reuters_labels_definitions.json"),
    )
    parser.add_argument(
        "--resplit_seed", type=int, default=42,
        help="Seed for 90/10 train/val resplit of deduped traindev pool",
    )
    parser.add_argument(
        "--val_fraction", type=float, default=0.1,
        help="Fraction of deduped traindev docs reserved for validation",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load and validate ZIP
    # ------------------------------------------------------------------ #
    if not os.path.exists(args.zip_path):
        print(f"ERROR: Reuters-21578.zip not found at {args.zip_path}")
        print("Download from: https://zenodo.org/records/6344750/files/Reuters-21578.zip")
        sys.exit(1)

    zip_md5 = hashlib.md5(open(args.zip_path, "rb").read()).hexdigest()
    EXPECTED_MD5 = "47e9dc181f1446e742c91d519987531e"
    if zip_md5 != EXPECTED_MD5:
        print(f"WARNING: Reuters-21578.zip MD5 mismatch (got {zip_md5}). Continuing.")
    else:
        print(f"Reuters-21578.zip MD5 verified: {zip_md5}")

    # ------------------------------------------------------------------ #
    # 2. Extract Zenodo splits
    # ------------------------------------------------------------------ #
    zenodo_splits = {}
    with zipfile.ZipFile(args.zip_path) as z:
        available = z.namelist()
        print(f"ZIP contents: {available}")
        for split in ["train", "dev", "test"]:
            fname = f"{split}.csv"
            if fname not in available:
                raise FileNotFoundError(f"{fname} not found in zip (contents: {available})")
            with z.open(fname) as f:
                reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8", errors="replace"))
                headers = reader.fieldnames
                if "text" not in headers:
                    raise ValueError(f"Expected 'text' column in {fname}, got: {headers[:5]}")
                label_cols = [h for h in headers if h != "text"]
                rows = list(reader)
            zenodo_splits[split] = {"rows": rows, "label_cols": label_cols}
            print(f"  Zenodo {split}: {len(rows)} rows, {len(label_cols)} labels")

    label_order = zenodo_splits["train"]["label_cols"]
    assert len(label_order) == 90, f"Expected 90 labels, got {len(label_order)}"
    for split in ["dev", "test"]:
        if zenodo_splits[split]["label_cols"] != label_order:
            raise ValueError(f"Label order mismatch between train and {split}")
    print("Label column order consistent across all splits. OK")

    # ------------------------------------------------------------------ #
    # 3. Cross-split overlap diagnosis (D8)
    # ------------------------------------------------------------------ #
    train_texts = set(r["text"].strip() for r in zenodo_splits["train"]["rows"])
    dev_texts   = set(r["text"].strip() for r in zenodo_splits["dev"]["rows"])
    test_texts  = set(r["text"].strip() for r in zenodo_splits["test"]["rows"])

    train_dev_overlap  = len(train_texts & dev_texts)
    train_test_overlap = len(train_texts & test_texts)
    dev_test_overlap   = len(dev_texts & test_texts)
    print(f"\nZenodo overlap audit:")
    print(f"  train/dev overlap:  {train_dev_overlap}  (D8 violation)")
    print(f"  train/test overlap: {train_test_overlap} (D8 violation)")
    print(f"  dev/test overlap:   {dev_test_overlap}  (D8 violation)")

    # ------------------------------------------------------------------ #
    # 4. A_R1b fix: Merge traindev, remove test-contaminated docs,
    #               deduplicate by text, seeded 90/10 resplit
    # ------------------------------------------------------------------ #
    print("\nApplying A_R1b fix: merge+dedup+remove_test_texts+resplit...")

    # Merge train+dev, deduplicate by text (train order first, preserving all labels)
    seen_texts = set()
    merged_traindev = []
    for r in zenodo_splits["train"]["rows"] + zenodo_splits["dev"]["rows"]:
        t = r["text"].strip()
        if t not in seen_texts:
            seen_texts.add(t)
            merged_traindev.append(r)

    n_after_dedup = len(merged_traindev)
    print(f"  Merged traindev: {len(zenodo_splits['train']['rows'])+len(zenodo_splits['dev']['rows'])} -> "
          f"{n_after_dedup} (removed {len(zenodo_splits['train']['rows'])+len(zenodo_splits['dev']['rows'])-n_after_dedup} dups)")

    # Remove any docs also in test (by text)
    merged_traindev_clean = [r for r in merged_traindev if r["text"].strip() not in test_texts]
    n_test_removed = n_after_dedup - len(merged_traindev_clean)
    print(f"  Removed {n_test_removed} traindev docs that also appear in test")
    print(f"  Clean traindev pool: {len(merged_traindev_clean)} docs")

    # Seeded 90/10 resplit
    rng = random.Random(args.resplit_seed)
    indices = list(range(len(merged_traindev_clean)))
    rng.shuffle(indices)
    n_val = max(1, int(len(indices) * args.val_fraction))
    val_indices   = sorted(indices[:n_val])
    train_indices = sorted(indices[n_val:])

    train_rows = [merged_traindev_clean[i] for i in train_indices]
    val_rows   = [merged_traindev_clean[i] for i in val_indices]
    test_rows  = zenodo_splits["test"]["rows"]  # UNTOUCHED

    print(f"  Resplit (seed={args.resplit_seed}): train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}")

    # Verify no remaining overlap
    final_train = set(r["text"].strip() for r in train_rows)
    final_val   = set(r["text"].strip() for r in val_rows)
    final_test  = set(r["text"].strip() for r in test_rows)
    assert len(final_train & final_val)  == 0, f"Remaining train/val overlap after fix!"
    assert len(final_train & final_test) == 0, f"Remaining train/test overlap after fix!"
    assert len(final_val   & final_test) == 0, f"Remaining val/test overlap after fix!"
    print("  All overlaps resolved. OK")

    # Save resplit index file
    resplit_info = {
        "seed": args.resplit_seed,
        "val_fraction": args.val_fraction,
        "zenodo_train_size": len(zenodo_splits["train"]["rows"]),
        "zenodo_dev_size":   len(zenodo_splits["dev"]["rows"]),
        "zenodo_test_size":  len(zenodo_splits["test"]["rows"]),
        "zenodo_overlap_train_dev":  train_dev_overlap,
        "zenodo_overlap_train_test": train_test_overlap,
        "zenodo_overlap_dev_test":   dev_test_overlap,
        "after_dedup": n_after_dedup,
        "after_test_removal": len(merged_traindev_clean),
        "final_train": len(train_rows),
        "final_val":   len(val_rows),
        "final_test":  len(test_rows),
        "note": "A_R1b: Zenodo train+dev merged, deduped, test-contaminated docs removed, seeded resplit",
    }
    with open(os.path.join(args.out_dir, "reuters_resplit_seed42.json"), "w") as f:
        json.dump(resplit_info, f, indent=2)
    print(f"  Resplit info saved to reuters_resplit_seed42.json")

    # ------------------------------------------------------------------ #
    # 5. Validate definitions file
    # ------------------------------------------------------------------ #
    if not os.path.exists(args.defs_path):
        raise FileNotFoundError(f"Definitions not found: {args.defs_path}")
    with open(args.defs_path, encoding="utf-8") as f:
        defs = json.load(f)
    defs_keys_lower = {k.lower(): k for k in defs.keys()}
    missing = [l for l in label_order if l.lower() not in defs_keys_lower]
    if missing:
        raise ValueError(f"Labels missing from definitions: {missing}")
    print(f"Definitions verified: all {len(label_order)} labels present. OK")

    # ------------------------------------------------------------------ #
    # 6. Build label texts
    # ------------------------------------------------------------------ #
    print("\nBuilding label texts...")
    label_texts = build_all_label_texts(label_order, args.defs_path, dataset="reuters")
    assert len(label_texts) == len(label_order)
    print(f"  {len(label_texts)} label texts built.")
    for i, (label, text) in enumerate(zip(label_order, label_texts)):
        if i < 3:
            print(f"  [{i}] {label}: {text[:80]}...")

    # ------------------------------------------------------------------ #
    # 7. Convert to arrays and save
    # ------------------------------------------------------------------ #
    output_splits = {
        "train": train_rows,
        "dev":   val_rows,    # 'dev' is the new validation set (seeded resplit)
        "test":  test_rows,
    }

    manifest = {
        "dataset": "Reuters-21578",
        "source": "Zenodo 10.5281/zenodo.6344750",
        "zip_md5": zip_md5,
        "expected_zip_md5": EXPECTED_MD5,
        "n_labels": len(label_order),
        "label_order": label_order,
        "note_A_R1b": (
            "A_R1b: Zenodo train+dev merged, deduplicated by text, docs also in test removed, "
            "then seeded 90/10 resplit (seed=42). "
            "Zenodo D8 overlaps: train/dev=31, train/test=4, dev/test=2 (all resolved). "
            f"Final: train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}."
        ),
        "splits": {},
    }

    for split, rows in output_splits.items():
        print(f"\nProcessing {split}: {len(rows)} docs...")
        texts = [row.get("text", "").strip() or "[EMPTY]" for row in rows]
        label_array = np.array(
            [[int(row.get(lbl, 0)) for lbl in label_order] for row in rows],
            dtype=np.float32,
        )
        counts = label_array.sum(axis=1)
        zero = int((counts == 0).sum())
        manifest["splits"][split] = {
            "n_docs": len(rows),
            "n_labels": int(label_array.shape[1]),
            "mean_labels_per_doc": float(counts.mean()),
            "min_labels_per_doc": int(counts.min()),
            "max_labels_per_doc": int(counts.max()),
            "zero_label_docs": zero,
        }
        if zero > 0:
            print(f"  WARNING: {zero} zero-label docs in {split}")
        print(f"  mean_labels={counts.mean():.3f}")

        np.save(os.path.join(args.out_dir, f"{split}_labels.npy"), label_array)
        with open(os.path.join(args.out_dir, f"{split}_texts.json"), "w", encoding="utf-8") as f:
            json.dump(texts, f, ensure_ascii=False)
        print(f"  Saved {split}_labels.npy {label_array.shape} and {split}_texts.json")

    with open(os.path.join(args.out_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(label_order, f, indent=2)
    with open(os.path.join(args.out_dir, "label_texts.json"), "w", encoding="utf-8") as f:
        json.dump(label_texts, f, indent=2, ensure_ascii=False)

    # SHA-256 manifest
    out_files = [
        "train_labels.npy", "train_texts.json",
        "dev_labels.npy",   "dev_texts.json",
        "test_labels.npy",  "test_texts.json",
        "labels.json",      "label_texts.json",
    ]
    manifest["output_sha256"] = {
        fname: sha256(os.path.join(args.out_dir, fname))
        for fname in out_files
        if os.path.exists(os.path.join(args.out_dir, fname))
    }
    with open(os.path.join(args.out_dir, "DATA_MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # ------------------------------------------------------------------ #
    # 8. Final assertions
    # ------------------------------------------------------------------ #
    tl = np.load(os.path.join(args.out_dir, "train_labels.npy"))
    assert tl.shape == (manifest["splits"]["train"]["n_docs"], len(label_order))
    assert tl.dtype == np.float32

    # Verify disjointness on saved files
    with open(os.path.join(args.out_dir, "train_texts.json"), encoding="utf-8") as f:
        saved_train = set(json.load(f))
    with open(os.path.join(args.out_dir, "dev_texts.json"), encoding="utf-8") as f:
        saved_dev = set(json.load(f))
    with open(os.path.join(args.out_dir, "test_texts.json"), encoding="utf-8") as f:
        saved_test = set(json.load(f))
    assert len(saved_train & saved_dev)  == 0, "train/dev overlap after save!"
    assert len(saved_train & saved_test) == 0, "train/test overlap after save!"
    assert len(saved_dev   & saved_test) == 0, "dev/test overlap after save!"

    print("\n" + "=" * 60)
    print("Reuters data preparation COMPLETE")
    print("=" * 60)
    for split in ["train", "dev", "test"]:
        s = manifest["splits"][split]
        print(f"  {split:6s}: {s['n_docs']:5d} docs, "
              f"mean {s['mean_labels_per_doc']:.3f} labels/doc, "
              f"zero_label={s['zero_label_docs']}")
    print("\nNOTE (A_R1b): Zenodo train+dev merged, deduped, test-contaminated removed, "
          f"seeded 90/10 resplit (seed={args.resplit_seed}).")
    print("  Zenodo D8 violations resolved: train/dev, train/test, dev/test overlaps removed.")


if __name__ == "__main__":
    main()
