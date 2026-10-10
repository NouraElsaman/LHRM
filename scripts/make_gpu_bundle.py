"""
Build the GPU bundle for RUN_0 on Kaggle/Colab.

Usage:
  python scripts/make_gpu_bundle.py --config configs/final/aapd_baseline.yaml \
      --run_id lhrm_run0_aapd_L_focal_seed42_v1 \
      --out bundles/

What it produces:
  bundles/lhrm_gpu_bundle_<date>.zip containing:
    run_baseline.py
    src/lhrm_baseline/ (full package)
    scripts/make_gpu_bundle.py
    configs/final/aapd_baseline.yaml
    configs/final/CONFIG_FREEZE.json
    notebooks/lhrm_run0_aapd_gpu.ipynb
    data/processed/AAPD/ (processed data + DATA_MANIFEST.json)
    docs/phase1/run0_preregistration.md
    BUNDLE_MANIFEST.json (hashes, git commit, label_order_hash, freeze_hash)

The bundle is self-contained: the Kaggle notebook can run from it without
internet access (except for bert-base-uncased which must be attached as a dataset).
"""
import argparse
import hashlib
import json
import os
import subprocess
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty  = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except Exception as e:
        return {"commit": "UNKNOWN", "dirty": True, "error": str(e)}


def collect_files(base, rel_dir, extensions=None, exclude=None):
    """Yield (abs_path, archive_path) for all files under rel_dir."""
    abs_dir = os.path.join(base, rel_dir)
    if not os.path.exists(abs_dir):
        return
    for dirpath, _, fnames in os.walk(abs_dir):
        for fname in sorted(fnames):
            if extensions and not any(fname.endswith(e) for e in extensions):
                continue
            if exclude and any(ex in dirpath for ex in exclude):
                continue
            full = os.path.join(dirpath, fname)
            rel  = os.path.relpath(full, base).replace("\\", "/")
            yield full, rel


def main():
    parser = argparse.ArgumentParser(description="Build GPU bundle for RUN_0")
    parser.add_argument("--config", default="configs/final/aapd_baseline.yaml")
    parser.add_argument("--run_id", default="lhrm_run0_aapd_L_focal_seed42_v1")
    parser.add_argument("--out", default="bundles")
    parser.add_argument("--include_data", action="store_true",
                        help="Include data/processed/AAPD/ in bundle (large; ~50 MB)")
    parser.add_argument("--require_clean", action="store_true",
                        help="Fail if working tree is dirty")
    args = parser.parse_args()

    git = get_git()
    if args.require_clean and git["dirty"]:
        print("ERROR: --require_clean set but working tree is dirty.")
        import sys; sys.exit(1)
    if git["dirty"]:
        print("WARNING: Working tree is dirty. Bundle will record dirty=true.")

    os.makedirs(os.path.join(ROOT, args.out), exist_ok=True)
    date_str = time.strftime("%Y%m%d_%H%M%S")
    bundle_name = f"lhrm_gpu_bundle_{date_str}.zip"
    bundle_path = os.path.join(ROOT, args.out, bundle_name)

    # Collect files
    to_include = []  # (abs_path, archive_path)

    # Core code
    for full, rel in collect_files(ROOT, "src", extensions=[".py"]):
        to_include.append((full, rel))
    to_include.append((os.path.join(ROOT, "run_baseline.py"), "run_baseline.py"))
    to_include.append((os.path.join(ROOT, "scripts", "make_gpu_bundle.py"),
                       "scripts/make_gpu_bundle.py"))

    # Config + freeze
    cfg_full = os.path.join(ROOT, args.config)
    to_include.append((cfg_full, args.config.replace("\\", "/")))
    freeze_path = os.path.join(ROOT, "configs", "final", "CONFIG_FREEZE.json")
    if os.path.exists(freeze_path):
        to_include.append((freeze_path, "configs/final/CONFIG_FREEZE.json"))

    # Notebooks: prefer CLI-driven notebook; also keep legacy for reference
    nb_cli = os.path.join(ROOT, "notebooks", "lhrm_run0_kaggle_cli.ipynb")
    nb_legacy = os.path.join(ROOT, "notebooks", "lhrm_run0_aapd_gpu.ipynb")
    if os.path.exists(nb_cli):
        to_include.append((nb_cli, "notebooks/lhrm_run0_kaggle_cli.ipynb"))
        print("Including CLI notebook:", "notebooks/lhrm_run0_kaggle_cli.ipynb")
    if os.path.exists(nb_legacy):
        to_include.append((nb_legacy, "notebooks/lhrm_run0_aapd_gpu.ipynb"))
    if not os.path.exists(nb_cli) and not os.path.exists(nb_legacy):
        print("WARNING: no notebook found; not included")

    # Label definitions
    defs_path = os.path.join(ROOT, "Label_Definition", "arxiv_labels_definitions.json")
    if os.path.exists(defs_path):
        to_include.append((defs_path, "Label_Definition/arxiv_labels_definitions.json"))

    # Documentation
    docs = [
        "docs/phase1/run0_preregistration.md",
        "docs/phase1/decision_registry_v4.md",
        "docs/phase1/aapd_data_decision.md",
    ]
    for doc in docs:
        fp = os.path.join(ROOT, doc)
        if os.path.exists(fp):
            to_include.append((fp, doc))

    # Cross-check reference and pre-registered val IDs (needed for preflight gate)
    crosscheck_ref = os.path.join(ROOT, "docs", "phase1", "run0_local_step0_crosscheck.json")
    if os.path.exists(crosscheck_ref):
        to_include.append((crosscheck_ref, "docs/phase1/run0_local_step0_crosscheck.json"))
    else:
        print("WARNING: run0_local_step0_crosscheck.json not found; preflight will use hardcoded refs")

    val_ids_path = os.path.join(ROOT, "experiments", "RUN_0", "run0_val_sample_ids.json")
    if os.path.exists(val_ids_path):
        to_include.append((val_ids_path, "experiments/RUN_0/run0_val_sample_ids.json"))
    else:
        print("WARNING: pre-registered val IDs not found; run0 will regenerate from seed=42")

    # Processed data (optional; large)
    if args.include_data:
        for full, rel in collect_files(ROOT, "data/processed/AAPD",
                                       exclude=["test_"]):
            to_include.append((full, rel))
        print("Including processed AAPD data (train + dev only, no test_*)")

    # Build bundle manifest
    manifest = {
        "bundle_name": bundle_name,
        "run_id": args.run_id,
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": git,
        "config": args.config,
        "files": {},
    }
    for full, rel in to_include:
        if os.path.exists(full):
            manifest["files"][rel] = sha256_file(full)

    # Add key hashes
    if os.path.exists(freeze_path):
        manifest["config_freeze_sha256"] = sha256_file(freeze_path)
    lo_path = os.path.join(ROOT, "data", "processed", "AAPD", "labels.json")
    if os.path.exists(lo_path):
        import json as _json
        with open(lo_path) as f:
            lo = _json.load(f)
        import hashlib as _h
        manifest["label_order_sha256"] = _h.sha256(_json.dumps(lo).encode()).hexdigest()
    dm_path = os.path.join(ROOT, "data", "processed", "AAPD", "DATA_MANIFEST.json")
    if os.path.exists(dm_path):
        manifest["data_manifest_sha256"] = sha256_file(dm_path)

    # Write manifest json then add to archive
    manifest_path = os.path.join(ROOT, args.out, f"BUNDLE_MANIFEST_{date_str}.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Create zip
    n_written = 0
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for full, rel in to_include:
            if os.path.exists(full):
                zf.write(full, rel)
                n_written += 1
        zf.write(manifest_path, "BUNDLE_MANIFEST.json")

    bundle_sha = sha256_file(bundle_path)
    print(f"Bundle written: {bundle_path}")
    print(f"  Files included: {n_written}")
    print(f"  SHA-256: {bundle_sha}")
    print(f"  Size: {os.path.getsize(bundle_path) / 1024:.1f} KB")
    print(f"\nManifest: {manifest_path}")
    print(f"\nNext steps:")
    print(f"  1. Upload {bundle_name} as a private Kaggle Dataset")
    print(f"  2. Create notebook notebooks/lhrm_run0_aapd_gpu.ipynb")
    print(f"  3. Attach the dataset and the bert-base-uncased dataset")
    print(f"  4. Set runtime: GPU (T4 or P100), Internet: OFF (or ON if bert not attached)")
    print(f"  5. Run all cells; download output")
    print(f"  6. python run_baseline.py import_artifacts --path /path/to/downloaded/")

    # Clean up temp manifest
    os.remove(manifest_path)


if __name__ == "__main__":
    main()
