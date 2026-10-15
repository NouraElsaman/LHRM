#!/usr/bin/env python3
"""
scripts/make_gpu_bundle_r_author.py

Builds the dedicated portable code bundle for R_AUTHOR Kaggle execution:
  bundles/lhrm_gpu_bundle_r_author_20261015.zip

Includes:
- src/lhrm_baseline/ (complete package including model, train, hrrm, inference, etc.)
- run_baseline.py
- configs/final/aapd_r_author_bce.yaml
- configs/final/CONFIG_FREEZE.json
- Label_Definition/arxiv_labels_definitions.json
- BUNDLE_MANIFEST.json
- SHA256SUMS.txt
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except Exception as e:
        return {"commit": "UNKNOWN", "dirty": True, "error": str(e)}


def collect_files(base: str, rel_dir: str, extensions=None, exclude=None):
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
            rel = os.path.relpath(full, base).replace("\\", "/")
            yield full, rel


def build_bundle(out_dir: str = "bundles", bundle_name: str = "lhrm_gpu_bundle_r_author_20261015.zip") -> str:
    os.makedirs(os.path.join(ROOT, out_dir), exist_ok=True)
    bundle_path = os.path.join(ROOT, out_dir, bundle_name)

    to_include = []

    # 1. Source package
    for full, rel in collect_files(ROOT, "src", extensions=[".py"]):
        to_include.append((full, rel))

    # 2. CLI entrypoint
    to_include.append((os.path.join(ROOT, "run_baseline.py"), "run_baseline.py"))

    # 3. Config files
    cfg_path = os.path.join(ROOT, "configs", "final", "aapd_r_author_bce.yaml")
    if os.path.exists(cfg_path):
        to_include.append((cfg_path, "configs/final/aapd_r_author_bce.yaml"))
    freeze_path = os.path.join(ROOT, "configs", "final", "CONFIG_FREEZE.json")
    if os.path.exists(freeze_path):
        to_include.append((freeze_path, "configs/final/CONFIG_FREEZE.json"))

    # 4. Label definitions
    defs_path = os.path.join(ROOT, "Label_Definition", "arxiv_labels_definitions.json")
    if os.path.exists(defs_path):
        to_include.append((defs_path, "Label_Definition/arxiv_labels_definitions.json"))

    # 5. Manifest metadata
    git_info = get_git()
    manifest = {
        "bundle_name": bundle_name,
        "run_id": "AAPD_R_AUTHOR_BCE_RECOVERY_20261015_A",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": git_info,
        "variant": "R_AUTHOR",
        "loss": "bce",
        "files": {},
    }

    for full, rel in to_include:
        if os.path.exists(full):
            manifest["files"][rel] = sha256_file(full)

    # Build ZIP archive
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for full, rel in to_include:
            if os.path.exists(full):
                z.write(full, rel)
        # Write BUNDLE_MANIFEST.json into root of zip
        manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
        z.writestr("BUNDLE_MANIFEST.json", manifest_bytes)
        
        # Write SHA256SUMS.txt into root of zip
        sums_lines = []
        for rel in sorted(manifest["files"].keys()):
            sums_lines.append(f"{manifest['files'][rel]}  {rel}")
        # Add manifest itself to sums
        m_hash = hashlib.sha256(manifest_bytes).hexdigest()
        sums_lines.append(f"{m_hash}  BUNDLE_MANIFEST.json")
        z.writestr("SHA256SUMS.txt", "\n".join(sums_lines) + "\n")

    b_sha = sha256_file(bundle_path)
    print(f"Created bundle: {bundle_path}")
    print(f"Bundle size: {os.path.getsize(bundle_path)} bytes")
    print(f"Bundle SHA-256: {b_sha}")

    # Also save local manifest sidecar
    sidecar_path = os.path.join(ROOT, out_dir, bundle_name.replace(".zip", "_MANIFEST.json"))
    manifest["bundle_sha256"] = b_sha
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return bundle_path, b_sha


def main():
    parser = argparse.ArgumentParser(description="Build R_AUTHOR GPU bundle")
    parser.add_argument("--out", default="bundles")
    parser.add_argument("--bundle_name", default="lhrm_gpu_bundle_r_author_20261015.zip")
    args = parser.parse_args()
    build_bundle(out_dir=args.out, bundle_name=args.bundle_name)


if __name__ == "__main__":
    main()
