# Repository Cleanup Report

## Deletions
- **Caches:** Removed `__pycache__`, `.pytest_cache` directories.
  - Reason: These are disposable and regenerate automatically.
- **Bundles:** Removed generated `.zip` files in `bundles/`, `dist/`, and `experiments/` (excluding the approved evidence ZIP `RUN_0_P1F_EXT1_EVIDENCE_20261009_222831.zip`).
  - Reason: Redundant duplicates of identical Kaggle bundles. Kept manifestations (e.g. `BUNDLE_MANIFEST_*.json`) for provenance.
- **Git History:** Ignored `.git/` in the new publication workspace.
  - Reason: Old history is contaminated with datasets.

## Migrations
- **scratch/test_variant_b.py -> tests/test_variant_b.py:** Migrated as a permanent suite requirement for the standard MLTC Variant B control.

## Preservations
- **experiments/RUN_0_P1F_EXT1/RUN_0_P1F_EXT1_EVIDENCE_20261009_222831.zip**: Required scientific evidence of DR1 extension.
- **Original Client Files**: Maintained in `legacy/` or adapted into `src/lhrm_baseline/`.
- **docs/**: Maintained to describe all phases of testing and decisions.

*Note: The original workspace and full Git history is backed up to D:\LHRM_Backup.*
