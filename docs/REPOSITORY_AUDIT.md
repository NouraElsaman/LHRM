# Repository Audit

## Context
The repository contains historically tracked large objects, datasets, and generated caches which must not be published to the remote `NouraElsaman/LHRM.git`. The remote is currently empty.

## Protected Paths
- `run_baseline.py`
- `src/lhrm_baseline/`
- `Model/lhrm.py`
- `legacy/lhrm_original.py`
- `scripts/`
- `configs/`
- `Label_Definition/`
- `tests/`
- `notebooks/`
- `docs/`

## Cleanup Candidates
- Historical `.zip` bundles (e.g. `bundles/`, `dist/`, `experiments/`) that duplicate evidence.
- Pycache and caches: `__pycache__`, `.pytest_cache`.
- Unused scratch scripts.
- Contaminated `.git/` history (will create a clean publication workspace).

## Action Plan
1. Delete redundant `.zip` bundles.
2. Delete caches.
3. Migrate `scratch/test_variant_b.py` to `tests/`.
4. Create an isolated clean publication workspace outside the current `.git` tracking.
5. Create meaningful Git commits in the new workspace.
6. Publish to GitHub.
