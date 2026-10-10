"""
Patch notebook cells 9 and 16 to use diagnostics module.
Run: python scripts/patch_notebook.py
"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_PATH = os.path.join(ROOT, "notebooks", "lhrm_run0_aapd_gpu.ipynb")

nb = json.load(open(NB_PATH, encoding="utf-8"))

# ── Cell 9: thin wrapper around diagnostics.compute_diagnostics ───────────────
cell9_new = """\
# Cell 9: val_diagnostics -- thin wrapper around diagnostics module (F1/F2/F3 fix)
# All diagnostic logic lives in lhrm_baseline.diagnostics; no duplicate code here.
from lhrm_baseline.diagnostics import compute_diagnostics, DiagnosticError
from lhrm_baseline.diagnostics import frequency_prior_hits_at_1

def val_diagnostics(model, device, train_labels_np, step=0):
    \"\"\"
    F1/F2/F3 fix (Phase 1F): thin wrapper calling diagnostics.compute_diagnostics.
      F1: requires diag['alpha'] tensor -- raises DiagnosticError if missing
      F2: integer prior hit-count with seeded tie-break (not float, not column-order)
      F3: fp32, no autocast (nullcontext only)
    \"\"\"
    from torch.utils.data import DataLoader
    vl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                    collate_fn=collate_fn, num_workers=0, pin_memory=True)
    result = compute_diagnostics(
        model=model,
        loader=vl,
        label_input_ids=label_ids,
        label_attention_mask=label_mask,
        val_labels=v_labels,
        device=device,
        train_labels=train_labels_np,
        n_labels=len(label_order),
        seed=SEED,
        amp_for_training=USE_AMP,  # informational only; no effect on fp32 measurement
    )
    result["step"] = step
    return result

# Pre-compute integer prior hits (seeded tie-break, once)
train_labels_np = np.load(os.path.join(DATA_DIR, "train_labels.npy"))
PRIOR_HITS = frequency_prior_hits_at_1(train_labels_np, v_labels, seed=SEED)
print(f"Prior P@1 (seeded tie-break): {PRIOR_HITS}/200 = {PRIOR_HITS/200:.4f}")
"""

# ── Cell 16: thin wrapper around diagnostics.run0_decision ────────────────────
cell16_new = """\
# Cell 16: Scripted decision (thin wrapper -- logic lives in diagnostics.run0_decision)
from lhrm_baseline.diagnostics import run0_decision as _run0_decision
from lhrm_baseline.diagnostics import evaluate_criteria
import json

# Attach integer prior hits so run0_decision can read them
for variant, r in all_results.items():
    r["prior_p_at_1_hits"] = PRIOR_HITS
    r["prior_p_at_1"]      = PRIOR_HITS / 200

decision = _run0_decision(all_results, candidates=CANDIDATES)
print("=" * 60)
print("SCRIPTED DECISION:", decision["status"], "--", decision.get("chosen_variant"))
print("Reason:", decision["reason"])
print("=" * 60)

OUT["decision"] = decision
with open(os.path.join(OUT_DIR, "run0_decision.json"), "w") as f:
    json.dump(decision, f, indent=2)
print("Written: run0_decision.json")
"""

# Apply patches
def _as_lines(src: str):
    """Convert source string to list of lines as the notebook format expects."""
    lines = src.split("\n")
    return [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

nb["cells"][9]["source"]  = _as_lines(cell9_new)
nb["cells"][16]["source"] = _as_lines(cell16_new)

json.dump(nb, open(NB_PATH, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("Notebook patched: cells 9 and 16 updated.")
print("  Cell 9:  val_diagnostics -> compute_diagnostics wrapper")
print("  Cell 16: run0_decision   -> diagnostics.run0_decision wrapper")
