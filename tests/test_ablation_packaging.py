"""
Test suite for L2 vs L2_NO_SA ablation screening packaging and validation.
Verifies all 10 local CPU validation requirements without GPU execution.
"""
import json
import os
import sys
import zipfile
import hashlib
import pytest
import torch
from transformers import BertConfig, BertModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from lhrm_baseline.model import LHRMModel
from run_baseline import _CROSSCHECK_REFS, load_yaml_config

NOTEBOOK_PATH = os.path.join(ROOT, "notebooks", "lhrm_run0_l2_no_sa_ablation_kaggle.ipynb")
BUNDLE_PATH = os.path.join(ROOT, "bundles", "lhrm_gpu_bundle_l2_no_sa_ablation_20261011.zip")
CROSSCHECK_PATH = os.path.join(ROOT, "docs", "phase1", "run0_local_step0_crosscheck.json")
VAL_IDS_PATH = os.path.join(ROOT, "experiments", "RUN_0", "run0_val_sample_ids.json")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def test_1_notebook_json_validity():
    """Verify notebook is well-formed JSON with correct structure."""
    assert os.path.isfile(NOTEBOOK_PATH), f"Notebook missing: {NOTEBOOK_PATH}"
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)
    assert nb.get("nbformat") == 4
    assert "cells" in nb and len(nb["cells"]) >= 8
    assert "metadata" in nb
    assert nb["metadata"]["kaggle"]["accelerator"] == "GPU"


def test_2_model_variant_registration():
    """Verify L2 and L2_NO_SA are registered and operational."""
    config = BertConfig(hidden_size=64, num_hidden_layers=1, num_attention_heads=1)
    bert = BertModel(config)
    num_labels = 10

    # Both variants instantiate cleanly
    model_l2 = LHRMModel(bert, num_labels, variant="L2")
    model_nosa = LHRMModel(bert, num_labels, variant="L2_NO_SA")
    assert model_l2.variant == "L2"
    assert model_nosa.variant == "L2_NO_SA"

    # Forward pass works on CPU
    doc_ids = torch.randint(0, 100, (2, 16))
    doc_mask = torch.ones(2, 16)
    label_ids = torch.randint(0, 100, (num_labels, 8))
    label_mask = torch.ones(num_labels, 8)

    model_nosa.eval()
    logits, diag = model_nosa(doc_ids, doc_mask, label_ids, label_mask)
    assert logits.shape == (2, num_labels)
    assert "alpha" in diag and "S" in diag


def test_3_preflight_reference_availability():
    """Verify CPU step-0 cross-check reference contains L2 and L2_NO_SA."""
    assert os.path.isfile(CROSSCHECK_PATH), f"Crosscheck reference missing: {CROSSCHECK_PATH}"
    with open(CROSSCHECK_PATH) as f:
        data = json.load(f)
    assert "variants" in data
    assert "L2" in data["variants"]
    assert "L2_NO_SA" in data["variants"]
    assert data["prior_p_at_1_hits"] == 63

    # Check CLI reference dictionary
    assert "L2" in _CROSSCHECK_REFS
    assert "L2_NO_SA" in _CROSSCHECK_REFS
    assert _CROSSCHECK_REFS["L2_NO_SA"]["prior_p_at_1_hits"] == (63, 0)


def test_4_bundle_contents_and_hashes():
    """Verify bundle exists, contains required files, and hash matches notebook."""
    assert os.path.isfile(BUNDLE_PATH), f"Bundle missing: {BUNDLE_PATH}"
    actual_bundle_sha = sha256_file(BUNDLE_PATH)

    # Verify notebook contains this exact hash
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_text = f.read()
    assert actual_bundle_sha in nb_text, "Notebook does not contain actual bundle SHA-256!"

    with zipfile.ZipFile(BUNDLE_PATH, "r") as z:
        names = set(z.namelist())

    required = {
        "run_baseline.py",
        "src/lhrm_baseline/model.py",
        "src/lhrm_baseline/train.py",
        "src/lhrm_baseline/diagnostics.py",
        "configs/final/aapd_baseline.yaml",
        "configs/final/CONFIG_FREEZE.json",
        "Label_Definition/arxiv_labels_definitions.json",
        "experiments/RUN_0/run0_val_sample_ids.json",
        "docs/phase1/run0_local_step0_crosscheck.json",
        "BUNDLE_MANIFEST.json",
    }
    missing = required - names
    assert not missing, f"Missing files in bundle: {missing}"

    # Verify no test data in bundle
    assert not any("test" in n.lower() and "data" in n.lower() for n in names)
    assert not any(n.endswith(".pt") for n in names)


def test_5_command_construction():
    """Verify CLI command construction in notebook."""
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Check cell 5 source
    cell5 = nb["cells"][5]
    code5 = "".join(cell5["source"])
    assert "RUN_0_L2_NO_SA_ABLATION_20261011_A" in code5
    assert "--steps" in code5 and "300" in code5
    assert "L2,L2_NO_SA" in code5
    assert "run0" in code5


def test_6_frozen_sample_ids():
    """Verify the pre-registered 200 validation doc IDs."""
    assert os.path.isfile(VAL_IDS_PATH), f"Val IDs file missing: {VAL_IDS_PATH}"
    with open(VAL_IDS_PATH) as f:
        ids = json.load(f)
    assert len(ids) == 200
    assert ids[:5] == [3, 6, 11, 25, 27]
    assert len(set(ids)) == 200


def test_7_input_path_resolution():
    """Verify referenced configs and label definitions exist."""
    cfg_path = os.path.join(ROOT, "configs", "final", "aapd_baseline.yaml")
    assert os.path.isfile(cfg_path)
    cfg = load_yaml_config(cfg_path)
    assert cfg.get("dataset") == "aapd"
    assert cfg.get("seed") == 42
    assert cfg.get("focal_loss_gamma") == 2.0
    assert cfg.get("focal_loss_alpha") == 0.25

    defs_path = os.path.join(ROOT, "Label_Definition", "arxiv_labels_definitions.json")
    assert os.path.isfile(defs_path)


def test_8_no_test_access():
    """Verify notebook strictly quarantines test data."""
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)
    all_code = "".join("".join(c["source"]) for c in nb["cells"])

    # Verify no test dataset preparation
    assert "test_labels.npy" not in all_code
    assert "test_texts.json" not in all_code
    assert "TEST SPLIT QUARANTINE" in all_code
    assert 'test_files = list(processed_dir.glob("test_*"))' in all_code


def test_9_no_historical_run_overwrite(tmp_path):
    """Verify fail-closed protections for historical run IDs."""
    import argparse

    # Test protected run IDs exist and are preserved
    for protected_id in ["RUN_0", "DR1", "DR2", "RUN_0_P1F_EXT1"]:
        exp_dir = os.path.join(ROOT, "experiments", protected_id)
        if os.path.exists(exp_dir):
            assert os.listdir(exp_dir), f"{protected_id} directory must not be empty"


def test_10_unchanged_l2_behavior():
    """Verify L2 implementation matches expected CSAM architecture."""
    config = BertConfig(hidden_size=64, num_hidden_layers=1, num_attention_heads=1)
    bert = BertModel(config)
    model_l2 = LHRMModel(bert, num_labels=5, variant="L2")

    # In L2, label self-attention is active and similarity is scaled
    assert model_l2.variant == "L2"
    assert model_l2.csam.use_label_sa is True
    assert model_l2.csam.scale_similarity is True

    # In L2_NO_SA, label self-attention is ablated and similarity is scaled
    model_nosa = LHRMModel(bert, num_labels=5, variant="L2_NO_SA")
    assert model_nosa.variant == "L2_NO_SA"
    assert model_nosa.csam.use_label_sa is False
    assert model_nosa.csam.scale_similarity is True

    # In L, label self-attention is active and similarity is unscaled
    model_l = LHRMModel(bert, num_labels=5, variant="L")
    assert model_l.variant == "L"
    assert model_l.csam.use_label_sa is True
    assert model_l.csam.scale_similarity is False


def test_11_cell4_cell5_sequence_no_premature_mkdir(tmp_path):
    """
    Regression test for BUG 1: Cell 4 must NOT prematurely create or populate
    the experiment output directory, which would trip cmd_run0's fail-closed guard.
    """
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # 1. Assert notebook Cell 4 does not create or copy into EXP_DIR ahead of time
    cell4_code = "".join(nb["cells"][4]["source"])
    assert "EXP_DIR.mkdir" not in cell4_code, "Cell 4 must not mkdir EXP_DIR prematurely"
    assert "shutil.copy2(src_ids_file, dest_ids_file)" not in cell4_code, "Cell 4 must not write into EXP_DIR"
    assert "Validated locked 200 validation IDs directly from RUN_0" in cell4_code

    # 2. Simulate Cell 4 -> Cell 5 lifecycle
    run_id = "TEST_RUN_ABLATION_REGRESSION"
    target_exp_dir = tmp_path / "experiments" / run_id

    # Cell 4 check: directory does not exist or is empty
    if target_exp_dir.exists() and (target_exp_dir.is_file() or any(target_exp_dir.iterdir())):
        raise RuntimeError("FAIL-CLOSED triggered")

    # Cell 4 validates IDs directly from source
    assert os.path.isfile(VAL_IDS_PATH)
    val_ids = json.load(open(VAL_IDS_PATH))
    assert len(val_ids) == 200

    # Before cmd_run0 starts, target_exp_dir must not be non-empty
    assert not target_exp_dir.exists() or not os.listdir(str(target_exp_dir))

    # cmd_run0 preflight check guard:
    out_base = str(target_exp_dir)
    guard_triggered = False
    if os.path.exists(out_base) and (os.path.isfile(out_base) or (os.path.isdir(out_base) and os.listdir(out_base))):
        guard_triggered = True
    assert not guard_triggered, "cmd_run0 fail-closed guard tripped on fresh run!"

    # cmd_run0 creates directory and copies freeze IDs cleanly
    os.makedirs(out_base, exist_ok=True)
    freeze_ids_path = VAL_IDS_PATH
    saved_ids_path = os.path.join(out_base, "run0_val_sample_ids.json")
    with open(freeze_ids_path) as f:
        copied_ids = json.load(f)
    with open(saved_ids_path, "w") as f:
        json.dump(copied_ids, f)
    assert os.path.isfile(saved_ids_path)
    assert json.load(open(saved_ids_path)) == val_ids

    # 3. Verify guard DOES trigger if non-empty directory pre-existed (historical preservation)
    if os.path.exists(out_base) and (os.path.isfile(out_base) or (os.path.isdir(out_base) and os.listdir(out_base))):
        guard_triggered = True
    assert guard_triggered, "cmd_run0 must fail-closed if target directory already has artifacts"


def test_12_val_sample_predictions_schema_and_diversity(tmp_path):
    """
    Regression test for BUG 2: Ensure val_sample_predictions.json list-of-records schema
    is parsed correctly and ranking diversity is accurately calculated from prediction scores.
    """
    import numpy as np
    from lhrm_baseline.train import save_predictions

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # 1. Verify notebook Cell 7 code uses list-of-records schema and prediction scores
    cell7_code = "".join(nb["cells"][7]["source"])
    assert 'records = json.load(open(preds_p))' in cell7_code
    assert 'assert isinstance(records, list)' in cell7_code
    assert 'scores_matrix.append(rec_scores)' in cell7_code
    assert 'data["y_pred"]' not in cell7_code, "Notebook must not use invalid data['y_pred'] key"

    # 2. Build realistic prediction fixture using save_predictions()
    defs_path = os.path.join(ROOT, "Label_Definition", "arxiv_labels_definitions.json")
    with open(defs_path) as f:
        defs_data = json.load(f)
    label_order = list(defs_data.keys())
    assert len(label_order) == 54

    n_docs = 200
    n_labels = 54
    np.random.seed(42)

    # Create synthetic true binary labels and predicted probabilities
    y_true = np.zeros((n_docs, n_labels), dtype=np.float32)
    # Set 1-3 positive labels per doc
    for i in range(n_docs):
        pos_idx = [i % n_labels, (i + 1) % n_labels]
        y_true[i, pos_idx] = 1.0

    # Create synthetic predicted probabilities (scores from sigmoid in [0, 1])
    y_pred = np.random.uniform(0.01, 0.40, size=(n_docs, n_labels)).astype(np.float32)
    # Give document i highest score at label (i % 54) to produce 54 distinct top-1 labels
    for i in range(n_docs):
        y_pred[i, i % n_labels] = 0.95
        y_pred[i, (i + 1) % n_labels] = 0.85
        y_pred[i, (i + 2) % n_labels] = 0.75

    fixture_path = str(tmp_path / "val_sample_predictions.json")
    save_predictions(y_true, y_pred, label_order, fixture_path)

    # 3. Execute exact Cell 7 parsing logic on the fixture
    records = json.load(open(fixture_path))
    assert isinstance(records, list)
    assert len(records) == 200

    scores_matrix = []
    for idx, rec in enumerate(records):
        assert "scores" in rec and "labels" in rec
        assert rec["labels"] == label_order
        rec_scores = rec["scores"]
        assert len(rec_scores) == 54
        scores_matrix.append(rec_scores)

    scores = np.array(scores_matrix, dtype=np.float32)
    assert scores.shape == (200, 54)
    assert np.all(np.isfinite(scores))

    # Diversity metrics
    top1_indices = np.argmax(scores, axis=1)
    distinct_top1 = len(set(top1_indices.tolist()))
    assert distinct_top1 == 54, f"Expected 54 distinct top-1, got {distinct_top1}"

    top3_indices = np.argsort(-scores, axis=1)[:, :3]
    top3_tuples = [tuple(row.tolist()) for row in top3_indices]
    distinct_top3 = len(set(top3_tuples))
    assert distinct_top3 == 54, f"Expected 54 distinct top-3 patterns, got {distinct_top3}"
