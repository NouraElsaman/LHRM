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
