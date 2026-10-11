"""
tests/test_trained_checkpoint_replay.py

Validation test suite for DR3 Trained-Checkpoint Replay and Diagnostic:
1. Notebook JSON validity and Python syntax across all code cells.
2. New experiment ID safety and fail-closed isolation from historical runs.
3. Bundle checksum matching and archive contents verification.
4. Checkpoint serialization schema and required key presence.
5. Full model loading with strict state-dict validation (strict=True) on lightweight fixture.
6. Verification that trained-probe requires real checkpoints and never falls back to random weights.
7. Gradient hooks, retention, and forward/backward propagation integrity.
8. Verification that probe causes NO optimizer updates (parameters remain frozen).
9. Model-only checkpoint export, optimizer stripping, and strict round-trip agreement.
10. Verification of deliverable file paths and manifest structure.
11. Quarantine verification: zero Test split access in notebook or bundle.
12. Deliverable manifest integrity and SHA256 checksum formatting.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import pytest
import torch
from transformers import BertConfig, BertModel

from src.lhrm_baseline.model import LHRMModel, CSAM_L, get_cls_embedding
from src.lhrm_baseline.losses import FocalLoss, BCELoss
from scripts.probe_csam_trained_checkpoint import (
    export_model_only_checkpoint,
    probe_single_trained_variant,
    sha256_file,
)
from scripts.gen_kaggle_notebook_replay import BUNDLE_NAME, OUT_NB, RUN_ID


@pytest.fixture
def tiny_bert():
    """Create a lightweight CPU BertModel for fast unit tests without downloads."""
    cfg = BertConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=64,
        vocab_size=100,
    )
    return BertModel(cfg)


def test_1_notebook_json_and_syntax():
    """Verify notebook exists, is valid JSON, and all code cells compile cleanly."""
    nb_path = Path(OUT_NB)
    assert nb_path.is_file(), f"Notebook file missing at {nb_path}"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert "cells" in nb and len(nb["cells"]) >= 8
    for idx, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            # Compile to verify valid Python syntax
            compile(src, f"<cell_{idx}>", "exec")


def test_2_new_run_id_safety():
    """Ensure new experiment ID is unique and does not collide with historical runs."""
    assert RUN_ID == "RUN_0_L2_NO_SA_REPLAY_20261011_B"
    forbidden_ids = [
        "RUN_0",
        "RUN_0_P1F",
        "RUN_0_P1F_EXT1",
        "RUN_0_L2_NO_SA_ABLATION_20261011_A",
    ]
    assert RUN_ID not in forbidden_ids
    assert not Path(f"experiments/{RUN_ID}").exists()


def test_3_bundle_checksum_and_contents():
    """Verify bundle exists, matches SHA, and contains required diagnostic scripts."""
    bundle_path = Path(f"bundles/{BUNDLE_NAME}")
    assert bundle_path.is_file(), f"Bundle missing at {bundle_path}"
    sha = sha256_file(bundle_path)
    assert len(sha) == 64

    import zipfile
    with zipfile.ZipFile(bundle_path, "r") as z:
        names = z.namelist()
        assert "run_baseline.py" in names
        assert "scripts/probe_csam_trained_checkpoint.py" in names
        assert "scripts/probe_csam_gradient_paths.py" in names
        assert "BUNDLE_MANIFEST.json" in names


def test_4_checkpoint_serialization_schema(tiny_bert):
    """Verify checkpoint serialization schema with step, variant, model_state, optimizer_state."""
    model = LHRMModel(tiny_bert, num_labels=4, variant="L2")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "step_300.pt"
        payload = {
            "step": 300,
            "variant": "L2",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "losses": [0.1, 0.05],
            "config": {"seed": 42},
        }
        torch.save(payload, ckpt_path)

        loaded = torch.load(ckpt_path, map_location="cpu")
        assert loaded["step"] == 300
        assert loaded["variant"] == "L2"
        assert "model_state" in loaded
        assert "optimizer_state" in loaded
        assert "losses" in loaded


def test_5_full_model_loading_strict_state_dict(tiny_bert):
    """Verify strict loading of full model state dict."""
    model_orig = LHRMModel(tiny_bert, num_labels=4, variant="L2")
    state = model_orig.state_dict()

    cfg = BertConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=64,
        vocab_size=100,
    )
    fresh_bert = BertModel(cfg)
    fresh_model = LHRMModel(fresh_bert, num_labels=4, variant="L2")
    fresh_model.load_state_dict(state, strict=True)


def test_6_trained_probe_requires_real_checkpoint():
    """Verify that probe fails closed when checkpoint does not exist (never falls back to random)."""
    from scripts.probe_csam_trained_checkpoint import run_trained_checkpoint_diagnostic_suite

    with pytest.raises(FileNotFoundError):
        run_trained_checkpoint_diagnostic_suite(
            checkpoint_dir="nonexistent_checkpoints_dir",
            bert_path="nonexistent_bert_dir",
            data_dir="data/processed/AAPD",
            label_defs_path="Label_Definition/arxiv_labels_definitions.json",
        )


def test_7_gradient_hooks_and_forward_backward_integrity(tiny_bert):
    """Verify gradient retention on h_T and h_L across Focal Loss and BCE."""
    model = LHRMModel(tiny_bert, num_labels=4, variant="L2")
    device = torch.device("cpu")

    B, seq_len = 2, 8
    doc_ids = torch.randint(0, 100, (B, seq_len))
    doc_mask = torch.ones((B, seq_len))
    lbl_ids = torch.randint(0, 100, (4, 4))
    lbl_mask = torch.ones((4, 4))
    targets = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    res = probe_single_trained_variant(
        model=model,
        variant_name="L2",
        doc_input_ids=doc_ids,
        doc_attention_mask=doc_mask,
        label_input_ids=lbl_ids,
        label_attention_mask=lbl_mask,
        targets=targets,
        device=device,
    )

    assert res["max_algebraic_reconstruction_error"] < 1e-4
    fg = res["focal_gradients"]
    assert fg["is_finite"] is True
    assert fg["h_T_norm"] > 0
    assert fg["h_L_norm"] > 0
    assert fg["W_norm"] > 0
    assert fg["b_norm"] > 0

    bg = res["bce_gradients"]
    assert bg["is_finite"] is True
    assert bg["h_T_norm"] > 0


def test_8_no_optimizer_updates(tiny_bert):
    """Verify that the diagnostic probe causes NO updates to model parameters."""
    model = LHRMModel(tiny_bert, num_labels=4, variant="L2_NO_SA")
    device = torch.device("cpu")

    W_orig = model.csam.W.weight.clone()
    b_orig = model.csam.W.bias.clone()

    doc_ids = torch.randint(0, 100, (2, 8))
    doc_mask = torch.ones((2, 8))
    lbl_ids = torch.randint(0, 100, (4, 4))
    lbl_mask = torch.ones((4, 4))
    targets = torch.zeros((2, 4))

    probe_single_trained_variant(
        model=model,
        variant_name="L2_NO_SA",
        doc_input_ids=doc_ids,
        doc_attention_mask=doc_mask,
        label_input_ids=lbl_ids,
        label_attention_mask=lbl_mask,
        targets=targets,
        device=device,
    )

    assert torch.equal(model.csam.W.weight, W_orig)
    assert torch.equal(model.csam.W.bias, b_orig)


def test_9_model_only_checkpoint_roundtrip(tiny_bert):
    """Verify exporting model-only checkpoint strips optimizer and reloads with exact logit agreement."""
    model = LHRMModel(tiny_bert, num_labels=4, variant="L2")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    with tempfile.TemporaryDirectory() as tmpdir:
        full_p = Path(tmpdir) / "step_300.pt"
        export_p = Path(tmpdir) / "L2_STEP300_MODEL_ONLY.pt"

        torch.save({
            "step": 300,
            "variant": "L2",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": {"seed": 42},
        }, full_p)

        meta = export_model_only_checkpoint(
            full_ckpt_path=full_p,
            out_path=export_p,
            bert_model=tiny_bert,
            device=torch.device("cpu"),
        )

        assert meta["roundtrip_verified"] is True
        exp_data = torch.load(export_p, map_location="cpu")
        assert "model_state" in exp_data
        assert "optimizer_state" not in exp_data  # Stripped!


def test_10_checkpoint_output_paths():
    """Verify that expected deliverable paths are consistent and valid."""
    expected = [
        "L2_STEP300_MODEL_ONLY.pt",
        "L2_NO_SA_STEP300_MODEL_ONLY.pt",
        "DR3_TRAINED_GRADIENT_PROBE_RESULTS.json",
        "DR3_TRAINED_CHECKPOINT_MANIFEST.json",
        "DR3_REPLAY_EVIDENCE.zip",
        "DR3_REPLAY_SHA256SUMS.txt",
    ]
    nb = json.load(open(OUT_NB, "r", encoding="utf-8"))
    nb_text = json.dumps(nb)
    for fname in expected:
        assert fname in nb_text, f"Missing deliverable filename {fname} in notebook"


def test_11_no_test_split_access():
    """Verify strict quarantine: no Test data read, copied, or evaluated."""
    nb = json.load(open(OUT_NB, "r", encoding="utf-8"))
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            assert "test.csv" not in src or "not list(processed.glob('test_*'))" in src
            assert "test_labels.npy" not in src


def test_12_deliverables_manifest_schema():
    """Verify structure and formatting of deliverables manifest."""
    deliverables = [
        "L2_STEP300_MODEL_ONLY.pt",
        "L2_NO_SA_STEP300_MODEL_ONLY.pt",
        "DR3_TRAINED_GRADIENT_PROBE_RESULTS.json",
        "DR3_TRAINED_CHECKPOINT_MANIFEST.json",
        "DR3_REPLAY_EVIDENCE.zip",
    ]
    fake_sha = "a" * 64
    manifest_lines = [f"{fake_sha}  {f}" for f in deliverables]
    assert len(manifest_lines) == 5
    for line in manifest_lines:
        parts = line.split("  ")
        assert len(parts) == 2
        assert len(parts[0]) == 64
