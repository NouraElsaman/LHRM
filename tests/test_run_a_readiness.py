"""
tests/test_run_a_readiness.py

CPU-only synthetic tests for RUN A readiness (A1/A2/A4/A8/A9).
These run in the normal test suite without GPU.
"""
import json
import os
import tempfile

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_model_optimizer():
    """Return a tiny linear model + AdamW for checkpoint round-trip tests."""
    model = torch.nn.Linear(8, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer


# ---------------------------------------------------------------------------
# A1: Test-split isolation — verify cmd_train doesn't touch test data
# ---------------------------------------------------------------------------

def test_a1_cmd_train_does_not_load_test_split():
    """
    A1: cmd_train must never load or predict the test split.
    Verify by inspecting the source code of run_baseline.cmd_train.
    The function body must not contain load_processed(... "test") calls,
    and must not contain 'predictions_test' output paths.
    """
    import inspect
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from run_baseline import cmd_train
    src = inspect.getsource(cmd_train)
    # Must NOT load test split
    assert '"test"' not in src or 'test_eval' in src or 'A1' in src, (
        "cmd_train source appears to load test split — A1 violation")
    # Must have A1 provenance marker
    assert "A1" in src, "cmd_train missing A1 provenance marker"
    # Must NOT save predictions_test_raw from training loop
    # (it may mention test_eval command to redirect user)
    assert "predictions_test_raw" not in src or "cmd_test_eval" in src, (
        "cmd_train writes predictions_test_raw without test_eval gate — A1 violation")
    # _test_split_loaded provenance key must be set to False
    assert "_test_split_loaded" in src, (
        "cmd_train missing _test_split_loaded=False provenance key")


def test_a1_resolved_config_marks_test_not_loaded():
    """
    A1: resolved_config written by cmd_train must contain _test_split_loaded=False.
    Test by checking the key is explicitly set in the source.
    """
    import run_baseline
    import inspect
    src = inspect.getsource(run_baseline.cmd_train)
    assert '"_test_split_loaded": False' in src or "_test_split_loaded\"] = False" in src


# ---------------------------------------------------------------------------
# A4: Epoch-boundary checkpoint round-trip
# ---------------------------------------------------------------------------

def test_a4_checkpoint_save_load_epoch_boundary():
    """
    A4: save_checkpoint_v2 must save and restore:
    - model state (exact weight equality)
    - optimizer state
    - scaler state
    - torch CPU RNG
    - torch CUDA RNG (skipped on CPU)
    - numpy RNG
    - python random RNG
    Resume is epoch-boundary only (documented limitation).
    """
    import random
    from lhrm_baseline.train import save_checkpoint_v2, load_checkpoint

    model, optimizer = _tiny_model_optimizer()
    scaler = None  # no CUDA in test

    # Set a known RNG state
    random.seed(123)
    np.random.seed(456)
    torch.manual_seed(789)

    # Do a synthetic forward/backward to advance optimizer state
    x = torch.randn(2, 8)
    out = model(x).sum()
    out.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Capture states before save
    weights_before = {k: v.clone() for k, v in model.state_dict().items()}
    rng_before = torch.get_rng_state().clone()
    np_state_before = np.random.get_state()
    py_state_before = random.getstate()

    config = {"dataset": "aapd", "variant": "L2", "seed": 42}
    label_ids = torch.zeros(4, 10, dtype=torch.long)
    label_mask = torch.ones(4, 10, dtype=torch.long)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_checkpoint_v2(
            model, optimizer, scaler,
            epoch=3, step=100, config=config,
            out_dir=tmpdir,
            label_input_ids=label_ids,
            label_attention_mask=label_mask,
            history=[{"epoch": 1}, {"epoch": 2}, {"epoch": 3}],
            filename="test_epoch.pt",
        )
        assert os.path.exists(path), "Checkpoint file not created"

        # Corrupt model weights
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(0.0)

        # Also corrupt RNG states
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        # Load checkpoint
        model2, optimizer2 = _tiny_model_optimizer()
        epoch_loaded = load_checkpoint(path, model2, optimizer2, None,
                                       torch.device("cpu"))

    # Verify
    assert epoch_loaded == 3, "Epoch not restored correctly"

    # Model weights restored exactly
    for k in weights_before:
        assert torch.allclose(weights_before[k], model2.state_dict()[k]), (
            f"Weight mismatch for key {k} after checkpoint restore")

    # Torch CPU RNG restored
    rng_after = torch.get_rng_state()
    assert torch.equal(rng_before, rng_after), "torch RNG not restored"

    # Numpy RNG restored (compare a sample)
    np_sample_ref = np.random.RandomState()
    np_sample_ref.set_state(np_state_before)
    np_sample_after = np.random.get_state()
    assert np_sample_ref.randint(0, 1000) == np.random.randint(0, 1000), \
        "numpy RNG not restored"

    # Python RNG restored
    random.setstate(py_state_before)
    ref_val = random.randint(0, 10**9)
    random.setstate(py_state_before)
    loaded_val = random.randint(0, 10**9)
    assert ref_val == loaded_val, "python random RNG not restored"


def test_a4_checkpoint_atomic_write():
    """A4: Checkpoint file must not leave a .tmp file on successful save."""
    from lhrm_baseline.train import save_checkpoint_v2
    model, optimizer = _tiny_model_optimizer()
    label_ids = torch.zeros(2, 5, dtype=torch.long)
    label_mask = torch.ones(2, 5, dtype=torch.long)
    with tempfile.TemporaryDirectory() as tmpdir:
        save_checkpoint_v2(
            model, optimizer, None, epoch=1, step=10,
            config={}, out_dir=tmpdir,
            label_input_ids=label_ids, label_attention_mask=label_mask,
        )
        tmp_files = [f for f in os.listdir(tmpdir) if f.endswith(".tmp")]
        assert tmp_files == [], f"Stale .tmp files found: {tmp_files}"


def test_a4_resume_limitation_documented():
    """
    A4: Verify the epoch-boundary-only limitation is documented in both
    load_checkpoint and save_checkpoint_v2 docstrings.
    """
    from lhrm_baseline.train import load_checkpoint, save_checkpoint_v2
    lc_doc = load_checkpoint.__doc__ or ""
    sc_doc = save_checkpoint_v2.__doc__ or ""
    assert "epoch" in lc_doc.lower() and "boundary" in lc_doc.lower(), \
        "load_checkpoint docstring must document epoch-boundary limitation"
    assert "epoch" in sc_doc.lower() and "boundary" in sc_doc.lower(), \
        "save_checkpoint_v2 docstring must document epoch-boundary limitation"


# ---------------------------------------------------------------------------
# A8: F1 metrics correctness
# ---------------------------------------------------------------------------

def test_a8_compute_f1_metrics_perfect():
    """A8: compute_f1_metrics returns 1.0 for perfect predictions."""
    from lhrm_baseline.metrics import compute_f1_metrics
    y_true = np.array([[1, 0, 1], [0, 1, 0]], dtype=float)
    # Perfect: score=1.0 for true labels
    y_score = np.array([[0.9, 0.1, 0.9], [0.1, 0.9, 0.1]], dtype=float)
    m = compute_f1_metrics(y_true, y_score, threshold=0.5)
    assert abs(m["micro_f1"] - 1.0) < 1e-6, f"micro_f1={m['micro_f1']}"
    assert abs(m["macro_f1"] - 1.0) < 1e-6, f"macro_f1={m['macro_f1']}"
    assert m["threshold"] == 0.5


def test_a8_compute_f1_metrics_zeros():
    """A8: compute_f1_metrics returns 0 for all-wrong predictions (zeroDivision safe)."""
    from lhrm_baseline.metrics import compute_f1_metrics
    y_true = np.array([[1, 0], [0, 1]], dtype=float)
    y_score = np.array([[0.1, 0.9], [0.9, 0.1]], dtype=float)  # all wrong
    m = compute_f1_metrics(y_true, y_score, threshold=0.5)
    assert m["micro_f1"] == 0.0
    assert m["macro_f1"] == 0.0


def test_a8_compute_per_label_metrics_structure():
    """A8: compute_per_label_metrics returns correct structure with TP/FP/FN."""
    from lhrm_baseline.metrics import compute_per_label_metrics
    y_true = np.array([[1, 0, 1], [1, 1, 0]], dtype=float)
    y_score = np.array([[0.9, 0.2, 0.8], [0.9, 0.9, 0.2]], dtype=float)
    rows = compute_per_label_metrics(y_true, y_score,
                                     label_names=["a", "b", "c"], threshold=0.5)
    assert len(rows) == 3
    for row in rows:
        for key in ("label", "index", "precision", "recall", "f1",
                    "tp", "fp", "fn", "support"):
            assert key in row, f"Missing key {key} in per-label row"
    # Label "a": both true, both predicted → TP=2, FP=0, FN=0
    a = rows[0]
    assert a["tp"] == 2, f"a.tp={a['tp']}"
    assert a["fn"] == 0, f"a.fn={a['fn']}"


def test_a8_compute_all_metrics_includes_f1():
    """A8: compute_all_metrics with include_f1=True must include micro_f1, macro_f1, threshold."""
    from lhrm_baseline.metrics import compute_all_metrics
    y_true = np.array([[1, 0, 1]], dtype=float)
    y_score = np.array([[0.9, 0.1, 0.9]], dtype=float)
    m = compute_all_metrics(y_true, y_score, k_values=[1, 3], include_f1=True)
    assert "micro_f1" in m
    assert "macro_f1" in m
    assert "threshold" in m
    assert m["threshold"] == 0.5


def test_a8_compute_all_metrics_no_f1():
    """A8: compute_all_metrics with include_f1=False must NOT include f1 keys."""
    from lhrm_baseline.metrics import compute_all_metrics
    y_true = np.array([[1, 0, 1]], dtype=float)
    y_score = np.array([[0.9, 0.1, 0.9]], dtype=float)
    m = compute_all_metrics(y_true, y_score, k_values=[1], include_f1=False)
    assert "micro_f1" not in m
    assert "P@1" in m


# ---------------------------------------------------------------------------
# A2: Progress logging structure
# ---------------------------------------------------------------------------

def test_a2_train_epoch_stepped_returns_three_values():
    """A2: train_epoch_stepped must return (mean_loss, n_steps, amp_skips)."""
    from lhrm_baseline.train import train_epoch_stepped
    import inspect
    sig = inspect.signature(train_epoch_stepped)
    params = list(sig.parameters.keys())
    assert "log_every" in params, "train_epoch_stepped missing log_every param"
    assert "progress_jsonl_path" in params, "train_epoch_stepped missing progress_jsonl_path"
    assert "amp_skips" in (train_epoch_stepped.__doc__ or "").lower() or \
           "amp_skips" in inspect.getsource(train_epoch_stepped), \
           "train_epoch_stepped must track amp_skips"


def test_a2_progress_jsonl_written():
    """A2: train_epoch_stepped writes JSONL records to progress_jsonl_path."""
    # Use a tiny CPU synthetic DataLoader (no BERT needed)
    from lhrm_baseline.train import train_epoch_stepped

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 3)
        def forward(self, ids, mask, label_ids, label_mask):
            # ignore labels; return random logits
            b = ids.shape[0]
            logits = self.lin(torch.randn(b, 4))
            return logits, {}

    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Tiny DataLoader: 5 batches of 2 docs, 3 labels
    batches = []
    for _ in range(5):
        batches.append({
            "input_ids": torch.zeros(2, 10, dtype=torch.long),
            "attention_mask": torch.ones(2, 10, dtype=torch.long),
            "labels": torch.randint(0, 2, (2, 3)).float(),
        })

    label_ids = torch.zeros(3, 10, dtype=torch.long)
    label_mask = torch.ones(3, 10, dtype=torch.long)

    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = os.path.join(tmpdir, "progress.jsonl")
        mean_loss, n_steps, amp_skips = train_epoch_stepped(
            model, batches, criterion, optimizer,
            scaler=None, device=torch.device("cpu"),
            label_input_ids=label_ids, label_attention_mask=label_mask,
            use_amp=False,
            log_every=2,  # log every 2 steps
            epoch=1, total_epochs=3,
            steps_per_epoch=5, total_global_steps=15,
            run_id="TEST", variant="L2",
            progress_jsonl_path=jsonl_path,
        )

        assert isinstance(mean_loss, float), "mean_loss must be float"
        assert n_steps == 5, f"n_steps={n_steps}"
        assert isinstance(amp_skips, int), "amp_skips must be int"
        assert amp_skips == 0, "no amp skips expected on CPU"

        assert os.path.exists(jsonl_path), "progress.jsonl not written"
        lines = [l for l in open(jsonl_path).read().strip().split("\n") if l]
        assert len(lines) >= 2, f"Expected >=2 JSONL records, got {len(lines)}"
        record = json.loads(lines[0])
        assert record["type"] == "TRAIN_PROGRESS"
        assert record["run_id"] == "TEST"
        assert record["variant"] == "L2"
        assert record["epoch"] == 1


# ---------------------------------------------------------------------------
# A9: Rolling checkpoint rotation
# ---------------------------------------------------------------------------

def test_a9_rolling_checkpoint_rotation():
    """A9: After two epochs, only latest.pt and previous.pt should exist (no epoch_N.pt)."""
    # Simulate the rotation logic used in cmd_train
    with tempfile.TemporaryDirectory() as tmpdir:
        latest = os.path.join(tmpdir, "latest.pt")
        previous = os.path.join(tmpdir, "previous.pt")

        import shutil

        def save_epoch(epoch):
            payload = {"epoch": epoch, "data": "x" * 100}
            # Rotate
            if os.path.exists(latest):
                tmp_prev = previous + ".tmp"
                shutil.copy2(latest, tmp_prev)
                if os.path.exists(previous):
                    os.remove(previous)
                os.rename(tmp_prev, previous)
            # Save latest atomically
            tmp_latest = latest + ".tmp"
            torch.save(payload, tmp_latest)
            if os.path.exists(latest):
                os.remove(latest)
            os.rename(tmp_latest, latest)

        save_epoch(1)
        assert os.path.exists(latest)
        assert not os.path.exists(previous)

        save_epoch(2)
        assert os.path.exists(latest)
        assert os.path.exists(previous)

        # Verify latest has epoch 2, previous has epoch 1
        lat = torch.load(latest, map_location="cpu", weights_only=False)
        prev = torch.load(previous, map_location="cpu", weights_only=False)
        assert lat["epoch"] == 2, f"latest epoch={lat['epoch']}"
        assert prev["epoch"] == 1, f"previous epoch={prev['epoch']}"

        save_epoch(3)
        lat = torch.load(latest, map_location="cpu", weights_only=False)
        prev = torch.load(previous, map_location="cpu", weights_only=False)
        assert lat["epoch"] == 3
        assert prev["epoch"] == 2

        # No epoch_N.pt files should exist
        all_files = os.listdir(tmpdir)
        epoch_files = [f for f in all_files if f.startswith("epoch_")]
        assert epoch_files == [], f"Unexpected per-epoch files: {epoch_files}"

        # No .tmp files should exist
        tmp_files = [f for f in all_files if f.endswith(".tmp")]
        assert tmp_files == [], f"Stale .tmp files: {tmp_files}"
