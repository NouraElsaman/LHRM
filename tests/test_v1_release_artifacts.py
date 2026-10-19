"""
Release Validation Test Suite for LHRM R_AUTHOR V1.0.

Validates all 15 Phase B Quality Assurance requirements:
1. Notebook JSON parsing and cell structural integrity.
2. AST syntax check on all code cells.
3. Generator reproducibility and exact source match with audited notebook.
4. CPU-first RNG resume restoration logic.
5. Checkpoint metadata and expected hashes.
6. 54-label schema and original order SHA-256.
7. Verification of frozen Test metrics.
8. Exact mathematical recalculation of Micro-F1 from TP/FP/FN.
9. Verification that HRRM comparison is Dev-only (no Test evaluation).
10. Assurance that inference notebooks contain zero training operations.
11. Assurance that inference notebooks do not access AAPD Test data.
12. Three Colab locked smoke-test input SHA-256 fingerprints and Top-1 references.
13. Markdown documentation link integrity.
14. Git privacy audit: .gitignore excludes handoffs and heavy binary blobs.
"""

import ast
import csv
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

NOTEBOOKS = [
    REPO_ROOT / "notebooks/lhrm_r_author_aapd_train_evaluate_kaggle.ipynb",
    REPO_ROOT / "notebooks/lhrm_r_author_aapd_train_evaluate_kaggle_RNG_RESUME_AUDITED.ipynb",
    REPO_ROOT / "notebooks/LHRM_R_AUTHOR_EPOCH6_AAPD_TEST_ONLY_FROZEN.ipynb",
    REPO_ROOT / "notebooks/LHRM_R_AUTHOR_EPOCH6_CLIENT_INFERENCE_KAGGLE.ipynb",
    REPO_ROOT / "notebooks/LHRM_R_AUTHOR_EPOCH6_CLIENT_INFERENCE_COLAB_SMOKE_TEXT_LOCKED_v1_1.ipynb",
]

EXPECTED_LABEL_ORDER_HASH = "a3d662f3920259a94ed61d83f98f21cabe7f908d98a5662ccd9adc2413cc000b"
EXPECTED_CHECKPOINT_SHA256 = "988bb64fca2b8ea21030585de7268a1cfb1fc8676b91b4d6e6f38c4f22c95643"
EXPECTED_COLAB_SUPPORT_SHA256 = "e6f9dec292e49e8eec48c3c47f56d138897953e081b06507f8fa99a615214031"


class TestV1ReleaseNotebooks:

    @pytest.mark.parametrize("nb_path", NOTEBOOKS)
    def test_notebook_exists_and_loads_valid_json(self, nb_path):
        assert nb_path.exists(), f"Notebook missing: {nb_path}"
        with open(nb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "cells" in data, f"Notebook missing 'cells': {nb_path}"
        assert len(data["cells"]) > 0, f"Notebook has 0 cells: {nb_path}"

    @pytest.mark.parametrize("nb_path", NOTEBOOKS)
    def test_notebook_code_cells_valid_python_ast(self, nb_path):
        with open(nb_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for idx, cell in enumerate(data["cells"]):
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            # Clean IPython magics (!, %, get_ipython) for AST parser
            cleaned_lines = []
            for line in src.splitlines():
                stripped = line.strip()
                if stripped.startswith("!") or stripped.startswith("%") or "get_ipython()" in stripped:
                    cleaned_lines.append("# " + line)
                else:
                    cleaned_lines.append(line)
            clean_code = "\n".join(cleaned_lines)
            try:
                ast.parse(clean_code)
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {nb_path.name} cell {idx}: {e}")


class TestV1ReproducibilityAndRNGResume:

    def test_generator_produces_exact_source_match_with_audited_notebook(self):
        import importlib.util
        gen_script = REPO_ROOT / "scripts/gen_kaggle_notebook_r_author.py"
        spec = importlib.util.spec_from_file_location("gen", gen_script)
        gen_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen_mod)

        generated_nb = gen_mod.build_notebook()

        audited_path = REPO_ROOT / "notebooks/lhrm_r_author_aapd_train_evaluate_kaggle_RNG_RESUME_AUDITED.ipynb"
        with open(audited_path, "r", encoding="utf-8") as f:
            audited_nb = json.load(f)

        assert len(generated_nb["cells"]) == len(audited_nb["cells"])
        for i in range(len(audited_nb["cells"])):
            src_gen = "".join(generated_nb["cells"][i].get("source", []))
            src_aud = "".join(audited_nb["cells"][i].get("source", []))
            assert src_gen == src_aud, f"Mismatch in cell {i} between generator and audited notebook!"

    def test_cpu_first_rng_resume_in_training_notebooks(self):
        for nb_path in [
            REPO_ROOT / "notebooks/lhrm_r_author_aapd_train_evaluate_kaggle.ipynb",
            REPO_ROOT / "notebooks/lhrm_r_author_aapd_train_evaluate_kaggle_RNG_RESUME_AUDITED.ipynb"
        ]:
            with open(nb_path, "r", encoding="utf-8") as f:
                nb = json.load(f)
            code_sources = ["".join(c.get("source", [])) for c in nb["cells"] if c.get("cell_type") == "code"]
            all_code = "\n".join(code_sources)

            assert 'map_location="cpu"' in all_code, f"Missing map_location='cpu' in {nb_path.name}"
            assert 'torch.set_rng_state(ckpt["torch_rng_state"].cpu())' in all_code, f"Missing CPU tensor cast in {nb_path.name}"
            assert 'torch.cuda.set_rng_state(ckpt["cuda_rng_state"].cpu())' in all_code, f"Missing CUDA CPU tensor cast in {nb_path.name}"


class TestV1ScientificResultsIntegrity:

    def test_54_labels_schema_and_order_hash(self):
        csv_path = REPO_ROOT / "docs/results/test_per_label_metrics_54.csv"
        assert csv_path.exists(), f"Missing {csv_path}"

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            labels = [row["label"] for row in reader]

        assert len(labels) == 54, f"Expected 54 labels, got {len(labels)}"
        computed_hash = hashlib.sha256(json.dumps(labels).encode()).hexdigest()
        assert computed_hash == EXPECTED_LABEL_ORDER_HASH, (
            f"Label order hash mismatch: expected {EXPECTED_LABEL_ORDER_HASH}, got {computed_hash}"
        )

    def test_test_summary_metrics_values(self):
        json_path = REPO_ROOT / "docs/results/test_summary_metrics.json"
        assert json_path.exists(), f"Missing {json_path}"

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["split"] == "TEST"
        assert data["n_documents"] == 1000
        assert data["n_labels"] == 54
        assert data["checkpoint_epoch"] == 6
        assert data["checkpoint_step"] == 40380
        assert data["selection_P@1"] == 0.895
        assert data["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256
        assert data["label_order_hash"] == EXPECTED_LABEL_ORDER_HASH

        m = data["metrics"]
        assert pytest.approx(m["P@1"], abs=1e-4) == 0.8490
        assert pytest.approx(m["micro_f1"], abs=1e-6) == 0.727787
        assert pytest.approx(m["macro_f1"], abs=1e-6) == 0.555253

    def test_micro_f1_mathematical_recalculation(self):
        json_path = REPO_ROOT / "docs/results/test_summary_metrics.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        gc = data["global_counts"]
        tp = gc["tp"]
        fp = gc["fp"]
        fn = gc["fn"]

        assert tp == 1671
        assert fp == 500
        assert fn == 750

        recalc_p = tp / (tp + fp)
        recalc_r = tp / (tp + fn)
        recalc_f1 = (2 * tp) / (2 * tp + fp + fn)

        assert pytest.approx(data["metrics"]["micro_precision"], abs=1e-12) == recalc_p
        assert pytest.approx(data["metrics"]["micro_recall"], abs=1e-12) == recalc_r
        assert pytest.approx(data["metrics"]["micro_f1"], abs=1e-12) == recalc_f1

    def test_hrrm_heldout_comparison_is_dev_only(self):
        hrrm_path = REPO_ROOT / "docs/results/hrrm_heldout_comparison.json"
        assert hrrm_path.exists(), f"Missing {hrrm_path}"

        with open(hrrm_path, "r", encoding="utf-8") as f:
            hrrm = json.load(f)

        assert hrrm["protocol"] == "NON_LEAKY_DEV_PARTITION"
        assert hrrm["n_fit_docs"] == 200
        assert hrrm["n_eval_docs"] == 200
        assert hrrm["delta"]["P@1"] == -0.005
        assert pytest.approx(hrrm["delta"]["micro_f1"], abs=1e-6) == -0.000949
        assert "test" not in hrrm["protocol"].lower()

    def test_15_epoch_metrics_continuity(self):
        epoch_csv = REPO_ROOT / "docs/results/epoch_metrics.csv"
        assert epoch_csv.exists(), f"Missing {epoch_csv}"

        with open(epoch_csv, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))

        assert len(reader) == 15, f"Expected 15 epochs, got {len(reader)}"
        for i, row in enumerate(reader, start=1):
            assert int(row["epoch"]) == i
            assert int(row["global_steps"]) == i * 6730
        assert int(reader[-1]["global_steps"]) == 100950


class TestV1InferenceSafetyAndColabSmoke:

    @pytest.mark.parametrize("nb_path", [
        REPO_ROOT / "notebooks/LHRM_R_AUTHOR_EPOCH6_CLIENT_INFERENCE_KAGGLE.ipynb",
        REPO_ROOT / "notebooks/LHRM_R_AUTHOR_EPOCH6_CLIENT_INFERENCE_COLAB_SMOKE_TEXT_LOCKED_v1_1.ipynb",
    ])
    def test_inference_notebooks_do_not_train_or_access_test_data(self, nb_path):
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        for idx, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            assert "loss.backward" not in src, f"Found training loss.backward in {nb_path.name} cell {idx}"
            assert "optimizer.step" not in src, f"Found training optimizer.step in {nb_path.name} cell {idx}"
            assert "test.csv" not in src, f"Found code reference to test.csv in {nb_path.name} cell {idx}"
            assert "test_data" not in src, f"Found code reference to test_data in {nb_path.name} cell {idx}"

    def test_colab_locked_smoke_test_fingerprints_and_outputs(self):
        colab_nb = REPO_ROOT / "notebooks/LHRM_R_AUTHOR_EPOCH6_CLIENT_INFERENCE_COLAB_SMOKE_TEXT_LOCKED_v1_1.ipynb"
        text = colab_nb.read_text(encoding="utf-8")

        # 3 Input SHA prefixes locked in Colab Cell 4
        assert "66650e1988e209be4c70d9f2ee8b1e4e7c399b1971d30fdd666a4eb57c6640ef" in text
        assert "8c28e37b6a814e44b8f5701bfeb6ee95d6f91c0a13b6d1ea3c4182d35c94054c" in text
        assert "9b148c0704239f7485fb077496dce78f35ce19240b2abd31961a41ea1fd712fe" in text

        # Check expected Top-1 predictions against verified execution outputs
        expected_top1 = ["cond-mat.stat-mech", "cs.lg", "cond-mat.stat-mech"]
        # Verify that inference_demo from evidence produces these exact Top-1 labels
        import zipfile
        import io
        p_colab_support = Path(r"C:\Users\DELL\Downloads\LHRM_R_AUTHOR_COLAB_CLIENT_SUPPORT_20261015.zip")
        if p_colab_support.exists():
            with zipfile.ZipFile(p_colab_support, "r") as z:
                ev_bytes = z.read("EVIDENCE.zip")
                with zipfile.ZipFile(io.BytesIO(ev_bytes), "r") as ez:
                    demo = json.loads(ez.read("inference_demo.json").decode("utf-8"))
                    actual_top1 = [d["neural_top_k"][0]["label"] for d in demo]
                    assert actual_top1 == expected_top1, f"Top-1 predictions mismatch: {actual_top1} vs {expected_top1}"


class TestV1DocumentationAndGitPrivacy:

    def test_markdown_documentation_links(self):
        for doc in [
            REPO_ROOT / "README.md",
            REPO_ROOT / "docs/results/R_AUTHOR_AAPD_V1_RESULTS.md",
            REPO_ROOT / "docs/experiments/RESEARCH_BACKLOG.md",
        ]:
            text = doc.read_text(encoding="utf-8")
            links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', text)
            for label, link in links:
                if link.startswith("http") or link.startswith("#"):
                    continue
                clean = link.split("#")[0]
                target = (doc.parent / clean).resolve() if not clean.startswith("/") else (REPO_ROOT / clean.lstrip("/")).resolve()
                assert target.exists(), f"Broken link in {doc.name}: [{label}]({link}) -> {target}"

    def test_gitignore_excludes_handoffs(self):
        gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        assert "docs/handoffs/" in gitignore, "Missing docs/handoffs/ in .gitignore"

        # Check git ignore status of handoff file
        result = subprocess.run(
            ["git", "check-ignore", "docs/handoffs/LHRM_R_AUTHOR_AG_IDE_REPO_SYNC_HANDOFF_20261015.md"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, "docs/handoffs/ is not properly ignored by git!"
