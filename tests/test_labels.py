"""
Unit tests for labels.py.
Verifies: no dict repr, case-insensitive lookup, correct text format.
"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lhrm_baseline.labels import build_label_text_arxiv, build_label_text_reuters, build_all_label_texts

ROOT = r"d:\Label-Aware-Hierarchical-Ranking-Model-for-Multi-Label-Text-Classification-main\Label-Aware-Hierarchical-Ranking-Model-for-Multi-Label-Text-Classification-main"
ARXIV_JSON = os.path.join(ROOT, "Label_Definition", "arxiv_labels_definitions.json")
REUTERS_JSON = os.path.join(ROOT, "Label_Definition", "reuters_labels_definitions.json")


def test_arxiv_no_dict_repr():
    """arXiv label text must NOT contain Python dict syntax."""
    import json
    with open(ARXIV_JSON) as f:
        defs = json.load(f)
    
    for key in defs.keys():
        text = build_label_text_arxiv(key, defs)
        assert "{'full_name'" not in text, f"{key}: dict repr leaked: {text[:100]}"
        assert '"full_name"' not in text, f"{key}: dict repr (json str) leaked: {text[:100]}"
        assert len(text) > len(key), f"{key}: text is too short (just the code?): {text}"


def test_arxiv_case_insensitive_lookup():
    """Lowercase label code (from CSV) should match original-case JSON key."""
    import json
    with open(ARXIV_JSON) as f:
        defs = json.load(f)
    
    # cs.AI in JSON; cs.ai in CSV
    text_original = build_label_text_arxiv("cs.AI", defs)
    text_lower = build_label_text_arxiv("cs.ai", defs)
    assert text_original == text_lower, f"Case-insensitive lookup failed: {text_original} vs {text_lower}"


def test_arxiv_text_format():
    """arXiv text should be 'full_name: definition'."""
    import json
    with open(ARXIV_JSON) as f:
        defs = json.load(f)
    
    text = build_label_text_arxiv("cs.AI", defs)
    # Should start with full_name
    assert text.startswith("Computer Science"), f"Unexpected text: {text[:80]}"
    # Should contain a colon separating full_name and definition
    assert ": " in text, f"Missing colon separator: {text[:80]}"


def test_reuters_text_format():
    """Reuters text should be 'label_code: definition'."""
    import json
    with open(REUTERS_JSON) as f:
        defs = json.load(f)
    
    text = build_label_text_reuters("acq", defs)
    assert text.startswith("acq:"), f"Unexpected: {text}"
    assert "acquisition" in text.lower() or "mergers" in text.lower(),         f"Reuters acq text doesn't mention acquisitions: {text}"


def test_build_all_label_texts_no_missing():
    """All 54 arXiv labels should produce valid texts (no AssertionError)."""
    import json, zipfile, csv, io
    
    # Load label order from CSV
    zip_path = os.path.join(ROOT, "Data", "AAPD.zip")
    with zipfile.ZipFile(zip_path) as z:
        with z.open("AAPD/train.csv") as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
            headers = next(iter(reader)).keys()
    label_cols = [h for h in headers if h != "abstract"]
    
    # Should not raise AssertionError
    texts = build_all_label_texts(label_cols, ARXIV_JSON, dataset="aapd")
    assert len(texts) == len(label_cols)
    
    for i, text in enumerate(texts):
        assert "{'full_name'" not in text, f"Label {label_cols[i]}: dict repr: {text[:80]}"
        assert len(text) > 5, f"Label {label_cols[i]}: text too short: {text}"
