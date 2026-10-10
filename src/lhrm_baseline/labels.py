"""
Label text builder.
Produces the text Y_i for each label to be encoded by BERT.
Fixes the dict-repr bug (R9) and case-sensitivity issue (D2).

PAPER reference: Section III.A, Table 2 — "full title + enriched definition"
"""
from __future__ import annotations
import json
import os
from typing import Dict, List, Optional


def load_definitions(json_path: str) -> Dict[str, dict | str]:
    """Load the label definitions JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_label_text_arxiv(label_code: str, defs: Dict[str, dict | str]) -> str:
    """
    Build label text for arXiv/AAPD labels.
    Each value in arxiv_labels_definitions.json is a dict with 'full_name' and 'definition'.
    
    PAPER format: "full_name: definition"  (Table 2 examples)
    Returns: e.g. "Computer Science - Artificial Intelligence: Research on intelligent systems..."
    
    Bug fixed (R9): original code used f"{lab}: {context.get(lab,'')}" which encoded
    the Python dict repr. We now access .full_name and .definition separately.
    """
    # Case-insensitive lookup (D2: CSV headers are lowercase, JSON keys are original-case)
    lookup = {k.lower(): k for k in defs.keys()}
    canonical_key = lookup.get(label_code.lower())
    
    if canonical_key is None:
        # Fallback: use just the label code
        return label_code
    
    entry = defs[canonical_key]
    
    if isinstance(entry, dict):
        # arxiv_labels_definitions.json: {"full_name": "...", "definition": "..."}
        full_name = entry.get("full_name", label_code)
        definition = entry.get("definition", "")
        if definition:
            return f"{full_name}: {definition}"
        return full_name
    elif isinstance(entry, str):
        # reuters_labels_definitions.json: plain string values
        return f"{label_code}: {entry}"
    else:
        return label_code


def build_label_text_reuters(label_code: str, defs: Dict[str, dict | str]) -> str:
    """
    Build label text for Reuters labels.
    Each value in reuters_labels_definitions.json is a plain string.
    
    PAPER format: "label_code: definition"
    """
    if label_code in defs:
        definition = defs[label_code]
        if isinstance(definition, str) and definition:
            return f"{label_code}: {definition}"
    return label_code


def build_all_label_texts(
    label_list: List[str],
    defs_path: str,
    dataset: str = "aapd"
) -> List[str]:
    """
    Build the list of label texts Y_1..Y_k in the same order as label_list.
    
    Args:
        label_list: Ordered list of label codes (from labels.json)
        defs_path: Path to the label definitions JSON file
        dataset: 'aapd' or 'reuters'
    
    Returns:
        List of strings, one per label, in the same order.
    
    Raises:
        AssertionError if any label produces an empty or code-only text.
    """
    defs = load_definitions(defs_path)
    texts = []
    missing = []
    
    for label in label_list:
        if dataset == "aapd":
            text = build_label_text_arxiv(label, defs)
        else:
            text = build_label_text_reuters(label, defs)
        texts.append(text)
        
        # Track which labels fell back to code-only
        lookup = {k.lower(): k for k in defs.keys()}
        if lookup.get(label.lower()) is None:
            missing.append(label)
    
    if missing:
        raise AssertionError(
            f"build_all_label_texts: {len(missing)} labels not found in definitions "
            f"(case-insensitive): {missing}"
        )
    
    # Verify no Python dict syntax leaked through
    for i, text in enumerate(texts):
        DICT_REPR_MARKER = "{" + "'full_name'"
        assert DICT_REPR_MARKER not in text, (
            f"Label {label_list[i]}: dict repr leaked into text: {text[:100]}"
        )
    
    return texts
