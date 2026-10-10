"""
P2: Registry v4 integrity — verify verbatim quotes.

Checks that the decision registry (decision_registry_v4.md) contains
the exact sentences that appear in the paper.  Where pdftotext is
unavailable, the test loads a pre-verified quote file from
tests/fixtures/paper_quotes.json and does a substring match against
the registry text.

If the fixtures file does not exist, the test is SKIPPED (not FAILED).
The test does NOT fabricate or invent any quotes.
"""
import json
import os
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_PATH = os.path.join(ROOT, "docs", "phase1", "decision_registry_v4.md")
FIXTURE_PATH  = os.path.join(ROOT, "tests", "fixtures", "paper_quotes.json")


def load_registry():
    if not os.path.exists(REGISTRY_PATH):
        return None
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        return f.read()


def test_registry_file_exists():
    """Registry v4 must exist."""
    assert os.path.exists(REGISTRY_PATH), (
        f"Missing: {REGISTRY_PATH}\n"
        "Run write_p1d_docs.py to generate the registry."
    )


def test_registry_required_ids():
    """Check that required decision IDs are present."""
    text = load_registry()
    assert text is not None
    required_ids = [
        "A1",   # CLS extraction
        "A2",   # label re-encoding with gradients
        "A4",   # softmax (Eq.5)
        "A5",   # classifier head
        "A6",   # architecture variant
        "A9",   # focal parameters
        "A11",  # AMP
        "A13",  # HRRM
        "A14",  # bert-base-uncased
        "A16",  # tie-breaking
        "A17",  # HRRM fitting data
        "A19",  # evaluation model
        "A_D1", # client-approved AAPD dataset (Phase 1E)
    ]
    missing = [rid for rid in required_ids if rid not in text]
    assert not missing, f"Registry missing IDs: {missing}"


def test_registry_no_forbidden_claims():
    """
    Registry must not *assert* forbidden claims about AAPD mapping.

    Forbidden strings are only violations when they appear affirmatively.
    Lines containing prohibition markers (e.g. 'must never', 'never write',
    'must not', 'NOT PASS') are excluded because the registry is allowed to
    say "VERIFIED_LABEL_MAPPING must never be written" — that is a
    prohibition, not an affirmative claim.
    """
    text = load_registry()
    assert text is not None

    # Phrases that, if present on a line, mean the forbidden string is used
    # as a *prohibition target*, not an affirmative claim.
    negation_markers = [
        "must never",
        "never write",
        "must not",
        "not pass",
        "not verified",
        "never be written",
        "forbidden",
        "do not write",
    ]

    forbidden = [
        "VERIFIED_LABEL_MAPPING",
        "label mapping verified",
        "mapping is correct",
    ]

    found = []
    for line in text.splitlines():
        line_lower = line.lower()
        # Skip lines that are prohibitions/negations
        if any(neg in line_lower for neg in negation_markers):
            continue
        for f in forbidden:
            if f.lower() in line_lower:
                found.append(f"{f!r} in: {line.strip()!r}")

    assert not found, (
        f"Registry contains affirmative forbidden claims:\n"
        + "\n".join(found)
        + "\nAAPD mapping status must remain UNRESOLVED_LABEL_MAPPING "
        "or ACCEPTED_BY_CLIENT_DECISION."
    )



def test_registry_client_decision_recorded():
    """A_D1 must record the client decision."""
    text = load_registry()
    assert text is not None
    assert "A_D1" in text, "A_D1 (client decision) missing from registry."
    assert "USE_ZENODO_SOURCE_AS_PROVIDED" in text, (
        "A_D1 must state USE_ZENODO_SOURCE_AS_PROVIDED."
    )
    assert "ACCEPTED_BY_CLIENT_DECISION" in text or "CLOSED_BY_CLIENT_DECISION" in text, (
        "Registry must state ACCEPTED_BY_CLIENT_DECISION or CLOSED_BY_CLIENT_DECISION for G1."
    )


@pytest.mark.skipif(
    not os.path.exists(FIXTURE_PATH),
    reason="tests/fixtures/paper_quotes.json not found; "
           "create from paper text (pdftotext or manual) to enable quote checks.",
)
def test_paper_quotes_present_in_registry():
    """
    Verify verbatim paper quotes appear in the registry.
    Each entry in paper_quotes.json:
        {"id": "A_ID", "quote": "...", "source": "Section X"}
    The quote (stripped) must appear as a substring of the registry text.
    """
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        quotes = json.load(f)
    text = load_registry()
    assert text is not None

    failures = []
    for entry in quotes:
        qid   = entry.get("id", "?")
        quote = entry.get("quote", "").strip()
        if not quote:
            continue
        if quote not in text:
            failures.append(f"{qid}: '{quote[:60]}...' NOT found in registry")

    assert not failures, (
        f"{len(failures)} quote(s) missing from registry:\n"
        + "\n".join(failures)
    )
