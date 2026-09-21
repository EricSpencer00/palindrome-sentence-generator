import json
from pathlib import Path

from experiments.dependency_template_recombination_20260921 import audit, frame, load_frames


def test_independent_audit_accepts_known_palindrome():
    row = audit("An aide rips nine memos; some men inspire Diana.")
    assert row["letters"] == 38
    assert row["two_pointer_exact"]
    assert row["sha256"] == row["reverse_sha256"]  # exact tape has identical hashes


def test_frames_are_role_extractable_and_source_is_not_a_finished_candidate():
    fs = load_frames()
    assert len(fs) >= 20
    assert all(set(("subject", "verb", "object", "adjunct", "source")) <= set(f) for f in fs)
    assert all(f["source"] != "" for f in fs)


def test_run_records_no_unverified_long_winner():
    p = Path("runs/dependency-template-recombination-20260921.json")
    data = json.loads(p.read_text())
    assert data["stats"]["exact_gt38"] == len(data["candidates"])
    assert data["reader_gate"].startswith("closed")
    assert data["novelty_preflight"]["source_sentences_only_frames"]
