import json
from pathlib import Path

from experiments.abba_inflectional_span_20261002 import run, seam


def test_complete_word_gate_and_independent_audits():
    data = run()
    assert data["stats"]["probes"] == 13824
    assert data["stats"]["full_word_compatible"] == 0
    assert data["stats"]["exact_gt38"] == 0
    assert data["best_frontier"]["word_span"]["target_span"] is not None
    assert not data["best_frontier"]["word_span"]["full_word_compatible"]
    assert data["provenance"]["independent_audits"] == [
        "two-pointer", "project validator", "forward/reverse SHA-256"
    ]


def test_boundary_audit_is_not_a_partial_word():
    row = seam("At dawn, the archivist read the note.", "The quiet clerk checked the ledger.")
    assert row["supported_depth"] == 0
    assert row["target_span"] is not None
    assert row["next_complete_word"] != "the"
