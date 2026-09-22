"""Regression checks for the sentence-boundary-only 568 expansion."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "incumbent-568-sentence-boundary-clause-intersection-20261002.json"


def test_parent_frontier_and_rejected_evidence_are_retained() -> None:
    payload = json.loads(ARTIFACT.read_text())
    assert payload["parent"]["letters"] == 568
    assert payload["parent"]["sha256"] == "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
    assert [entry["letters"] for entry in payload["preserved_frontier"]] == [568, 560, 558, 556]
    assert {entry["letters"] for entry in payload["rejected_evidence"]} == {592, 596}
    assert payload["comparison_evidence"]["source_commit"] == "9cb68296"


def test_sentence_spans_clause_residual_and_full_text_gate() -> None:
    payload = json.loads(ARTIFACT.read_text())
    row = payload["rows"][0]
    seam = row["reviewer_seam"]
    assert seam["requested_normalized_windows"] == [[91, 135], [433, 477]]
    assert seam["raw_after_letter_boundaries"] == [[119, 178], [598, 660]]
    assert seam["sentence_boundary_spans"] == [[121, 180], [600, 662]]
    live = row["live_bidirectional_residual"]
    assert live["final_residual"] == ""
    assert live["committed_character_contradictions"] == 0
    assert row["independent_audit"]["normalized_letters"] == 592
    assert row["independent_audit"]["two_pointer_exact"] is True
    assert row["independent_audit"]["sha_equal"] is True
    assert row["full_text_gate"] == {
        "lowercase_after_terminal": False,
        "comma_splice": False,
        "fragment": False,
        "all_emitted_units_complete": True,
        "inserted_unit_duplicated": False,
        "introduced_frame_repetition": False,
        "worsened_inherited_repetition": False,
        "status": "passed structural full-text gate; readability remains unpromoted",
    }
    assert row["grammar_novelty"]["complete_clause_count_per_side"] == 4
    assert len(row["attempts"]) <= 8
    assert row["attempts"][0]["status"] == "accepted"
