"""Regression checks for the reviewer-selected reflected shell replacement."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "incumbent-568-repeated-shell-intersection-20261002.json"


def test_reviewer_offsets_and_rejected_children_are_preserved() -> None:
    payload = json.loads(ARTIFACT.read_text())
    assert payload["parent"]["letters"] == 568
    assert payload["parent"]["sha256"] == "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
    assert [entry["letters"] for entry in payload["preserved_frontier"]] == [568, 560, 558, 556]
    assert {entry["letters"] for entry in payload["rejected_evidence"]} == {596}
    assert payload["comparison_evidence"]["source_commit"] == "9cb68296"
    row = payload["rows"][0]
    seam = row["reviewer_seam"]
    assert seam["requested_normalized_windows"] == [[170, 197], [469, 496]]
    assert seam["reflected_normalized_windows"] == [[72, 99], [371, 398]]
    assert seam["recomputed_raw_after_letter_spans"] == [[228, 267], [650, 690], [92, 130], [507, 548]]


def test_reflected_shell_residuals_and_full_text_gate_close() -> None:
    payload = json.loads(ARTIFACT.read_text())
    row = payload["rows"][0]
    live = row["live_bidirectional_residuals"]
    assert live["a_pair"]["final_residual"] == ""
    assert live["b_pair"]["final_residual"] == ""
    assert live["committed_character_contradictions"] == 0
    assert row["independent_audit"]["normalized_letters"] == 592
    assert row["independent_audit"]["two_pointer_exact"] is True
    assert row["independent_audit"]["sha_equal"] is True
    assert row["full_text_gate"] == {
        "fragments": False,
        "sentence_boundary_corruption": False,
        "inserted_unit_duplicated": False,
        "worsened_inherited_unit": False,
        "catalogue_shortcut": False,
        "word_order_shortcut": False,
        "status": "passed structural gate; readability remains unpromoted",
    }
    assert len(row["attempts"]) <= 8
    assert row["attempts"][0]["status"] == "accepted"
