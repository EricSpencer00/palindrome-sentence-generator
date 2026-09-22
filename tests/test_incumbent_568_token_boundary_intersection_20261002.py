"""Regression checks for the intact token-boundary 568 seam attempt."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "incumbent-568-token-boundary-intersection-20261002.json"


def test_parent_frontier_and_rejected_split_child_are_preserved() -> None:
    payload = json.loads(ARTIFACT.read_text())
    assert payload["parent"]["letters"] == 568
    assert payload["parent"]["sha256"] == "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
    assert [entry["letters"] for entry in payload["preserved_frontier"]] == [568, 560, 558, 556]
    rejected = payload["rejected_evidence"]
    assert rejected["letters"] == 596
    assert rejected["status"] == "rejected"
    assert "split-word seam" in rejected["reason"]
    assert payload["comparison_evidence"]["source_commit"] == "9cb68296"


def test_intact_token_boundary_residual_closes_without_repair() -> None:
    payload = json.loads(ARTIFACT.read_text())
    row = payload["rows"][0]
    seam = row["live_seam"]
    assert seam["normalized_cut_letters"] == 64
    assert seam["left_cursor_state"]["complete_left_clause"] is True
    assert seam["right_cursor_state"]["complete_left_clause"] is True
    assert seam["left_cursor_state"]["next_token_intact"] is True
    assert seam["right_cursor_state"]["next_token_intact"] is True
    assert seam["left_residual_final"] == seam["right_reverse_residual_final"] == ""
    assert seam["committed_character_contradictions"] == 0
    assert row["shortcut_gate"] == {
        "inserted_unit_duplicated": False,
        "fragments": False,
        "sentence_boundary_corruption": False,
        "catalogue_shortcut": False,
        "word_order_shortcut": False,
    }
    assert row["independent_audit"]["normalized_letters"] == 596
    assert row["independent_audit"]["two_pointer_exact"] is True
    assert row["independent_audit"]["sha_equal"] is True
    assert len(row["attempts"]) <= 8
    assert row["attempts"][0]["status"] == "accepted"
