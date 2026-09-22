"""Regression checks for the varied partial-word seam expansion."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "incumbent-568-partial-varied-intersection-20261002.json"


def test_authoritative_parent_frontier_and_rejections_are_preserved() -> None:
    payload = json.loads(ARTIFACT.read_text())
    assert payload["parent"] == {
        "artifact": "runs/incumbent-560-outer-causal-scene-20261002.json",
        "id": "outer-causal-scene-568-working-incumbent",
        "letters": 568,
        "sha256": "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380",
    }
    assert [entry["letters"] for entry in payload["preserved_frontier"]] == [568, 560, 558, 556]
    assert {entry["letters"] for entry in payload["rejected_evidence"]} == {592, 596}
    assert any(entry.get("source_commit") == "4388c9cd" for entry in payload["rejected_evidence"])
    assert payload["comparison_evidence"]["source_commit"] == "9cb68296"


def test_partial_seam_online_intersection_and_full_text_gate() -> None:
    row = json.loads(ARTIFACT.read_text())["rows"][0]
    assert row["seam"] == {
        "normalized_cuts": [196, 372],
        "raw_after_letter_boundaries": [266, 508],
        "partial_words": ["ma|ra", "ar|am"],
        "replacement_spans": [[264, 277], [498, 511]],
        "old_left": "Mara saw God.",
        "old_right": "Dog was Aram.",
    }
    assert row["independent_audit"]["normalized_letters"] == 600
    assert row["independent_audit"]["two_pointer_exact"] is True
    assert row["independent_audit"]["sha_equal"] is True
    assert row["audit"]["project_validator_exact"] is True
    assert row["grammar"]["role_families"] == ["control", "observation"]
    assert row["online_state"]["final_residual"] == ""
    assert row["online_state"]["accepted_paired_expansions"] == 2
    assert row["online_state"]["attempted_paired_expansions"] <= 8
    assert all(attempt["residual"]["final_residual"] == "" for attempt in row["online_state"]["attempts"])
    assert all(attempt["novelty_gate"]["full_clause_novel"] for attempt in row["online_state"]["attempts"])
    accepted = [attempt for attempt in row["online_state"]["attempts"] if attempt["status"] == "accepted"]
    assert {attempt["left_grammar_state"]["role_family"] for attempt in accepted} == {"observation", "control"}
    assert all(row["sentence_gate"].values())
