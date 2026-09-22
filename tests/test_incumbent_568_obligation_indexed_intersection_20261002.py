"""Regression checks for the focused 568 seam run."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "incumbent-568-obligation-indexed-intersection-20261002.json"


def test_loaded_parent_frontier_and_comparison_are_preserved() -> None:
    payload = json.loads(ARTIFACT.read_text())
    assert payload["parent"] == {
        "artifact": "runs/incumbent-560-outer-causal-scene-20261002.json",
        "id": "outer-causal-scene-568-working-incumbent",
        "letters": 568,
        "sha256": "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380",
    }
    assert [entry["letters"] for entry in payload["preserved_frontier"]] == [568, 560, 558, 556]
    assert payload["comparison_evidence"]["letters"] == 666
    assert payload["comparison_evidence"]["sha256"] == "bab693719482af36c7e223a687f94552ad3efda6825d481014134a7d7ae7148d"
    assert payload["comparison_evidence"]["source_commit"] == "9cb68296"


def test_partial_word_obligation_closes_online_in_bounded_run() -> None:
    payload = json.loads(ARTIFACT.read_text())
    row = payload["rows"][0]
    seam = row["live_seam"]
    assert seam["left_partial_join"] == "deli|vers"
    assert seam["right_partial_join"] == "rev|iled"
    assert seam["final_residual"] == ""
    assert seam["committed_character_contradictions"] == 0
    assert row["independent_audit"]["normalized_letters"] == 596
    assert row["independent_audit"]["two_pointer_exact"] is True
    assert row["independent_audit"]["sha_equal"] is True
    assert len(row["attempts"]) <= 8
    assert row["attempts"][-1]["status"] == "accepted"
