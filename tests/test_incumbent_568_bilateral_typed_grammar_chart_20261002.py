"""Regression coverage for the fresh bilateral typed-chart seam."""
from __future__ import annotations

import json
from pathlib import Path

from experiments.incumbent_568_bilateral_typed_grammar_chart_20261002 import (
    AttachmentState,
    ClauseState,
    ObjectState,
    PredicateState,
    ReverseGrammarChart,
    SubjectState,
    TypedGrammarChart,
    consume_against_residual,
    build_payload,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "incumbent-568-bilateral-typed-grammar-chart-20261002.json"


def test_fresh_chart_preserves_lineage_and_preflight() -> None:
    payload = json.loads(ARTIFACT.read_text())
    assert payload["parent"] == {
        "artifact": "runs/incumbent-560-outer-causal-scene-20261002.json",
        "id": "outer-causal-scene-568-working-incumbent",
        "letters": 568,
        "sha256": "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380",
    }
    assert [entry["letters"] for entry in payload["preserved_frontier"]] == [568, 560, 558, 556]
    assert payload["novelty_preflight"]["prior_568_608_artifacts"] == 13
    assert payload["novelty_preflight"]["scanned_artifacts"] == 14
    assert payload["novelty_preflight"]["corresponding_implementations_verified"] is True
    assert payload["novelty_preflight"]["relation_reuse"]["all_new_edges_absent_from_scanned_artifacts"] is False
    assert payload["novelty_preflight"]["selected_geometry"]["normalized_cuts"] == [194, 374]
    assert payload["novelty_preflight"]["excluded_622_648_geometry"]["normalized_windows"] == [[135, 162], [460, 487]]


def test_online_chart_path_and_full_render_gate() -> None:
    row = json.loads(ARTIFACT.read_text())["rows"][0]
    assert row["independent_audit"]["normalized_letters"] == 594
    assert row["independent_audit"]["two_pointer_exact"] is True
    assert row["independent_audit"]["sha_equal"] is True
    assert row["audit"]["project_validator_exact"] is True
    assert row["seam"]["normalized_cuts"] == [194, 374]
    assert row["bilateral_chart"]["independent_side_charts"] is True
    assert row["bilateral_chart"]["right_state_discovered_at_terminal"] is True
    assert row["online_state"]["final_residual"] == ""
    assert row["online_state"]["owner"] == "bilateral_chart"
    assert row["global_shortcut_flags"]["posthoc_equality_trace"] is False
    assert row["global_shortcut_flags"]["preauthored_accepted_pair"] is False
    assert row["global_shortcut_flags"]["uses_622_parent"] is False
    assert row["global_gate"]["human_certified"] is False
    assert row["provenance"]["reader_status"].startswith("pending human")
    attempts = row["online_state"]["attempts"]
    assert any(attempt["status"].startswith("rejected") for attempt in attempts)
    assert attempts[-1]["status"] == "accepted"
    assert attempts[-1]["residual"]["final_residual"] == ""
    assert attempts[-1]["residual"]["left_cursor_after"] > attempts[-1]["residual"]["left_cursor_before"]
    assert all("trace" in attempt["residual"] for attempt in attempts)


def test_historical_scan_corrects_false_594_novelty_claim() -> None:
    payload = json.loads(ARTIFACT.read_text())
    preflight = payload["novelty_preflight"]
    snapshot = preflight["historical_scan"]
    assert snapshot["snapshot_commit"] == "7e358cfd"
    assert snapshot["counted_as_prior_568_608_evidence"] is False
    assert snapshot["artifact_sha256"] == "40e54f901cfbe13cb02b1d54a7a5a13a7078dd5643e3a7500a5e225308fc06c9"
    assert {match["path"] for match in snapshot["matches"]} == {
        "rows[0].bridge_attempt.new_right",
        "rows[0].bridge_attempt.attempt_rendered",
    }
    reuse = preflight["relation_reuse"]
    assert reuse["historical_reused_edges"] == ["nadia sees aron"]
    assert reuse["all_new_edges_absent_from_scanned_artifacts"] is False
    assert reuse["all_new_edges_absent_from_568_608_preflight"] is True
    row = payload["rows"][0]
    assert row["seam"]["fresh_against_preflight"] is False
    assert row["global_gate"]["new_relations_unique"] is False
    assert row["independent_audit"]["normalized_letters"] == 594
    assert row["independent_audit"]["two_pointer_exact"] is True


def test_chart_records_a_concrete_character_obstruction() -> None:
    left = ClauseState(
        SubjectState("Nora"),
        PredicateState("sees"),
        ObjectState("Aidan"),
        # Same typed state, but the opposing chart deliberately lacks this
        # emitted prefix; this exercises the executable contradiction path.
        AttachmentState(),
    )
    opposing = ClauseState(
        SubjectState("Aron"),
        PredicateState("stops"),
        ObjectState("Nora"),
        AttachmentState(),
    )
    state = consume_against_residual(left, ReverseGrammarChart((opposing,)), 194, 373)
    assert state["status"] == "rejected_character_contradiction"
    assert state["committed_character_contradictions"] == 1
    assert state["final_residual"]
    assert state["contradiction"]["left_cursor"] == 194
    assert state["contradiction"]["right_reverse_cursor"] == 373
    assert state["contradiction"]["expected_choices"]


def test_build_payload_remains_executable() -> None:
    payload = build_payload()
    assert payload["rows"][0]["independent_audit"]["normalized_letters"] > 568
    assert payload["rows"][0]["bilateral_chart"]["online_residual"]["final_residual"] == ""
    assert payload["stats"]["committed_character_contradictions"] == 0
