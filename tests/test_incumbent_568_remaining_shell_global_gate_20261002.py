"""Regression checks for the globally gated remaining-shell construction."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "incumbent-568-remaining-shell-global-gate-20261002.json"


def test_authoritative_parent_and_prior_evidence_are_preserved() -> None:
    payload = json.loads(ARTIFACT.read_text())
    assert payload["parent"] == {
        "artifact": "runs/incumbent-560-outer-causal-scene-20261002.json",
        "id": "outer-causal-scene-568-working-incumbent",
        "letters": 568,
        "sha256": "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380",
    }
    assert [entry["letters"] for entry in payload["preserved_frontier"]] == [568, 560, 558, 556]
    assert any(entry.get("source_commit") == "e3df2e6f" and entry["letters"] == 594 for entry in payload["rejected_evidence"])
    assert {entry["letters"] for entry in payload["rejected_evidence"]} >= {592, 594, 596, 600}
    assert payload["comparison_evidence"]["source_commit"] == "9cb68296"


def test_recomputed_shell_reverse_lattice_and_global_gate() -> None:
    row = json.loads(ARTIFACT.read_text())["rows"][0]
    assert row["requested_seam"] == {"normalized_windows": [[121, 148], [446, 473]], "raw_windows": [[156, 193], [614, 652]]}
    assert row["recomputed_seam"] == {
        "normalized_shell_spans": [[108, 135], [433, 460]],
        "raw_shell_spans": [[142, 179], [600, 638]],
        "old_left": "Mara stops rats. A tub? He maps Nora.",
        "old_right": "Aron, spam. Eh, but a star spots Aram.",
        "reflection_verified": True,
    }
    assert row["independent_audit"]["normalized_letters"] == 594
    assert row["independent_audit"]["two_pointer_exact"] is True
    assert row["independent_audit"]["sha_equal"] is True
    assert row["audit"]["project_validator_exact"] is True
    gate = row["full_render_gate"]
    assert all(gate[key] for key in gate if key != "status")
    assert gate["global_frame_count_deltas"]
    assert gate["raw_clause_counter_source"] == "rendered_text"
    assert gate["global_subject_count_deltas"] == {"aron": 2, "nadia": 2, "star": 2}
    assert gate["global_object_count_deltas"] == {"aidan": 2, "nora": 1, "rats": 1}
    assert gate["global_predicate_count_deltas"] == {"sees": 2, "spots": 2, "stops": 1}
    assert gate["introduced_vs_full_counts"]["deltas"]["subjects"] == gate["global_subject_count_deltas"]
    assert gate["introduced_vs_full_counts"]["deltas"]["objects"] == gate["global_object_count_deltas"]
    assert row["online_state"]["final_residual"] == ""
    assert row["online_state"]["attempted_paired_expansions"] == 4
    assert row["online_state"]["accepted_paired_expansions"] == 3
    assert any(attempt["status"] == "rejected" for attempt in row["online_state"]["attempts"])
    assert row["construction"]["whole_sentence_search"] is False
    assert row["construction"]["self_palindrome_seeded"] is False
