"""Regression checks for the corrected raw-clause recomputed-seam run."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs" / "incumbent-568-clause-lattice-recomputed-seam-20261002.json"


def test_authoritative_lineage_and_all_prior_evidence() -> None:
    payload = json.loads(ARTIFACT.read_text())
    assert payload["parent"]["letters"] == 568
    assert payload["parent"]["sha256"] == "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
    assert [entry["letters"] for entry in payload["preserved_frontier"]] == [568, 560, 558, 556]
    assert {entry["letters"] for entry in payload["rejected_evidence"]} >= {592, 594, 596, 600}
    assert payload["comparison_evidence"]["source_commit"] == "9cb68296"
    assert payload["config"]["raw_clause_global_gate"] is True


def test_recomputed_complete_shell_and_independent_raw_global_counts() -> None:
    row = json.loads(ARTIFACT.read_text())["rows"][0]
    assert row["requested_seam"] == {"normalized_windows": [[170, 197], [469, 496]], "raw_windows": [[227, 263], [644, 680]]}
    assert row["recomputed_seam"] == {
        "normalized_shell_spans": [[163, 204], [364, 405]],
        "raw_shell_spans": [[220, 277], [498, 559]],
        "old_left": "Nora saw Noel live. Noel, I sit. Pat notes. Mara saw God.",
        "old_right": "Dog was Aram. Seton, tap. 'Tis I, Leon. “Evil Leon” was Aron.",
        "reflection_verified": True,
    }
    assert row["independent_audit"]["normalized_letters"] == 592
    assert row["independent_audit"]["two_pointer_exact"] is True
    assert row["independent_audit"]["sha_equal"] is True
    assert row["audit"]["project_validator_exact"] is True
    gate = row["full_render_gate"]
    assert all(gate[key] for key in gate if key != "status")
    assert gate["raw_clause_counter_source"] == "rendered_text"
    assert gate["count_delta_cap"] == 3
    assert gate["global_subject_count_deltas"] == {"aron": 3, "nadia": 2, "nora": 1, "star": 2}
    assert gate["global_object_count_deltas"] == {"aidan": 2, "aron": 1, "nora": 3, "rats": 2}
    assert gate["global_predicate_count_deltas"] == {"sees": 2, "spots": 3, "stops": 3}
    assert all(value == 1 for value in gate["global_frame_count_deltas"].values())
    assert gate["introduced_vs_full_counts"]["deltas"] == {
        "subjects": gate["global_subject_count_deltas"],
        "predicates": gate["global_predicate_count_deltas"],
        "objects": gate["global_object_count_deltas"],
        "frames": gate["global_frame_count_deltas"],
    }
    assert row["online_state"]["final_residual"] == ""
    assert row["online_state"]["attempted_paired_expansions"] == 5
    assert row["online_state"]["accepted_paired_expansions"] == 4
    assert any(attempt["status"] == "rejected" for attempt in row["online_state"]["attempts"])
