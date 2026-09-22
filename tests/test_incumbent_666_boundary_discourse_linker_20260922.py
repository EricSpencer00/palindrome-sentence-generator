import json

from experiments.incumbent_666_boundary_discourse_linker_20260922 import (
    CHILD_SHA256,
    LINKER_LEFT,
    LINKER_RIGHT,
    NEW_LEFT,
    NEW_RIGHT,
    OUT,
    SWITCH_LEFT,
    SWITCH_RIGHT,
    independent_audit,
    normalize,
)


def test_boundary_linker_persists_obstruction_and_exact_changed_seam():
    payload = json.loads(OUT.read_text())
    row = next(row for row in payload["rows"] if row["id"] == "boundary-linker-switch-graft-666")
    result = independent_audit(row["rendered"])

    assert row["parent_sha256"] == "cafd77235f82d9ff4f68814dc7e03d196bf719bf1ec5d541e172073502e12297"
    assert row["promotion_status"]["promoted"] is False
    assert payload["linker_attempt"]["attempt_count"] == 5
    assert payload["linker_attempt"]["closures"] == []
    obstruction = payload["linker_attempt"]["deepest_obstruction"]
    assert obstruction["cursor"] == 13
    assert obstruction["residual"]
    assert obstruction["reason"] == "character_contradiction"
    assert row["live_seam"]["normalized_window_left"] == list(SWITCH_LEFT)
    assert row["live_seam"]["normalized_window_right"] == list(SWITCH_RIGHT)
    assert result["normalized_letters"] == 666
    assert result["two_pointer_exact"]
    assert result["sha256_forward"] == CHILD_SHA256
    assert result["sha_equal"]
    assert row["live_seam"]["final_residual"] == ""
    assert len(row["live_seam"]["left_trace"]) == 70
    assert len(row["live_seam"]["right_trace"]) == 70


def test_linker_is_connective_and_neighbor_aware_and_changed_graft_is_clean():
    payload = json.loads(OUT.read_text())
    row = next(row for row in payload["rows"] if row["id"] == "boundary-linker-switch-graft-666")
    attempt = row["linker_attempt"]

    assert attempt["normalized_windows"] == {
        "left": list(LINKER_LEFT),
        "right": list(LINKER_RIGHT),
    }
    assert attempt["relation_inventory"] == ["because", "so", "after", "when", "then"]
    assert all(set(candidate["neighbor_checks"]) == {
        "left_before",
        "left_after",
        "right_before",
        "right_after",
    } for candidate in attempt["attempts"])
    assert row["live_seam"]["boundary_dedup"]["passed"] is True
    assert row["semantic_roles"] == {
        "varied_relations": ["sees", "stops", "spots"],
        "complete_svo_clauses": True,
        "repeated_subject_verb_frames": [],
        "repeated_neighboring_clauses": False,
        "vocative_or_appositive_fragments": False,
    }
    assert row["readability_delta"]["material_full_text_improvement"] is True
    assert row["readability_delta"]["repeated_saw_noel_live_before"] == 2
    assert row["readability_delta"]["repeated_saw_noel_live_after"] == 0
    assert normalize(NEW_LEFT) == normalize(NEW_RIGHT)[::-1]
