import json

from experiments.incumbent_666_contextual_reciprocal_seam_20260922 import (
    CHILD_SHA256,
    NEW_LEFT,
    NEW_RIGHT,
    OUT,
    PARENT_SHA256,
    SEAM_LEFT,
    SEAM_RIGHT,
    independent_audit,
    normalize,
)


def test_contextual_reciprocal_seam_is_exact_and_unpromoted():
    payload = json.loads(OUT.read_text())
    row = next(row for row in payload["rows"] if row["id"] == "contextual-reciprocal-seam-liam-666")
    result = independent_audit(row["rendered"])

    assert row["parent_sha256"] == PARENT_SHA256
    assert row["promotion_status"]["promoted"] is False
    assert row["promotion_status"]["status"] == "rejected_for_naturalness_debt"
    assert result["normalized_letters"] == 666
    assert result["two_pointer_exact"]
    assert result["sha256_forward"] == CHILD_SHA256
    assert result["sha_equal"]
    assert row["live_seam"]["normalized_window_left"] == list(SEAM_LEFT)
    assert row["live_seam"]["normalized_window_right"] == list(SEAM_RIGHT)
    assert row["live_seam"]["final_residual"] == ""
    assert len(row["live_seam"]["left_trace"]) == 16
    assert len(row["live_seam"]["right_trace"]) == 16
    assert normalize(NEW_LEFT) == normalize(NEW_RIGHT)[::-1]


def test_contextual_repair_gates_global_and_neighbor_novelty():
    payload = json.loads(OUT.read_text())
    row = next(row for row in payload["rows"] if row["id"] == "contextual-reciprocal-seam-liam-666")
    novelty = row["operator"]["novelty"]

    assert row["operator"]["cartesian_sweep"] is False
    assert row["operator"]["global_frozen_parent_checked"] is True
    assert novelty["global_clause_novelty"] is True
    assert novelty["global_frame_novelty"] is True
    assert novelty["candidate_frame_duplicates"] == []
    assert all(novelty["neighbor_duplicate_checks"].values())
    assert novelty["complete_english_clauses"] is True
    assert row["semantic_roles"] == {
        "complete_english_clauses": True,
        "varied_relations": ["sees"],
        "repeated_subject_verb_frames": [],
        "repeated_neighboring_clauses": False,
        "left_semantically_ambiguous_or_telegraphic": True,
        "right_generated_filler": True,
    }
    assert row["readability_delta"]["material_full_text_improvement"] is False
