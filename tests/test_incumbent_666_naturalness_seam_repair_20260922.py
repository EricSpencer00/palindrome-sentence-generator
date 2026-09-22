import json

from experiments.incumbent_666_naturalness_seam_repair_20260922 import (
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


def test_naturalness_seam_is_exact_and_unpromoted():
    payload = json.loads(OUT.read_text())
    row = next(row for row in payload["rows"] if row["id"] == "naturalness-seam-repair-liam-666")
    result = independent_audit(row["rendered"])

    assert row["parent_sha256"] == PARENT_SHA256
    assert row["promotion_status"]["promoted"] is False
    assert result["normalized_letters"] == 666
    assert result["two_pointer_exact"]
    assert result["sha256_forward"] == CHILD_SHA256
    assert result["sha_equal"]
    assert row["live_seam"]["normalized_window_left"] == list(SEAM_LEFT)
    assert row["live_seam"]["normalized_window_right"] == list(SEAM_RIGHT)
    assert row["live_seam"]["final_residual"] == ""
    assert len(row["live_seam"]["left_trace"]) == 13
    assert len(row["live_seam"]["right_trace"]) == 13
    assert normalize(NEW_LEFT) == normalize(NEW_RIGHT)[::-1]


def test_naturalness_gates_reject_ambiguity_filler_and_repetition_before_admission():
    payload = json.loads(OUT.read_text())
    row = next(row for row in payload["rows"] if row["id"] == "naturalness-seam-repair-liam-666")
    naturalness = row["operator"]["naturalness"]

    assert row["operator"]["cartesian_sweep"] is False
    assert row["operator"]["global_frozen_parent_checked"] is True
    assert naturalness["global_clause_novelty"] is True
    assert naturalness["global_frame_novelty"] is True
    assert naturalness["candidate_frame_duplicates"] == []
    assert all(naturalness["neighbor_duplicate_checks"].values())
    assert naturalness["complete_natural_clauses"] is True
    assert naturalness["no_compound_name_hack"] is True
    assert naturalness["no_ambiguous_adverb_or_argument_structure"] is True
    assert naturalness["no_generic_filler"] is True
    assert row["semantic_roles"] == {
        "complete_natural_clauses": True,
        "varied_relations": ["stops", "spots"],
        "repeated_subject_verb_frames": [],
        "repeated_neighboring_clauses": False,
    }
