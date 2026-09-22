import json

from experiments.incumbent_666_central_mini_scene_comparison_20260922 import (
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


def test_central_mini_scene_is_exact_and_pending_review():
    payload = json.loads(OUT.read_text())
    row = next(row for row in payload["rows"] if row["id"] == "central-mini-scene-comparison-leon-noel-666")
    result = independent_audit(row["rendered"])

    assert row["parent_sha256"] == PARENT_SHA256
    assert row["promotion_status"]["promoted"] is True
    assert row["promotion_status"]["status"] == "promoted_active_readability_frontier"
    assert row["promotion_status"]["comparison_retained"]["sha256"] == PARENT_SHA256
    assert result["normalized_letters"] == 666
    assert result["two_pointer_exact"]
    assert result["sha256_forward"] == CHILD_SHA256
    assert result["sha_equal"]
    assert row["live_seam"]["normalized_window_left"] == list(SEAM_LEFT)
    assert row["live_seam"]["normalized_window_right"] == list(SEAM_RIGHT)
    assert row["live_seam"]["clause_boundary_cursors"] == [13, 27, 42]
    assert row["live_seam"]["final_residual"] == ""
    assert len(row["live_seam"]["left_trace"]) == 42
    assert len(row["live_seam"]["right_trace"]) == 42
    assert normalize(NEW_LEFT) == normalize(NEW_RIGHT)[::-1]
    assert payload["active_readability_frontier"]["sha256"] == CHILD_SHA256
    assert payload["comparison_retained"]["sha256"] == PARENT_SHA256


def test_central_mini_scene_has_connected_novel_complete_clauses():
    payload = json.loads(OUT.read_text())
    row = next(row for row in payload["rows"] if row["id"] == "central-mini-scene-comparison-leon-noel-666")
    novelty = row["operator"]["novelty"]

    assert row["operator"]["cartesian_sweep"] is False
    assert row["operator"]["global_frozen_parent_checked"] is True
    assert novelty["global_clause_novelty"] is True
    assert novelty["global_frame_novelty"] is True
    assert novelty["candidate_frame_duplicates"] == []
    assert novelty["connected_entity_chain"]["shared_links"] == ["Leon", "Noel", "Nadia", "Aidan"]
    assert novelty["neighbor_context"]["right"]["before"] == "Dog sees Ira"
    assert all(novelty["neighbor_duplicate_checks"].values())
    assert novelty["complete_finite_svo_clauses"] is True
    assert row["semantic_roles"] == {
        "complete_finite_svo_clauses": True,
        "connected_entity_chains": True,
        "varied_relations": ["stops", "spots"],
        "repeated_subject_verb_frames": [],
        "repeated_neighboring_clauses": False,
    }
    assert row["readability_delta"] == {
        "material_full_text_improvement": True,
        "connected_central_event_scene": True,
        "complete_finite_svo_clauses": True,
        "inherited_debt_remains": True,
    }
