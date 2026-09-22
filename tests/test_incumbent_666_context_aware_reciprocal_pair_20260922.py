import json

from experiments.incumbent_666_context_aware_reciprocal_pair_20260922 import (
    NEW_LEFT,
    NEW_RIGHT,
    OUT,
    PARENT_SHA256,
    independent_audit,
    normalize,
)


CHILD_SHA256 = "a1b4ebaaba06893fdfa2a495676b887355e361266b2960a61818981e59e0da37"


def test_context_aware_pair_is_exact_and_pending_review():
    payload = json.loads(OUT.read_text())
    row = next(row for row in payload["rows"] if row["id"] == "context-aware-reciprocal-pair-noel-sara-666")
    result = independent_audit(row["rendered"])

    assert row["parent_sha256"] == PARENT_SHA256
    assert row["promotion_status"]["promoted"] is True
    assert row["promotion_status"]["status"] == "promoted_active_readability_frontier"
    assert result["normalized_letters"] == 666
    assert result["two_pointer_exact"]
    assert result["sha256_forward"] == CHILD_SHA256
    assert result["sha_equal"]
    assert row["audit"]["project_validator_exact"]
    assert normalize(NEW_LEFT) == normalize(NEW_RIGHT)[::-1]
    assert row["readability_delta"]["material_full_text_improvement"] is True
    assert row["promotion_status"]["full_text_rationale"]["material_full_text_improvement"] is True
    assert row["promotion_status"]["comparison_retained"]["sha256"] == PARENT_SHA256
    assert payload["active_readability_frontier"]["sha256"] == CHILD_SHA256
    assert payload["comparison_retained"]["sha256"] == PARENT_SHA256


def test_pair_tracks_context_links_states_and_full_parent_novelty():
    payload = json.loads(OUT.read_text())
    row = next(row for row in payload["rows"] if row["id"] == "context-aware-reciprocal-pair-noel-sara-666")
    seam = row["live_seam"]
    pair = seam["paired_expansions"][0]

    assert seam["normalized_left"] == [127, 140]
    assert seam["normalized_right"] == [526, 539]
    assert seam["raw_left"] == [171, 188]
    assert seam["raw_right"] == [719, 736]
    assert seam["old_left"] == "Mara sees Nadia. "
    assert seam["old_right"] == "Aidan sees Aram. "
    assert seam["new_left"] == "Noel stops Aras. "
    assert seam["new_right"] == "Sara spots Leon. "
    assert seam["paired_cursors_after"] == [13, 13]
    assert seam["residuals"] == {"left": "", "right": ""}
    assert seam["exact_reverse_equation"] is True
    assert pair["paired_cursors_before"] == [0, 0]
    assert pair["paired_cursors_after"] == [13, 13]
    assert pair["left_stream"]["exact"] is True
    assert pair["right_stream"]["exact"] is True
    assert len(pair["left_stream"]["trace"]) == 13
    assert len(pair["right_stream"]["trace"]) == 13
    assert pair["complete_finite_clause"] == {"left": True, "right": True}
    assert pair["global_clause_novelty"] == {"left": True, "right": True}
    assert pair["global_frame_novelty"] == {"left": True, "right": True}
    assert pair["frame_novelty_evidence"]["candidate_frames_extracted_from_rendered_tokens"] == ["noel|stops", "sara|spots"]
    assert pair["frame_novelty_evidence"]["candidate_frames_absent_from_parent"] is True
    assert pair["neighboring_entity_links"] == {
        "left": {"direction": "after", "entity": "Noel", "context_clause": "Nadia saw Noel live."},
        "right": {"direction": "before", "entity": "Leon", "context_clause": "Evil Leon was Aidan."},
    }
    assert pair["state_before"]["left_active_entity"] == "Nadia"
    assert pair["state_before"]["right_active_entity"] == "Aram"
    assert pair["state_after"]["left_active_entity"] == "Aras"
    assert pair["state_after"]["right_active_entity"] == "Leon"
    assert pair["removed_repeated_patterns"] == {
        "right_aidan_sees_aram": True,
        "left_mara_sees_nadia_frame": True,
    }
    assert pair["accepted"] is True
    assert row["switch_after_rejection"]["attempted"] is False
