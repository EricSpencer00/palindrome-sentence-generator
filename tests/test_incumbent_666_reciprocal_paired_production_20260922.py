import json

from experiments.incumbent_666_reciprocal_paired_production_20260922 import (
    NEW_LEFT,
    NEW_RIGHT,
    OUT,
    PARENT_SHA256,
    independent_audit,
    normalize,
)


CHILD_SHA256 = "90806f96a64a289969527ecfff84ed97a91e909d4af7785fd7c4198b542a3f2e"


def test_paired_production_is_exact_and_pending_full_text_review():
    payload = json.loads(OUT.read_text())
    row = next(row for row in payload["rows"] if row["id"] == "reciprocal-paired-production-mara-nadia-666")
    result = independent_audit(row["rendered"])

    assert row["parent_sha256"] == PARENT_SHA256
    assert row["promotion_status"] == {
        "promoted": False,
        "status": "pending_full_text_readability_review",
        "reason": "The paired production is independently exact and globally novel on both sides, but full-text semantic and repetition debt remains; it stays unpromoted pending readability review.",
    }
    assert result["normalized_letters"] == 666
    assert result["two_pointer_exact"]
    assert result["sha256_forward"] == CHILD_SHA256
    assert result["sha_equal"]
    assert row["audit"]["project_validator_exact"]
    assert normalize(NEW_LEFT) == normalize(NEW_RIGHT)[::-1]
    assert row["growth_over_parent"] == 0


def test_paired_seam_updates_both_sides_atomically_with_empty_residuals():
    payload = json.loads(OUT.read_text())
    row = next(row for row in payload["rows"] if row["id"] == "reciprocal-paired-production-mara-nadia-666")
    seam = row["live_seam"]
    pair = seam["paired_expansions"][0]

    assert seam["normalized_left"] == [140, 156]
    assert seam["normalized_right"] == [510, 526]
    assert seam["raw_left"] == [188, 209]
    assert seam["raw_right"] == [698, 719]
    assert seam["old_left"] == "Nadia saw Noel live. "
    assert seam["old_right"] == "Evil Leon was Aidan. "
    assert seam["new_left"] == "Evil Mara was Nadia. "
    assert seam["new_right"] == "Aidan saw Aram live. "
    assert seam["paired_expansion_count"] == 1
    assert seam["max_paired_expansions"] == 8
    assert seam["paired_cursors_after"] == [16, 16]
    assert seam["residuals"] == {"left": "", "right": ""}
    assert seam["exact_reverse_equation"] is True
    assert pair["owners"] == ["left", "right"]
    assert pair["paired_cursors_before"] == [0, 0]
    assert pair["paired_cursors_after"] == [16, 16]
    assert len(pair["left_stream"]["trace"]) == 16
    assert len(pair["right_stream"]["trace"]) == 16
    assert pair["left_stream"]["exact"] is True
    assert pair["right_stream"]["exact"] is True
    assert pair["complete_finite_clause"] == {"left": True, "right": True}
    assert pair["global_clause_novelty"] == {"left": True, "right": True}
    assert pair["global_frame_novelty"] == {"left": True, "right": True}
    assert pair["frame_novelty_evidence"]["candidate_frames_extracted_from_rendered_tokens"] == ["aidan|saw", "evil mara|was"]
    assert pair["frame_novelty_evidence"]["candidate_frames_absent_from_parent"] is True
    assert pair["state_before"]["left_active_discourse_entity"] == "Nadia"
    assert pair["state_before"]["right_active_discourse_entity"] == "Aram"
    assert pair["state_before"]["right_subject_object_stack"]["objects"] == ["Aram"]
    assert pair["fragment_or_catalogue_rejection"] is False
    assert pair["accepted"] is True
    assert pair["atomic_state_update"]["atomic_entity_update"] == {
        "left": {"from": "Mara", "to": "Nadia"},
        "right": {"from": "Aidan", "to": "Aram"},
    }
    assert row["switch_after_rejection"]["attempted"] is False
    assert row["full_text_review"]["material_full_text_improvement"] is False
    assert row["full_text_review"]["grammar_debt"] is True
    assert row["full_text_review"]["repetition_debt_present"] is True
    assert row["full_text_review"]["semantic_debt"]
    assert row["full_text_review"]["repetition_debt_details"]
