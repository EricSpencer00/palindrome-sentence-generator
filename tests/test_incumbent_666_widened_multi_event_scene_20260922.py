import json

from experiments.incumbent_666_widened_multi_event_scene_20260922 import (
    NEW_LEFT,
    NEW_RIGHT,
    OUT,
    PARENT_SHA256,
    WIDE_LEFT,
    WIDE_RAW_LEFT,
    WIDE_RAW_RIGHT,
    WIDE_RIGHT,
    SWITCH_LEFT,
    SWITCH_RAW_LEFT,
    SWITCH_RAW_RIGHT,
    SWITCH_RIGHT,
    clause_boundaries,
    independent_audit,
    normalize,
)


def test_authored_bridge_is_bounded_and_parent_remains_independently_exact():
    payload = json.loads(OUT.read_text())
    row = next(row for row in payload["rows"] if row["id"] == "widened-multi-event-scene-no-closure-666")

    assert row["parent_sha256"] == PARENT_SHA256
    assert row["promotion_status"]["promoted"] is False
    parent_result = independent_audit(row["rendered"])
    assert parent_result["normalized_letters"] == 666
    assert parent_result["two_pointer_exact"]
    assert parent_result["sha256_forward"] == PARENT_SHA256
    assert row["growth_over_parent"] == 0

    attempt = row["bridge_attempt"]
    assert attempt["normalized_windows"] == {"left": list(WIDE_LEFT), "right": list(WIDE_RIGHT)}
    assert attempt["raw_windows"] == {"left": list(WIDE_RAW_LEFT), "right": list(WIDE_RAW_RIGHT)}
    assert len(normalize(NEW_LEFT)) == len(normalize(NEW_RIGHT)) == 84
    assert attempt["clause_boundary_cursors"] == {
        "left": [13, 27, 39, 49, 61, 73, 84],
        "right": [14, 26, 38, 51, 63, 74, 84],
    }
    assert attempt["gates"] == {
        "global_clause_novelty": True,
        "global_frame_novelty": True,
        "complete_english_clauses": True,
        "connected_entity_chain": True,
        "causal_temporal_anaphoric_link": True,
        "catalogue_or_self_contained_palindrome": False,
        "neighboring_discourse_continuity": True,
    }
    assert attempt["left_trace"]["exact"] is False
    assert attempt["right_trace"]["exact"] is False
    assert attempt["left_trace"]["cursor"] == 0
    assert attempt["right_trace"]["cursor"] == 0
    assert attempt["left_trace"]["residual"].startswith("mailsawira")
    assert attempt["right_trace"]["residual"].startswith("arispotsira")
    assert attempt["exact_child_saved"] is False


def test_contradiction_switch_is_recorded_once_on_the_next_actual_seam():
    payload = json.loads(OUT.read_text())
    row = next(row for row in payload["rows"] if row["id"] == "widened-multi-event-scene-no-closure-666")
    switched = row["switched_attempt"]

    assert switched["normalized_windows"] == {"left": list(SWITCH_LEFT), "right": list(SWITCH_RIGHT)}
    assert switched["raw_windows"] == {"left": list(SWITCH_RAW_LEFT), "right": list(SWITCH_RAW_RIGHT)}
    assert switched["reason"] == "immediate switch after first reciprocal contradiction"
    assert switched["context_letter_lengths"] == {"left": 84, "right": 84}
    assert switched["authored_clause_boundary_residual_cursors"] == {
        "left": [97, 111, 123, 133, 145, 157, 168],
        "right": [14, 26, 38, 51, 63, 74, 84],
    }
    assert switched["left_trace"]["exact"] is False
    assert switched["right_trace"]["exact"] is False
    assert switched["left_trace"]["cursor"] == 84
    assert switched["right_trace"]["cursor"] == 0
    assert switched["left_trace"]["residual"].startswith("mailsawira")
    assert switched["right_trace"]["residual"].startswith("arispotsira")
    assert switched["exact_child_saved"] is False
    assert "preserve 568 incumbent" in payload["next_operator"]
