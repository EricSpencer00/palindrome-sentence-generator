import json

from experiments.incumbent_666_center_obstruction_widened_bridge_20260922 import (
    CENTER_LEFT,
    CENTER_RAW,
    NEW_LEFT,
    NEW_RIGHT,
    OUT,
    PARENT_SHA256,
    WIDE_LEFT,
    WIDE_RIGHT,
    independent_audit,
    normalize,
)


def test_center_obstruction_and_parent_preservation_are_recorded():
    payload = json.loads(OUT.read_text())
    row = next(row for row in payload["rows"] if row["id"] == "widened-bridge-no-closure-center-obstruction-666")
    center = row["center_obstruction"]

    assert row["parent_sha256"] == PARENT_SHA256
    assert row["promotion_status"]["promoted"] is False
    parent_result = independent_audit(row["rendered"])
    assert parent_result["normalized_letters"] == 666
    assert parent_result["two_pointer_exact"]
    assert parent_result["sha256_forward"] == PARENT_SHA256
    assert center["normalized_window"] == list(CENTER_LEFT)
    assert center["raw_window"] == list(CENTER_RAW)
    assert center["length_pattern"] == {
        "clause_lengths": [10, 10],
        "token_letter_lengths": [[3, 4, 3], [3, 4, 3]],
        "split_cursor": 10,
        "available_total": 20,
    }
    assert center["rejected_disconnected_pair"] == {
        "left": "Ari sees God.",
        "right": "God sees Ari.",
        "exact_reverse": False,
        "reason": "Although it is a 10-letter disconnected God/Ari alternative, it contradicts the required reciprocal tape (the exact reverse is Dog sees Ira) and does not continue the neighboring Aidan/Nadia discourse chain.",
    }


def test_widened_bridge_persists_single_bounded_obstruction():
    payload = json.loads(OUT.read_text())
    row = next(row for row in payload["rows"] if row["id"] == "widened-bridge-no-closure-center-obstruction-666")
    attempt = row["bridge_attempt"]

    assert attempt["normalized_windows"] == {"left": list(WIDE_LEFT), "right": list(WIDE_RIGHT)}
    assert attempt["clause_boundary_cursors"] == {"left": [13, 27, 52], "right": [13, 27, 39, 52]}
    assert normalize(NEW_LEFT).startswith("aramspotsliam")
    assert len(normalize(NEW_LEFT)) == len(normalize(NEW_RIGHT)) == 52
    assert attempt["gates"] == {
        "connected_entity_chain": True,
        "causal_temporal_anaphoric_link": True,
        "global_clause_novelty": True,
        "global_frame_novelty": False,
        "complete_english_clauses": True,
        "catalogue_or_self_contained_unit": False,
        "neighboring_discourse_continuity": True,
    }
    assert attempt["left_trace"]["exact"] is False
    assert attempt["right_trace"]["exact"] is False
    assert attempt["left_trace"]["cursor"] == 0
    assert attempt["right_trace"]["cursor"] == 0
    assert attempt["final_residual"]["left"]
    assert attempt["final_residual"]["right"]
    assert attempt["exact_child_saved"] is False
    assert payload["next_operator"] == "change to untouched actual seam normalized [197,281)/[385,469), raw [263,384)/[520,644); preserve 568 incumbent"
