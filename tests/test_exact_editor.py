import pytest

from llm_palindrome.exact_editor import (
    ExactEditorState,
    HalfSpan,
    lexical_surface_evidence,
    materialized_surface_audits,
    new_state,
    paired_span_edit,
    replay_edits,
    surface_audit,
)


def test_exact_tape_exists_before_any_surface_is_rendered():
    state = new_state(half_text="an inward", center_text="", intent="cross-boundary witness")
    assert state.full_tape == "aninwarddrawnina"
    assert state.full_tape == state.full_tape[::-1]
    assert state.letters == 16


def test_paired_edit_preserves_exactness_and_does_not_accept_supplied_counterpart():
    state = new_state(half_text="an inward", intent="witness")
    child, event = paired_span_edit(
        state,
        HalfSpan(0, 2),
        "a",
        source={"kind": "fixture", "id": "paired-edit"},
        notes="replace the opening span",
    )
    assert child is not None and event["exact_by_construction"]
    assert child.full_tape == child.full_tape[::-1]
    assert "right_text" not in event["input"]
    assert child.parent_id == state.state_id


def test_surface_must_match_exact_tape_and_boundaries_are_independent():
    state = new_state(half_text="a dog", intent="catalogued fixture only")
    exact = surface_audit(state, "A dog, god a.")
    assert exact["independent_exactness"]["matches_state_tape"]
    assert exact["independent_exactness"]["direct_symmetric_position_comparison"]
    wrong = surface_audit(state, "A dog, a cat.")
    assert not wrong["mechanically_eligible"]
    assert not wrong["independent_exactness"]["matches_state_tape"]


def test_surface_audit_never_calls_exactness_readability():
    state = new_state(half_text="an inward", intent="not a complete sentence")
    audit = surface_audit(state, "An inward drawn in a.")
    assert audit["independent_exactness"]["matches_state_tape"]
    assert audit["human_reader_study"] == "not_run"


def test_lexical_lattice_records_cross_boundary_segmentations_without_promoting_one():
    state = new_state(half_text="an inward", intent="cross-boundary witness")
    evidence = lexical_surface_evidence(state)
    assert evidence["letters"] == 16
    assert evidence["complete_segmentation_count"] >= 1
    assert evidence["surface_rendering_is_not_certification"]


def test_materialized_surface_audits_never_promotes_a_wrong_tape():
    state = new_state(half_text="an inward", intent="cross-boundary witness")
    audits = materialized_surface_audits(state, max_segmentations=32)
    assert audits
    assert all(row["mechanically_eligible"] for row in audits)
    assert all(row["independent_exactness"]["matches_state_tape"] for row in audits)


def test_replay_preserves_rejections_and_exact_child_chain():
    state = new_state(half_text="an inward", intent="replay")
    final, events = replay_edits(
        state,
        [
            (HalfSpan(0, 2), "a", {"kind": "fixture", "id": "ok"}, "valid"),
            (HalfSpan(99, 100), "x", {"kind": "fixture", "id": "bad"}, "out of bounds"),
        ],
    )
    assert len(events) == 2
    assert events[0]["accepted"]
    assert not events[1]["accepted"]
    assert final.full_tape == final.full_tape[::-1]


def test_center_must_be_exactly_palindromic():
    with pytest.raises(ValueError, match="center_must_be_palindromic"):
        new_state(half_text="a", center_text="ab")
