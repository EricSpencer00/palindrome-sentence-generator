import pytest

from experiments.joint_phrase_infill import (
    Seed,
    apply_replacements,
    intersect,
    move_for,
    operator_controls,
    word_offsets,
)


def test_offsets_and_move_are_mirrored_at_character_level():
    seed = Seed("x", 0, ("step", "on", "no", "pets"), "steponnopets", "warning")
    assert word_offsets(seed.words) == [(0, 4), (4, 6), (6, 8), (8, 12)]
    move = move_for(seed, 1)
    assert seed.normalized[move["left_start"]:move["left_end"]] == "step"
    assert seed.normalized[move["right_start"]:move["right_end"]] == "pets"
    whole = move_for(seed, "whole_left")
    assert whole["left_start"] == 0
    assert whole["left_end"] == 6


def test_intersection_requires_exact_reverse_not_near_match():
    assert intersect(["step on", "other"], ["no pets", "wrong"]) == [("step on", "no pets")]
    assert intersect(["step on"], ["no pet"]) == []


def test_bilateral_replacement_preserves_exactness_and_rejects_mismatch():
    seed = Seed("x", 0, ("step", "on", "no", "pets"), "steponnopets", "warning")
    move = {"left_start": 0, "left_end": 6, "right_start": 6, "right_end": 12}
    assert apply_replacements(seed, move, "step on", "no pets") == "steponnopets"
    with pytest.raises(ValueError):
        apply_replacements(seed, move, "step on", "no pet")


def test_operator_controls_validate_both_cases():
    controls = operator_controls()
    assert controls["positive"]["right"] == "no pets"
    assert controls["near_match_rejected"]["right"] == "no pet"
