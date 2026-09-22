from experiments.working_overhang_growth_20260921 import audit
from experiments.working_overhang_phrase_20260921 import phrase_units


def test_phrase_inventory_contains_only_non_self_mirroring_units():
    units = phrase_units(["at", "dusk", "the", "harbor"], limit=20)
    assert all(" " in unit for unit in units)
    assert all(unit.replace(" ", "") != unit.replace(" ", "")[::-1] for unit in units)


def test_phrase_lane_candidate_is_exact_and_independently_audited():
    text = "one morning is now an aide rips nine memos some men inspire diana won sign in rome no"
    result = audit(text)
    assert result["letters"] == 68
    assert result["two_pointer_exact"] is True
    assert result["sha_equal"] is True
    assert result["validator_exact"] is True
