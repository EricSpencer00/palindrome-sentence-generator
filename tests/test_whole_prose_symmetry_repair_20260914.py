from experiments.whole_prose_symmetry_repair_20260914 import (
    better_diagnostic,
    independent_scan,
    mismatch_positions,
    parse_text,
    surface_diagnostics,
)


def test_mismatch_positions_are_independent_of_word_boundaries():
    assert mismatch_positions("abca") == [1, 2]
    assert mismatch_positions("abba") == []


def test_independent_scan_checks_the_rendered_surface_directly():
    scan = independent_scan("A man, a plan, a canal: Panama.")
    assert scan["exact_letter_palindrome"]
    assert scan["direct_symmetric_position_comparison"]
    assert scan["letters"] == 21


def test_parser_rejects_non_object_and_accepts_only_text():
    assert parse_text("not json") == (None, "reply_has_no_json_object")
    assert parse_text('{"text":"A complete sentence."}') == ("A complete sentence.", None)


def test_surface_diagnostics_never_calls_nonexact_prose_eligible():
    result = surface_diagnostics("A baker shares warm bread with a child.", "a baker shares bread")
    assert not result["mechanically_eligible"]
    assert result["human_readability"] == "not_certified"


def test_frontier_keeps_a_lower_mismatch_surface_over_a_later_regression():
    good = {"parseable": True, "mismatch_count": 8, "mismatch_rate": 0.08, "letters": 100}
    later = {"parseable": True, "mismatch_count": 12, "mismatch_rate": 0.07, "letters": 130}
    assert not better_diagnostic(later, good)
