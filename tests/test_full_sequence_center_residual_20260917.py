"""Regression tests for the full-sequence grammar product.

These tests pin down two correctness properties that the broad experiment
previously missed: a seam may leave a multi-letter palindromic residual inside
one word, and ordinary overlap with a quarantined catalogue is diagnostic
rather than an automatic rejection.
"""

from experiments.full_sequence_grammar_product_20260917 import (
    anti_shortcut,
    exact_audit,
    search_pattern,
    shortcut_violations,
)


def test_multi_letter_center_residual_is_admitted():
    # Outer letters of ``ab`` match the final two letters of ``acaba``; the
    # remaining center is ``aca``.  The old <=1 rule incorrectly rejected it.
    result = search_pattern(
        ("L", "R"),
        state_budget=100,
        banks={"L": ("ab",), "R": ("acaba",)},
        catalogue_fixture=True,
    )

    assert len(result.paths) == 1
    candidate = result.paths[0]
    assert candidate["rendered"] == "ab acaba"
    assert exact_audit(candidate["rendered"])["exact"]


def test_catalogue_word_overlap_is_not_a_shortcut_violation():
    flags = anti_shortcut(("note", "writes"))

    assert flags["catalogue_word_overlap"] == ["note"]
    assert shortcut_violations(flags) == {}


def test_near_duplicate_catalogue_sequence_is_rejected():
    # Swapping only the fixture's endpoint words changes the tape but still
    # presents borrowed catalogue prose as if it were newly generated.
    words = (
        "cod", "note", "i", "dissent", "a", "fast", "never", "prevents",
        "a", "fatness", "i", "diet", "on", "doc",
    )
    flags = anti_shortcut(words)

    assert flags["catalogue_tape"] is False
    assert flags["catalogue_sequence_derivative"] is True
    assert "catalogue_sequence_derivative" in shortcut_violations(flags)
