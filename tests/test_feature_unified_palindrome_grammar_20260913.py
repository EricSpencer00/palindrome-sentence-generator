"""Tests for the connected feature-grammar palindrome constructor."""
from __future__ import annotations

import re
from itertools import product

from experiments.feature_unified_palindrome_grammar_20260913 import (
    GRAMMARS, _canonical_choice, _consume, _next_side, _surface, independent_exact,
    independent_parse, run, solve,
)


def control(grammar):
    return _surface(_canonical_choice(grammar, leaf, {}).word for leaf in grammar.leaves)


def test_controls_are_long_connected_english_grammar_derivations():
    for grammar in GRAMMARS:
        text = control(grammar)
        assert len(re.sub("[^a-z]", "", text.casefold())) >= 100
        assert independent_parse(grammar, text)["intact"]


def test_independent_parser_rejects_agreement_valency_and_missing_object():
    relative = next(grammar for grammar in GRAMMARS if grammar.identifier == "relative-declarative")
    control_text = control(relative)
    assert independent_parse(relative, control_text)["intact"]
    assert not independent_parse(relative, control_text.replace(" monitors ", " operates "))["intact"]
    assert not independent_parse(relative, control_text.replace(" catalogs the detailed manuscript", " catalogs"))["intact"]
    assert not independent_parse(relative, control_text.replace(" catalog", " recorded", 1))["intact"]


def test_partial_debt_uses_letters_not_mirrored_word_boundaries():
    # Synthetic letters are an algorithm control, never a generated candidate.
    debt, owner = _consume("", 0, "ab", 1)
    debt, owner = _consume(debt, owner, "a", -1)
    # Right-side input is reversed before consumption: rendered ``ccb`` emits
    # ``bcc`` toward the outstanding left-side letters.
    debt, owner = _consume(debt, owner, "bcc", -1)
    # The residual closes at a centre inside the non-palindromic token ``ccb``.
    assert debt == "cc" and debt == debt[::-1]
    audit = independent_exact("Ab ccb a")
    assert audit["exact"] and audit["normalized"] == "abccba"


def test_residual_owner_requires_the_opposite_edge_next():
    # This protects the core construction invariant.  If a left leaf leaves
    # unmatched letters, expanding another left leaf would reduce the method
    # to post-hoc whole-string palindrome filtering.
    assert _next_side("abc", 1) == -1
    assert _next_side("abc", -1) == 1
    assert _next_side("", 1) == 1


def test_saved_trace_is_not_needed_for_surface_parse_or_exactness():
    grammar = GRAMMARS[0]
    text = control(grammar)
    assert independent_parse(grammar, text)["intact"]
    assert not independent_exact(text)["exact"]
    # A made-up/corrupted trace is neither accepted nor consulted by either audit.
    corrupted_trace = [{"leaf": "matrix_verb", "word": "preserves", "side": "left"}]
    assert corrupted_trace[0]["word"] not in text.casefold()
    assert independent_parse(grammar, text)["intact"]


def test_tiny_exhaustive_oracle_agrees_with_incremental_letter_debt():
    # Synthetic vocabulary is confined to this oracle test.  It proves the
    # outside-in residual machine neither loses nor invents a closure when word
    # boundaries differ across the centre.
    slots = (("ab", "ac"), ("c", "cc"), ("ba", "ca"))
    brute = {" ".join(words) for words in product(*slots)
             if independent_exact(" ".join(words))["exact"]}
    found = set()

    def visit(lo, hi, left, right, debt, owner):
        if lo > hi:
            if debt == debt[::-1]:
                found.add(" ".join(left + right))
            return
        # Debt records letters supplied by ``owner``.  Cancellation must
        # continue at the other edge, even where it crosses a word boundary.
        side = _next_side(debt, owner)
        index = lo if side == 1 else hi
        for word in slots[index]:
            incoming = word if side == 1 else word[::-1]
            result = _consume(debt, owner, incoming, side)
            if result is None:
                continue
            new_debt, new_side = result
            if side == 1:
                visit(index + 1, hi, left + (word,), right, new_debt, new_side)
            else:
                visit(lo, index - 1, left, (word,) + right, new_debt, new_side)

    visit(0, len(slots) - 1, (), (), "", 0)
    assert found == brute


def test_bounded_search_records_the_single_derivation_invariant_and_separates_admission():
    result = run(state_cap=2_000)
    assert result["config"]["single_connected_derivation"]
    assert result["config"]["exactness_enforced_after_each_lexical_leaf"]
    assert result["readable_survivors"] == []
    for grammar, report in zip(GRAMMARS, result["runs"]):
        assert report["source_control"]["independent_sentence_witness"]["intact"]
        assert not report["source_control"]["mechanically_admitted"]
        for row in report["exact_closures"]:
            assert row["independent_exact_audit"]["exact"]
            assert row["independent_sentence_witness"]["intact"]
            assert row["cancellation_trace"]
        # A hard cap reports incompleteness; it never becomes false exhaustion.
        capped = solve(grammar, state_cap=1)
        assert not capped["exhausted"]
