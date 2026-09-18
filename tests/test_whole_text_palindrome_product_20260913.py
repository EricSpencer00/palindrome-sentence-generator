from itertools import product

import pytest

from experiments.whole_text_palindrome_product_20260913 import (
    compile_slots, construct, exhaustive_reference, replay_path,
)


def test_product_matches_independent_exhaustive_oracle():
    # Synthetic alphabets are mechanism fixtures, never generated English.
    options = (("a", "ab", "ba"), ("b", "aa", "aba"), ("a", "bb"))
    for choices in product(options, repeat=3):
        grammar = compile_slots(choices)
        result = construct(grammar)
        assert result["states_exhausted"]
        assert {tuple(row["words"]) for row in result["records"]} == exhaustive_reference(choices)


def test_center_can_lie_inside_word_and_unequal_sides():
    # Three letters in the first lexical slot, two in the second: the midpoint
    # must be inside slot zero. A question-versus-answer trie would miss this.
    result = construct(compile_slots((("aba",), ("ba",))))
    assert result["records"][0]["tape"] == "ababa"
    assert result["records"][0]["center_characters"] == 1


def test_even_middle_and_shared_boundary():
    result = construct(compile_slots((("ab",), ("ba",))))
    assert result["records"][0]["tape"] == "abba"
    assert result["records"][0]["center_characters"] == 0


def test_budget_does_not_claim_exhaustion():
    result = construct(compile_slots((("ab",), ("ba",))), max_states=1)
    assert result["truncated"]
    assert not result["states_exhausted"]
    assert result["pending_states"] > 0


def test_replay_rejects_disconnected_path():
    grammar = compile_slots((("ab", "ba"), ("ab",)))
    assert not replay_path(grammar, (grammar.edges[-1],))["ok"]


def test_input_rejects_unmodeled_letters():
    with pytest.raises(ValueError):
        compile_slots((("a-b",),))
