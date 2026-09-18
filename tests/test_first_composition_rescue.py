import pytest

from experiments.first_composition_rescue import (
    MirrorPair,
    assert_counterbalance,
    compose,
    pair_id,
    sample_blocks,
)
from llm_palindrome.validator import is_palindrome


def make_pair(left, right, index):
    return MirrorPair(pair_id(left, right), index, "test", "hash", tuple(left), tuple(right))


def test_two_pair_composition_is_exact_and_uses_both_mirror_pairs():
    outer = make_pair(["step", "on"], ["no", "pets"], 0)
    inner = make_pair(["rats", "live", "on"], ["no", "evil", "star"], 1)
    words, layout = compose(outer, inner)
    assert is_palindrome(" ".join(words))
    assert [row["role"] for row in layout] == [
        "outer_left", "inner_left", "inner_right", "outer_right",
    ]
    assert {row["pair_id"] for row in layout} == {outer.pair_id, inner.pair_id}


def test_sampling_is_seeded_disjoint_and_requires_even_count():
    pairs = [make_pair([f"a{i}"], [f"b{i}"], i) for i in range(12)]
    # The fake words are fine here: sampling never assumes they are English.
    first = sample_blocks(pairs, 9, pair_count=4)
    second = sample_blocks(pairs, 9, pair_count=4)
    assert first == second
    assert len({pair.pair_id for block in first for pair in block}) == 4
    with pytest.raises(ValueError):
        sample_blocks(pairs, 9, pair_count=3)


def test_counterbalance_requires_identical_material_but_different_order():
    base = {
        "block_id": "B01", "depth": 2, "letters": 20, "words": 6,
        "character_multiset": list("aabb"), "word_multiset": ["a", "b"],
        "rendered": "abba", "plain": "a b",
    }
    good = [
        {**base, "condition": "order_a_outer", "plain": "a b"},
        {**base, "condition": "order_b_outer", "plain": "b a"},
    ]
    assert_counterbalance(good)
    bad = [{**good[0], "words": 5}, good[1]]
    with pytest.raises(AssertionError):
        assert_counterbalance(bad)
