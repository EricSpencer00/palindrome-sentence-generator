"""The two searches put a new word at opposite ends of the same half.

A scorer conditions each word on its neighbour in reading order, so it has to
know which end moved. `placement` alone does not say: outside-in appends on the
left and prepends on the right, center-out does the reverse. These tests pin
the `growth` flag against the search that reports it, so a scorer can trust it.
"""
from __future__ import annotations

import pytest

from llm_palindrome.centerout import centerout_search
from llm_palindrome.scoring import FreqScorer, adjacent
from llm_palindrome.search import WordTries, beam_search

VOCAB = ["a", "an", "as", "at", "no", "on", "or", "so", "to",
         "star", "rats", "live", "evil", "step", "pets", "was", "saw",
         "not", "ton", "dog", "god", "part", "trap", "time", "emit"]


class SpyScorer:
    """Wraps a real scorer and checks the growth flag against the word list."""

    def __init__(self):
        self.inner = FreqScorer(VOCAB)
        self.calls = []
        self.violations = []

    def word_delta(self, left, right, placement, word, growth):
        seq = left if placement == "L" else right
        placed = seq[-1] if growth == "append" else seq[0]
        if placed != word:
            self.violations.append((placement, growth, word, seq))
        self.calls.append((placement, growth))
        return self.inner.word_delta(left, right, placement, word, growth)


@pytest.mark.parametrize("search,kwargs,expected", [
    (beam_search, {}, {("L", "append"), ("R", "prepend")}),
    (centerout_search, {"center": "level"}, {("L", "prepend"), ("R", "append")}),
])
def test_growth_matches_the_end_that_moved(search, kwargs, expected):
    spy = SpyScorer()
    search(WordTries(VOCAB), spy, min_letters=20, beam_width=8, seed=0, **kwargs)

    assert spy.calls, "search never scored a word"
    assert not spy.violations, (
        f"growth flag disagreed with the word's position: {spy.violations[:3]}")
    assert set(spy.calls) <= expected, (
        f"unexpected (placement, growth) pairs: {set(spy.calls) - expected}")


def test_adjacent_reads_opposite_ends_for_the_two_growths():
    left = ("rats", "live", "on")
    assert adjacent(left, (), "L", "append") == "live"    # "on" was appended
    assert adjacent(left, (), "L", "prepend") == "live"   # "rats" was prepended


def test_adjacent_is_none_without_a_neighbour():
    assert adjacent(("star",), (), "L", "append") is None
    assert adjacent((), ("star",), "R", "prepend") is None
