"""The pair generator: does it produce mirror-pairs, and are they ours?"""
import time

import pytest

from llm_palindrome.pairs import (acceptable_pair, hunt, junction,
                                  pair_vocabulary, split_at_mirror)
from llm_palindrome.validator import is_palindrome


class TestSplit:
    def test_splits_when_the_mirror_lands_on_a_boundary(self):
        assert split_at_mirror(["step", "on", "no", "pets"]) == (
            ["step", "on"], ["no", "pets"])

    def test_none_when_the_mirror_runs_through_a_word(self):
        """"racecar" mirrors inside itself: a centre, not a pair."""
        assert split_at_mirror(["racecar"]) is None

    def test_none_when_no_prefix_reaches_exactly_half(self):
        assert split_at_mirror(["a", "man", "a", "plan", "a", "canal",
                                "panama"]) is None

    def test_the_halves_of_a_split_spell_each_other_backwards(self):
        left, right = split_at_mirror(["step", "on", "no", "pets"])
        letters = lambda ws: "".join(ws)
        assert letters(left) == letters(right)[::-1]


class TestAcceptable:
    def test_rejects_a_word_shared_across_the_mirror(self):
        assert not acceptable_pair(["no", "it", "cab"], ["bat", "it", "on"])

    def test_rejects_halves_below_the_word_floor(self):
        assert not acceptable_pair(["step", "on"], ["no", "pets"], min_words=3)

    def test_accepts_distinct_halves(self):
        assert acceptable_pair(["draw", "no", "tip"], ["pit", "on", "ward"])

    def test_junction_is_the_pair_of_words_around_the_mirror(self):
        assert junction(["draw", "no", "tip"], ["pit", "on", "ward"]) == (
            "tip", "pit")


class TestVocabulary:
    def test_drops_acronyms_a_frequency_list_calls_words(self):
        """"utc" and "csi" are common, and neither is a word."""
        lexicon = {"step", "pets", "on", "no"}
        vocab = pair_vocabulary(["step", "pets", "utc", "csi"],
                                lambda w: 5.0, lexicon)
        assert set(vocab) == {"step", "pets"}

    def test_keeps_only_the_two_single_letters_that_are_words(self):
        lexicon = {"a", "i", "b"}
        vocab = pair_vocabulary(["a", "i", "b"], lambda w: 5.0, lexicon)
        assert set(vocab) == {"a", "i"}


class TestHunt:
    """A short walk over a tiny vocabulary — enough to prove the contract."""

    @pytest.fixture(scope="class")
    def found(self):
        from llm_palindrome.search import WordTries
        words = ["step", "on", "no", "pets", "draw", "ward", "dad", "a",
                 "star", "rats", "live", "evil", "not", "ton", "was", "saw"]
        tries = WordTries(words)
        return list(hunt(tries, shards=12, node_budget=8000, min_letters=12,
                         max_letters=22, min_words=3, per_family=2,
                         deadline=time.time() + 20))

    def test_it_finds_some(self, found):
        assert found

    def test_every_pair_is_a_palindrome_when_joined(self, found):
        for left, right in found:
            assert is_palindrome(" ".join(left + right))

    def test_no_pair_half_is_a_palindrome_on_its_own(self, found):
        """Criterion 3, enforced at the unit: a self-palindromic half would
        make the mirror do no work for that sentence."""
        for left, right in found:
            assert not is_palindrome(" ".join(left)) or len(left) == 1

    def test_families_are_capped(self, found):
        from collections import Counter
        seen = Counter(junction(l, r) for l, r in found)
        assert max(seen.values()) <= 2

    def test_no_pair_repeats(self, found):
        keys = [(" ".join(l), " ".join(r)) for l, r in found]
        assert len(set(keys)) == len(keys)
