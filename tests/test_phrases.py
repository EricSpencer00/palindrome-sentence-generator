"""Tests for multi-word units in the search.

The trie has always held single words, so a unit's spelling and its LETTERS
were the same string — `w[::-1]` reversed the letters, `len(w)` counted them,
and `consume(w, overhang)` matched them. A phrase breaks all three at once:
"new york" has eight letters, not nine, and its mirror is "kroywen".

So the tests here are mostly about that separation. The one that matters most
is the last: whatever the trie holds, what comes out has to be a palindrome.
"""
import pytest

from llm_palindrome.centerout import centerout_search
from llm_palindrome.search import WordTries, unit_letters
from llm_palindrome.validator import is_palindrome, normalize


class TestUnitLetters:
    def test_single_word_is_its_own_letters(self):
        assert unit_letters("stop") == "stop"

    def test_phrase_drops_the_spaces(self):
        assert unit_letters("new york") == "newyork"

    def test_letters_are_what_the_palindrome_sees(self):
        phrase = "step on no pets"
        assert unit_letters(phrase) == normalize(phrase)


class TestWordTriesWithPhrases:
    def _tries(self):
        return WordTries(["new", "york", "new york", "stop", "pots"])

    def test_phrase_is_found_by_its_letters_not_its_spelling(self):
        t = self._tries()
        assert "new york" in t.left_candidates("newyork", limit=50)

    def test_phrase_is_not_found_under_its_spaced_spelling(self):
        t = self._tries()
        assert "new york" not in t.left_candidates("new york", limit=50)

    def test_reversed_trie_keys_a_phrase_on_its_reversed_letters(self):
        t = self._tries()
        assert "new york" in t.right_candidates("kroywen", limit=50)

    def test_single_words_still_resolve_alongside_phrases(self):
        t = self._tries()
        assert "stop" in t.left_candidates("stop", limit=50)

    def test_a_phrase_and_its_parts_can_both_be_offered(self):
        t = self._tries()
        found = t.left_candidates("newyork", limit=50)
        assert "new" in found and "new york" in found


class TestSearchWithPhraseUnits:
    """The invariant the whole feature has to preserve."""

    VOCAB = ["a", "man", "a plan", "a canal", "panama", "no", "on", "step",
             "pets", "step on", "no pets", "rats", "star", "live", "evil",
             "we", "ew", "top", "pot", "so", "os"]

    def test_search_over_phrase_units_still_closes_a_palindrome(self):
        tries = WordTries(self.VOCAB)
        scorer = _LengthScorer()
        words = centerout_search(tries, scorer, min_letters=8, beam_width=40,
                                 seed=0, max_steps=200, candidate_limit=200)
        assert words, "search found nothing to check"
        assert is_palindrome(" ".join(words))

    def test_letters_are_counted_without_the_spaces_inside_phrases(self):
        """A state holding "a plan" owes five letters, not six."""
        from llm_palindrome.centerout import COState
        state = COState(sort_key=0.0, left=("a plan",), right=(), overhang="",
                        owner="R")
        assert state.letters == 5

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_every_seed_closes_a_valid_palindrome(self, seed):
        tries = WordTries(self.VOCAB)
        words = centerout_search(tries, _LengthScorer(), min_letters=6,
                                 beam_width=40, seed=seed, max_steps=200,
                                 candidate_limit=200)
        if words:
            assert is_palindrome(" ".join(words))


class _LengthScorer:
    """Prefers longer units, so phrases get chosen and the test exercises them."""

    def word_delta(self, left, right, placement, word, growth):
        return float(len(unit_letters(word)))
