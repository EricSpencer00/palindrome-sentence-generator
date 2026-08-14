"""Tests for center-out growth (the mirror image of the outside-in search)."""
import pytest

from llm_palindrome.centerout import consume_suffix, centerout_search
from llm_palindrome.search import WordTries
from llm_palindrome.scoring import FreqScorer
from llm_palindrome.validator import is_palindrome, normalize

TINY_DICT = [
    "stop", "pots", "spot", "tops", "live", "evil", "rats", "star",
    "no", "on", "a", "madam", "was", "saw", "draw", "ward", "dog", "god",
    "never", "even", "now", "won", "net", "ten", "i", "am", "level", "deed",
]


class TestConsumeSuffix:
    """Left-side words are PREPENDED, so they fill the END of the owed run,
    not the beginning. That is the one asymmetry center-out introduces."""

    def test_word_is_suffix_of_overhang(self):
        assert consume_suffix("dc", "abdc") == ("ab", False)

    def test_word_overruns_overhang(self):
        assert consume_suffix("abdc", "dc") == ("ab", True)

    def test_exact_match(self):
        assert consume_suffix("dc", "dc") == ("", False)

    def test_mismatch(self):
        assert consume_suffix("xy", "abdc") is None


class TestCenterOutSearch:
    def test_output_is_a_valid_palindrome(self):
        tries = WordTries(TINY_DICT)
        scorer = FreqScorer(TINY_DICT)
        words = centerout_search(tries, scorer, min_letters=20,
                                 beam_width=30, center="level", seed=1)
        assert words, "center-out search returned nothing"
        text = " ".join(words)
        assert is_palindrome(text), f"not a palindrome: {text!r}"
        assert len(normalize(text)) >= 20

    def test_center_appears_in_the_middle(self):
        tries = WordTries(TINY_DICT)
        scorer = FreqScorer(TINY_DICT)
        words = centerout_search(tries, scorer, min_letters=20,
                                 beam_width=30, center="madam", seed=2)
        assert "madam" in words

    def test_empty_center_also_works(self):
        tries = WordTries(TINY_DICT)
        scorer = FreqScorer(TINY_DICT)
        words = centerout_search(tries, scorer, min_letters=16,
                                 beam_width=30, center="", seed=5)
        assert is_palindrome(" ".join(words))
