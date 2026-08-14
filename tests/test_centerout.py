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

    def test_deadline_returns_best_so_far_instead_of_running_on(self):
        """A public endpoint needs a bound. With an unreachable target and a
        short deadline, the search must still hand back a valid palindrome."""
        import time
        tries = WordTries(TINY_DICT)
        scorer = FreqScorer(TINY_DICT)
        t0 = time.monotonic()
        words = centerout_search(tries, scorer, min_letters=8, beam_width=30,
                                 center="", seed=3, max_steps=10**6,
                                 deadline=t0 + 0.4)
        elapsed = time.monotonic() - t0
        assert elapsed < 3.0, f"deadline ignored, ran {elapsed:.1f}s"
        assert words, "deadline should return best-so-far, not nothing"
        assert is_palindrome(" ".join(words))

    def test_maximize_letters_prefers_the_longest_closure(self):
        """Default picks the best-reading closure; a site that wants the longest
        one within a time budget needs to optimise for length instead."""
        tries = WordTries(TINY_DICT)
        scorer = FreqScorer(TINY_DICT)
        by_score = centerout_search(tries, scorer, min_letters=12, beam_width=40,
                                    center="", seed=11, max_steps=90)
        by_len = centerout_search(tries, scorer, min_letters=12, beam_width=40,
                                  center="", seed=11, max_steps=90,
                                  maximize="letters")
        assert is_palindrome(" ".join(by_len))
        assert len(normalize(" ".join(by_len))) >= len(normalize(" ".join(by_score)))

    def test_empty_center_also_works(self):
        tries = WordTries(TINY_DICT)
        scorer = FreqScorer(TINY_DICT)
        words = centerout_search(tries, scorer, min_letters=16,
                                 beam_width=30, center="", seed=5)
        assert is_palindrome(" ".join(words))
