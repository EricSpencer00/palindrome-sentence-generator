"""Tests for the LLM-guided palindrome search (Norvig-style overhang search)."""
import pytest

from llm_palindrome.validator import normalize, is_palindrome
from llm_palindrome.search import consume, WordTries, beam_search
from llm_palindrome.scoring import FreqScorer
from llm_palindrome.textify import textify


TINY_DICT = [
    "stop", "pots", "spot", "tops", "live", "evil", "rats", "star",
    "no", "on", "a", "madam", "was", "saw", "draw", "ward", "dog", "god",
    "never", "even", "now", "won", "net", "ten", "i", "am",
]


class TestValidator:
    def test_normalize_strips_punctuation_and_case(self):
        assert normalize("Rats live, on NO evil star!") == "ratsliveonnoevilstar"

    def test_is_palindrome_true(self):
        assert is_palindrome("A man, a plan, a canal: Panama")

    def test_is_palindrome_false(self):
        assert not is_palindrome("This is not a palindrome.")


class TestConsume:
    """consume(letters, overhang) -> (new_overhang, flipped) or None.

    The overhang is the run of letters one half owes the other. A new word's
    letters must line up with it: either the word is swallowed by the overhang
    (same side keeps the remainder) or the word overruns it (remainder flips
    to the other side).
    """

    def test_word_is_prefix_of_overhang(self):
        assert consume("live", "liveon") == ("on", False)

    def test_word_overruns_overhang(self):
        assert consume("liveon", "live") == ("on", True)

    def test_word_exactly_matches(self):
        assert consume("live", "live") == ("", False)

    def test_mismatch_returns_none(self):
        assert consume("lix", "live") is None

    def test_empty_overhang_accepts_any_word(self):
        assert consume("dog", "") == ("dog", True)


class TestWordTries:
    def test_candidates_matching_overhang(self):
        tries = WordTries(TINY_DICT)
        # Right side owes "star...": right-side words are matched reversed,
        # so "rats" (reversed: "star") must be among candidates.
        cands = tries.right_candidates("star")
        assert "rats" in cands
        # and a word whose reversal merely starts with "star" also matches
        # (none in the tiny dict beyond rats itself, so just check no junk)
        assert all(
            "star".startswith(w[::-1]) or w[::-1].startswith("star")
            for w in cands
        )

    def test_left_candidates_forward_match(self):
        tries = WordTries(TINY_DICT)
        cands = tries.left_candidates("liveon")
        assert "live" in cands


class TestBeamSearch:
    def test_produces_valid_palindrome_from_tiny_dict(self):
        tries = WordTries(TINY_DICT)
        scorer = FreqScorer(TINY_DICT)
        words = beam_search(tries, scorer, min_letters=30, beam_width=30, seed=42)
        assert words, "search returned no result"
        text = " ".join(words)
        assert is_palindrome(text)
        assert len(normalize(text)) >= 30
        assert all(w in TINY_DICT for w in words)

    def test_respects_min_letters(self):
        tries = WordTries(TINY_DICT)
        scorer = FreqScorer(TINY_DICT)
        words = beam_search(tries, scorer, min_letters=60, beam_width=30, seed=7)
        assert len(normalize(" ".join(words))) >= 60


class TestLMPruning:
    def test_prune_callback_shapes_the_beam(self):
        """A prune hook lets a language model rescore surviving branches."""
        tries = WordTries(TINY_DICT)
        scorer = FreqScorer(TINY_DICT)
        seen = []

        def prune(states):
            seen.append(len(states))
            return states  # identity: must not break correctness

        words = beam_search(tries, scorer, min_letters=30, beam_width=20,
                            seed=3, prune=prune, prune_every=2)
        assert seen, "prune hook was never called"
        assert is_palindrome(" ".join(words))


class TestTextify:
    def test_preserves_letters_and_adds_sentences(self):
        words = ["rats", "live", "on", "no", "evil", "star",
                 "rats", "live", "on", "no", "evil", "star"]
        text = textify(words, words_per_sentence=4)
        assert normalize(text) == normalize(" ".join(words))
        assert text.count(".") >= 2, "should span multiple sentences"
        assert text[0].isupper()
