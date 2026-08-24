"""The estimator behind the paper's headline number.

`experiments/mirror_cost.py` is the only thing in the repository that computes
the price of the mirror, and every claim in `paper/` rests on it. These tests
cover the parts that could be wrong without being obviously wrong: the
segmentation is total (so no span is silently dropped from an average), the two
directions are handled by the same code path in the same units, and the
coverage statistic counts what it says it counts.

The language model is not exercised here. Loading GPT-2 in the suite would cost
more than the whole rest of it, and `Scorer` is thirty lines whose only job is
to sum token log-probabilities.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from mirror_cost import (ALPHABET, coverage, load_vocab, segment, sentences,
                         spans)

STRATEGIES = ["unigram", "fewest", "greedy"]


@pytest.fixture(scope="module")
def vocab():
    return load_vocab(str(Path(__file__).resolve().parents[1]
                          / "data" / "lexicon.txt"))


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_segmentation_is_total(vocab, strategy):
    """Anything made of letters segments, however unlike English it is.

    This is what lets the estimator average over every span instead of over the
    ones that happened to split, and it is why the penalty for a bad split is
    the language model's rather than a constant chosen in the source.
    """
    for letters in ["amanaplanacanalpanama",
                    "amanaplanacanalpanama"[::-1],
                    "qxzjvwkfbgpmycn",
                    "a"]:
        words = segment(letters, vocab, strategy)
        assert "".join(words) == letters


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_segmentation_preserves_letters(vocab, strategy):
    """A segmentation may not add, drop or reorder a letter.

    The whole measurement is a comparison over one letter sequence, so a
    segmenter that quietly changed the letters would compare two different
    texts and report the difference as a cost.
    """
    letters = "thequickbrownfoxjumpsoverthelazydog"
    for direction in (letters, letters[::-1]):
        assert "".join(segment(direction, vocab, strategy)) == direction


def test_every_unit_is_a_word_or_a_bare_letter(vocab):
    letters = "screenactorsguildandexercisednominally"
    for strategy in STRATEGIES:
        for word in segment(letters, vocab, strategy):
            assert word in vocab or (len(word) == 1 and word in ALPHABET)


def test_forward_english_segments_into_real_words(vocab):
    """The forward direction should barely need the single-letter escape.

    If it did need it often, the reported gap would be measuring the escape
    hatch rather than the mirror.
    """
    letters = "thegovernmentannouncedanewpolicyonwednesday"
    assert coverage(segment(letters, vocab, "unigram"), vocab) > 0.85


def test_reversed_english_does_not(vocab):
    """The asymmetry the measurement is about, at the level of the dictionary.

    Reversed English leaves roughly half its letters outside any real word.
    The bound here is loose on purpose; the paper reports the distribution.
    """
    letters = "thegovernmentannouncedanewpolicyonwednesday"[::-1]
    assert coverage(segment(letters, vocab, "unigram"), vocab) < 0.75


def test_coverage_ignores_short_words(vocab):
    """One- and two-letter units do not count as coverage even when real.

    That corner is where a segmenter can tile anything and claim success, and
    `shortwords.is_real_short` makes the same judgement for the search.
    """
    assert coverage(["a", "i", "of"], vocab) == 0.0
    assert coverage(["cat", "dog"], vocab) == 1.0
    assert coverage(["cat", "x"], vocab) == pytest.approx(3 / 4)


def test_vocab_excludes_the_junk_two_letter_strings():
    path = Path(__file__).resolve().parents[1] / "data" / "lexicon.txt"
    v = load_vocab(str(path))
    assert "the" in v and "an" in v
    for junk in ["bn", "cu", "eb", "ek", "iw"]:
        assert junk not in v


def test_spans_reach_the_requested_length_and_match_their_text():
    """Spans meet the target length and the two readings share one source.

    The length is a floor rather than an exact count: requiring exactness
    selects for word-length compositions that happen to sum to the target, and
    the price is a difference of two per-letter figures within one span, so the
    denominator only has to match on the two sides of a span.
    """
    import random
    text = ("The quick brown fox jumps over the lazy dog. "
            "Pack my box with five dozen liquor jugs. "
            "How vexingly quick daft zebras jump.") * 40
    sents = sentences(text)
    for n in (20, 30):
        got = spans(sents, n, 5, random.Random(0))
        assert got, f"no spans of at least {n} letters"
        for natural, letters in got:
            assert len(letters) >= n
            assert letters.isalpha() and letters.islower()
            # The natural text must be the span the letters came from, so the
            # natural-spacing score is over the same material.
            assert "".join(c for c in natural.lower() if c.isalpha()) == letters
