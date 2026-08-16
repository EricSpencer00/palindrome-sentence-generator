"""The v3 goal, as far as a machine can check it.

See docs/NORTH-STAR.md. A paragraph of coherent English prose, at least 100
words, whose letters read identically both ways, built from sentences that are
not themselves palindromes and were not written by somebody else.

Six of the nine criteria are mechanical: length, the mirror, no
self-palindromic units beyond the centre, no repetition, disjoint halves, and
novelty. All six now hold of the shipped endpoint, which they did not before —
the units are walked out of the vocabulary by `llm_palindrome/pairs.py` rather
than lifted from the palindrome record, and there are enough of them to carry a
hundred words.

**The other three do not hold, and nothing here should be read as saying they
do.** Grammaticality, having a subject, and reading as prose rather than as a
list of sentences that happen to parse need a blinded batch with real-prose and
salad controls. They must not be replaced by a proxy that can be automated:
four proxies have disagreed with blind judging in this project and none has
ever agreed on ranking. What ships today is a paragraph of terse two- and
three-word sentences — "War dog. Rob a log. No cotton." — and it is the
material, not the assembly, that has to improve next.

Three of these were xfail(strict=True) until the bank existed, so that the
suite would fail the moment one started passing. That is what happened, and
this file was updated rather than the marker being widened.
"""
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def served():
    from server.v2 import letter_paragraph
    return letter_paragraph(sentences=9)


def sentences_of(text):
    return [s.strip() for s in text.split(".") if s.strip()]


def letters_of(text):
    return "".join(c.lower() for c in text if c.isalpha())


class TestStructure:
    """Criteria 2-5. These pass, and are free once the assembly is right."""

    def test_2_the_whole_text_mirrors(self, served):
        """Checked without our own validator, which is the thing under test."""
        letters = letters_of(served["text"])
        assert letters == letters[::-1]

    def test_3_at_most_one_sentence_is_a_palindrome_alone(self, served):
        """The centre may be. Nothing else, or the mirror does no work."""
        from llm_palindrome.validator import is_palindrome

        selfish = [s for s in sentences_of(served["text"]) if is_palindrome(s)]
        assert len(selfish) <= 1, selfish

    def test_4_no_sentence_repeats(self, served):
        said = [s.lower() for s in sentences_of(served["text"])]
        assert len(set(said)) == len(said)

    def test_5_the_halves_share_no_sentence(self, served):
        said = [s.lower() for s in sentences_of(served["text"])]
        half = len(said) // 2
        assert not (set(said[:half]) & set(said[half + 1:]))


class TestMaterial:
    """Criteria 1 and 9 — the material, which is what v3 was about."""

    def test_1_at_least_a_hundred_words(self, served):
        import re

        assert len(re.findall(r"[A-Za-z]+", served["text"])) >= 100

    def test_9_every_unit_is_novel(self, served):
        from llm_palindrome.paragraphs import is_novel_palindrome

        borrowed = [f"{l} {r}" for l, r
                    in zip(served["units"], served["mirrors"])
                    if not is_novel_palindrome(f"{l} {r}")]
        assert not borrowed, borrowed

    def test_9_the_whole_text_is_novel(self, served):
        from llm_palindrome.paragraphs import is_novel_palindrome

        for sentence in sentences_of(served["text"]):
            assert is_novel_palindrome(sentence), sentence


class TestTheGoalIsWrittenDown:
    """The document is the contract; losing it is how shortcuts get taken."""

    def test_the_north_star_exists(self):
        assert Path("docs/NORTH-STAR.md").exists()

    def test_it_lists_every_criterion(self):
        text = Path("docs/NORTH-STAR.md").read_text()
        for n in range(1, 10):
            assert f"| {n} |" in text, f"criterion {n} missing"

    def test_it_names_the_shortcuts_that_were_actually_taken(self):
        """Each was taken here. Naming them is what makes them recognisable."""
        text = Path("docs/NORTH-STAR.md").read_text().lower()
        for shortcut in ("word-order", "self-palindromic", "verbatim",
                         "proxy", "borrowed"):
            assert shortcut in text, shortcut
