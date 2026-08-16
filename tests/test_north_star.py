"""The v3 goal, as a test that currently fails.

See docs/NORTH-STAR.md. A paragraph of coherent English prose, at least 100
words, whose letters read identically both ways, built from sentences that are
not themselves palindromes and were not written by somebody else.

These are the criteria a machine can check: length, the mirror, no
self-palindromic units beyond the centre, no repetition, disjoint halves, and
novelty. The ones it cannot — grammaticality, having a subject, reading as
prose rather than as a list of sentences that happen to parse — need a blinded
batch with real-prose and salad controls, and must not be replaced by a proxy
that can be automated. Four proxies have disagreed with blind judging in this
project and none has ever agreed on ranking.

The passing ones are marked as such so a regression is caught. The failing ones
are xfail(strict=True), which means the suite FAILS if one starts passing
without this file being updated — the point is to notice v3 arriving, not to
carry a permanently red suite.
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
    """Criteria 1 and 9. These fail, and are what v3 is about."""

    @pytest.mark.xfail(strict=True, reason="v3 target: 91 words, needs 100")
    def test_1_at_least_a_hundred_words(self, served):
        import re

        assert len(re.findall(r"[A-Za-z]+", served["text"])) >= 100

    @pytest.mark.xfail(strict=True,
                       reason="v3 target: every unit is a catalogued palindrome")
    def test_9_every_unit_is_novel(self, served):
        from llm_palindrome.paragraphs import is_novel_palindrome

        borrowed = [f"{l} {r}" for l, r
                    in zip(served["units"], served["mirrors"])
                    if not is_novel_palindrome(f"{l} {r}")]
        assert not borrowed, borrowed

    @pytest.mark.xfail(strict=True,
                       reason="v3 target: the assembled text recites the canon")
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
