"""Tests for the control sentences that blind the judge.

The judge's verdict on the palindromes only means something if the judge is
shown to pass real English and reject word salad in the same batch. Generation
6 failed that check on 3 of 5 real controls — and the judge was right every
time: the sampler was pulling wikitext fragments like

    See . " section Reproduction ( "
    " We decided to go in and film her doing her thing [ ...
    During the winter of 1981 , only females lighter than 1 .

A control that is not actually a well-formed sentence cannot test anything.
"""
from experiments.phrase_loop import is_clean_sentence


class TestIsCleanSentence:
    def test_accepts_an_ordinary_sentence(self):
        assert is_clean_sentence("The dog ran across the field.")

    def test_rejects_unbalanced_quotes(self):
        assert not is_clean_sentence('" We decided to go in and film her thing')

    def test_rejects_brackets(self):
        assert not is_clean_sentence("See . [ section Reproduction ]")

    def test_rejects_an_ellipsis(self):
        assert not is_clean_sentence("She went on to say ...")

    def test_rejects_a_dangling_numeral(self):
        assert not is_clean_sentence("only females lighter than 1 .")

    def test_rejects_a_stray_internal_period(self):
        assert not is_clean_sentence("See . section Reproduction here now")

    def test_requires_terminal_punctuation(self):
        assert not is_clean_sentence("The dog ran across the field")

    def test_requires_a_capital_opening(self):
        assert not is_clean_sentence("the dog ran across the field.")

    def test_rejects_too_short(self):
        assert not is_clean_sentence("Yes.")
