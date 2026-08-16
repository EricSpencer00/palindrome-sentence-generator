"""Tests for choosing where sentences break.

`textify` cuts every N words, which is what the judge saw in generation 1: it
rejected 15 of 15 palindrome sentences while passing 3 of 3 real ones. A fixed
stride puts the period wherever it lands, so a run of words that did read as a
clause gets cut through the middle of it and a period arrives mid-phrase.

The search already knows where the text is weakest — the bigram model scored
every join on the way in. Breaking at the LOWEST-scoring joins puts the period
where the text has already fallen apart, which is the only place a period can
be honest.

The palindrome invariant is what makes any of this legal: punctuation and case
are invisible to `normalize`, so sentence breaks are free. Every test here
checks that first.
"""
import pytest

from llm_palindrome.textify import segment_at_weak_joins, textify
from llm_palindrome.validator import normalize


class _Bigrams:
    """Attested pairs score high; everything else scores low."""

    PAIRS = {("the", "cat"), ("cat", "sat"), ("on", "the"), ("the", "mat"),
             ("a", "dog"), ("dog", "ran")}

    def forward(self, a, b):
        return 5.0 if (a, b) in self.PAIRS else -5.0


class TestSegmentAtWeakJoins:
    def test_breaks_at_the_weakest_join(self):
        words = ["the", "cat", "sat", "zebra", "a", "dog", "ran"]
        sents = segment_at_weak_joins(words, _Bigrams(), sentences=2)
        assert sents[0] == ["the", "cat", "sat"]

    def test_produces_the_requested_number_of_sentences(self):
        words = ["the", "cat", "sat", "on", "the", "mat", "a", "dog", "ran"]
        sents = segment_at_weak_joins(words, _Bigrams(), sentences=3)
        assert len(sents) == 3

    def test_every_word_survives_in_order(self):
        words = ["the", "cat", "sat", "zebra", "a", "dog", "ran"]
        sents = segment_at_weak_joins(words, _Bigrams(), sentences=3)
        assert [w for s in sents for w in s] == words

    def test_never_emits_an_empty_sentence(self):
        words = ["the", "cat", "sat", "zebra", "a", "dog", "ran"]
        sents = segment_at_weak_joins(words, _Bigrams(), sentences=5)
        assert all(s for s in sents)

    def test_asking_for_more_sentences_than_words_gives_one_word_each(self):
        words = ["the", "cat", "sat"]
        sents = segment_at_weak_joins(words, _Bigrams(), sentences=99)
        assert len(sents) == 3

    def test_single_word_is_a_single_sentence(self):
        assert segment_at_weak_joins(["cat"], _Bigrams(), sentences=3) == [["cat"]]

    def test_empty_input_gives_no_sentences(self):
        assert segment_at_weak_joins([], _Bigrams(), sentences=3) == []

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 8])
    def test_the_letters_are_never_altered(self, n):
        """The whole feature is only legal because of this."""
        words = ["the", "cat", "sat", "zebra", "a", "dog", "ran", "on", "the", "mat"]
        sents = segment_at_weak_joins(words, _Bigrams(), sentences=n)
        rendered = " ".join(" ".join(s) for s in sents)
        assert normalize(rendered) == normalize(" ".join(words))


class TestTextifyAcceptsSegments:
    def test_textify_renders_supplied_segments(self):
        out = textify(["the", "cat", "sat", "a", "dog", "ran"],
                      segments=[["the", "cat", "sat"], ["a", "dog", "ran"]])
        assert out == "The cat sat. A dog ran."

    def test_supplied_segments_preserve_the_letters(self):
        words = ["the", "cat", "sat", "a", "dog", "ran"]
        out = textify(words, segments=[["the", "cat"], ["sat", "a", "dog", "ran"]])
        assert normalize(out) == normalize(" ".join(words))


class _NullBigrams:
    def forward(self, a, b):
        return 0.0


class TestSegmentAtUnits:
    """Isolate the attested unit as its own sentence.

    Once clause-length n-grams are placed, the output contains runs of real
    English — "i was unable to make" — glued between letter-filler. Cutting at
    the weakest bigram join ignores that structure and slices through the unit.
    The search knows exactly where those runs are: they are the units it chose.
    """

    def test_a_long_unit_becomes_its_own_sentence(self):
        from llm_palindrome.textify import segment_at_units
        segs = segment_at_units(["oo", "i was unable to make", "eg"], min_unit_words=3)
        assert ["i", "was", "unable", "to", "make"] in segs

    def test_short_units_are_grouped_around_it(self):
        from llm_palindrome.textify import segment_at_units
        segs = segment_at_units(["oo", "eg", "i was unable to make", "na", "red"],
                                min_unit_words=3)
        assert segs[0] == ["oo", "eg"] and segs[-1] == ["na", "red"]

    def test_every_word_survives_in_order(self):
        from llm_palindrome.textify import segment_at_units
        units = ["oo", "i was unable to make", "na", "the end of the", "red"]
        segs = segment_at_units(units, min_unit_words=3)
        flat = [w for s in segs for w in s]
        assert flat == [w for u in units for w in u.split()]

    def test_no_empty_sentences(self):
        from llm_palindrome.textify import segment_at_units
        segs = segment_at_units(["a b c d", "e f g h"], min_unit_words=3)
        assert all(s for s in segs)

    def test_all_short_units_give_one_sentence(self):
        from llm_palindrome.textify import segment_at_units
        assert segment_at_units(["oo", "eg", "na"], min_unit_words=3) == [["oo", "eg", "na"]]

    def test_letters_are_preserved(self):
        from llm_palindrome.textify import segment_at_units, textify
        units = ["oo", "i was unable to make", "na", "red"]
        words = [w for u in units for w in u.split()]
        out = textify(words, segments=segment_at_units(units, min_unit_words=3))
        assert normalize(out) == normalize(" ".join(words))
