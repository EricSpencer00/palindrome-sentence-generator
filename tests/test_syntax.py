"""Tag-shape tests over a hand-built table, so no corpus download is needed."""
from collections import Counter

import pytest

from llm_palindrome.syntax import (plausible, readings, sentence_like, shaped,
                                   tag_table, sentence_shapes, tag_trigrams)

TAGGED = [
    [("the", "DET"), ("dog", "NOUN"), ("sat", "VERB"), (".", ".")],
    [("a", "DET"), ("man", "NOUN"), ("saw", "VERB"), ("rats", "NOUN")],
    [("rats", "NOUN"), ("live", "VERB")],
    [("draw", "VERB"), ("it", "PRON")],
]
TABLE = tag_table(TAGGED)
SHAPES = sentence_shapes(TAGGED, min_words=2, max_words=9)
TRIGRAMS = tag_trigrams(TAGGED)


class TestTagTable:
    def test_a_word_carries_every_tag_it_was_seen_with(self):
        assert TABLE["draw"] == {"VERB"}
        assert TABLE["rats"] == {"NOUN"}

    def test_it_is_lowercased(self):
        assert tag_table([[("The", "DET")]]) == {"the": frozenset({"DET"})}


class TestReadings:
    def test_an_unknown_word_kills_every_reading(self):
        """Fails closed: a word Brown never saw is not vouched for."""
        assert readings(["dog", "quokka"], TABLE) == []

    def test_the_product_is_capped(self):
        table = {w: frozenset({"NOUN", "VERB", "ADJ"}) for w in "abcdefgh"}
        assert readings(list("abcdefgh"), table, limit=10) == []

    def test_one_tag_each_gives_one_reading(self):
        assert readings(["the", "dog", "sat"], TABLE) == [
            ("DET", "NOUN", "VERB")]


class TestShaped:
    def test_a_sentence_shape_from_the_corpus_matches(self):
        assert shaped(["a", "dog", "sat"], TABLE, SHAPES)

    def test_a_shape_the_corpus_never_had_does_not(self):
        assert not shaped(["sat", "the"], TABLE, SHAPES)

    def test_the_full_stop_is_not_part_of_the_shape(self):
        """Keeping it makes every shape one slot too long."""
        assert ("DET", "NOUN", "VERB", ".") not in SHAPES


class TestPlausible:
    def test_an_attested_trigram_run_passes(self):
        assert plausible(["a", "dog", "sat"], TABLE, TRIGRAMS)

    def test_an_unattested_one_does_not(self):
        assert not plausible(["sat", "the", "dog"], TABLE, TRIGRAMS)

    def test_two_words_have_no_trigram_to_fail(self):
        assert plausible(["rats", "live"], TABLE, TRIGRAMS)


class TestSentenceLike:
    def test_it_wants_a_verb(self):
        assert not sentence_like(["a", "man"], TABLE, SHAPES)

    def test_it_wants_a_subject_shaped_opening(self):
        assert not sentence_like(["draw", "it"], TABLE, SHAPES)

    def test_it_accepts_a_subject_and_a_verb(self):
        assert sentence_like(["rats", "live"], TABLE, SHAPES)

    def test_the_conditions_must_hold_in_one_reading(self):
        """Separately they pass phrases that are not sentences in any reading."""
        table = {"draw": frozenset({"VERB", "NOUN"}),
                 "it": frozenset({"PRON"})}
        shapes = {("NOUN", "PRON")}          # shaped, but with no verb
        assert not sentence_like(["draw", "it"], table, shapes)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("nltk") is None,
    reason="nltk not installed")
class TestBrown:
    def test_the_hand_cases_come_out_as_measured(self):
        """The claim in the module docstring, run rather than remembered."""
        from llm_palindrome.syntax import brown_tables
        table, shapes, _ = brown_tables()
        assert sentence_like("to host a test on".split(), table, shapes)
        assert sentence_like("no rats live".split(), table, shapes)
        assert not sentence_like("not set at so hot".split(), table, shapes)
        assert not sentence_like("draw at left one man".split(), table, shapes)
