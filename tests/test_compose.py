"""Tests for composing sentences rather than quoting them.

v2 gets judged-coherent English into a palindrome by placing whole Wikipedia
sentences atomically. It works, and it is quotation: 0 of 5 composed sentences
pass a blinded judge, 100% of the passing ones are verbatim corpus text.

The way out is the project's own idea applied one level up. Palindromicity is
not something the model gets right, it is something the search cannot violate,
because the trie only offers letter-valid continuations. Grammaticality can be
made the same kind of thing: take the part-of-speech skeleton of a real
sentence, and fill it with words the bigram model likes. What comes out matches
a shape English actually uses and is a string nobody wrote.

Then the palindrome constraint gets its say, and it is the harsh one — a
composed sentence is only useful if its mirror can be spelled at all.
"""
import pytest

from llm_palindrome.compose import (compose_sentences, mine_templates,
                                    pos_lexicon)


TAGGED = [
    [("the", "DET"), ("dog", "NOUN"), ("ran", "VERB")],
    [("the", "DET"), ("cat", "NOUN"), ("sat", "VERB")],
    [("a", "DET"), ("bird", "NOUN"), ("flew", "VERB")],
    [("dogs", "NOUN"), ("bark", "VERB")],
]


class _Bigrams:
    def forward(self, a, b):
        return 1.0 if (a, b) in {("the", "dog"), ("dog", "ran"),
                                 ("the", "cat"), ("cat", "sat")} else -4.0


class TestPosLexicon:
    def test_maps_a_word_to_its_tags(self):
        lex = pos_lexicon(TAGGED)
        assert lex["dog"] == {"NOUN"}

    def test_a_word_can_carry_several_tags(self):
        lex = pos_lexicon([[("run", "NOUN")], [("run", "VERB")]])
        assert lex["run"] == {"NOUN", "VERB"}

    def test_lowercases(self):
        assert "the" in pos_lexicon([[("The", "DET")]])


class TestMineTemplates:
    def test_finds_the_shape_of_a_sentence(self):
        got = mine_templates(TAGGED, min_words=3, max_words=5)
        assert ("DET", "NOUN", "VERB") in got

    def test_counts_repeated_shapes(self):
        got = mine_templates(TAGGED, min_words=3, max_words=5)
        assert got[("DET", "NOUN", "VERB")] == 3

    def test_respects_the_word_bounds(self):
        got = mine_templates(TAGGED, min_words=3, max_words=5)
        assert ("NOUN", "VERB") not in got


class TestComposeSentences:
    def _args(self):
        return dict(templates=mine_templates(TAGGED, 3, 5),
                    lexicon=pos_lexicon(TAGGED), bigrams=_Bigrams(),
                    vocab={"the", "dog", "cat", "ran", "sat", "a", "bird", "flew"})

    def test_every_composed_sentence_matches_a_template(self):
        lex = pos_lexicon(TAGGED)
        out = compose_sentences(n=5, **self._args())
        shapes = set(mine_templates(TAGGED, 3, 5))
        for s in out:
            words = s.split()
            assert any(all(tag in lex[w] for w, tag in zip(words, shape))
                       for shape in shapes if len(shape) == len(words))

    def test_every_word_is_in_the_vocabulary(self):
        args = self._args()
        for s in compose_sentences(n=5, **args):
            assert all(w in args["vocab"] for w in s.split())

    def test_prefers_attested_joins(self):
        """'the dog ran' scores; 'a bird flew' does not."""
        out = compose_sentences(n=1, **self._args())
        assert out[0] in {"the dog ran", "the cat sat"}

    def test_excludes_sentences_the_corpus_already_contains(self):
        args = self._args()
        out = compose_sentences(n=5, exclude={"the dog ran", "the cat sat"}, **args)
        assert "the dog ran" not in out and "the cat sat" not in out

    def test_returns_at_most_n(self):
        assert len(compose_sentences(n=2, **self._args())) <= 2

    def test_can_require_a_spellable_mirror(self):
        """The only property the palindrome actually demands of a unit."""
        args = self._args()
        out = compose_sentences(n=5, mirror_ok=lambda letters: False, **args)
        assert out == []


class TestTemplatesExcludePunctuation:
    """A tagged sentence ends with a '.' token, and a template that keeps it
    turns the full stop into a slot the composer fills with a WORD. The first
    run produced "manner as possible to" from shapes like DET NOUN VERB '.'.
    """

    def test_punctuation_tags_are_dropped_from_the_shape(self):
        tagged = [[("the", "DET"), ("dog", "DET"), ("ran", "VERB"), (".", ".")]]
        got = mine_templates(tagged, min_words=3, max_words=5)
        assert (("DET", "DET", "VERB")) in got

    def test_length_bounds_apply_after_punctuation_is_dropped(self):
        tagged = [[("a", "DET"), ("b", "NOUN"), (".", ".")]]
        assert mine_templates(tagged, min_words=3, max_words=5) == {}


class TestPoolOrdering:
    """`_by_tag` returned words in dict order and the composer truncated the
    pool, so it saw 300 arbitrary words per tag — "irregularities" before
    "the". The pool has to be ranked before it is cut.
    """

    def test_pool_is_ordered_by_the_supplied_rank(self):
        from llm_palindrome.compose import _by_tag
        lex = {"zebra": {"NOUN"}, "dog": {"NOUN"}}
        table = _by_tag(lex, {"zebra", "dog"}, rank=lambda w: 0 if w == "dog" else 1)
        assert table["NOUN"][0] == "dog"

    def test_pool_without_a_rank_is_still_returned(self):
        from llm_palindrome.compose import _by_tag
        assert _by_tag({"dog": {"NOUN"}}, {"dog"})["NOUN"] == ["dog"]


class TestSpanNovelty:
    """`exclude` held whole corpus SENTENCES, which is not enough.

    "it is a good idea" is not a sentence in Brown, but it sits inside one, and
    a composition that reproduces it is a quote whatever the sentence table
    says. Novelty has to be checked against spans, not sentences.
    """

    def test_a_composition_matching_a_corpus_span_is_rejected(self):
        args = dict(templates=mine_templates(TAGGED, 3, 5),
                    lexicon=pos_lexicon(TAGGED), bigrams=_Bigrams(),
                    vocab={"the", "dog", "cat", "ran", "sat", "a", "bird", "flew"})
        out = compose_sentences(n=5, is_novel=lambda s: False, **args)
        assert out == []

    def test_novel_compositions_survive(self):
        args = dict(templates=mine_templates(TAGGED, 3, 5),
                    lexicon=pos_lexicon(TAGGED), bigrams=_Bigrams(),
                    vocab={"the", "dog", "cat", "ran", "sat", "a", "bird", "flew"})
        assert compose_sentences(n=5, is_novel=lambda s: True, **args)


class TestSentenceShapedTemplates:
    """A template is only useful if what fills it can be a sentence.

    Brown's tagged "sentences" include headings and list items, so mining by
    length alone yields shapes like ADP DET ADJ NOUN — which fills as "for more
    information" and reads as a fragment. A judge rejected 19 of 20 of these.

    Requiring a verb, and an opener that can be a subject, is the cheapest
    filter that separates a clause from a noun phrase.
    """

    def test_a_shape_without_a_verb_is_rejected(self):
        got = mine_templates([[("for", "ADP"), ("more", "ADJ"), ("info", "NOUN")]],
                             min_words=3, max_words=6, sentence_shaped=True)
        assert got == {}

    def test_a_shape_not_opening_on_a_subject_is_rejected(self):
        got = mine_templates([[("for", "ADP"), ("it", "PRON"), ("ran", "VERB")]],
                             min_words=3, max_words=6, sentence_shaped=True)
        assert got == {}

    def test_a_clause_is_kept(self):
        got = mine_templates([[("the", "DET"), ("dog", "NOUN"), ("ran", "VERB")]],
                             min_words=3, max_words=6, sentence_shaped=True)
        assert ("DET", "NOUN", "VERB") in got

    def test_a_pronoun_opener_is_a_subject(self):
        got = mine_templates([[("it", "PRON"), ("ran", "VERB"), ("far", "ADV")]],
                             min_words=3, max_words=6, sentence_shaped=True)
        assert ("PRON", "VERB", "ADV") in got

    def test_the_filter_is_off_by_default(self):
        got = mine_templates([[("for", "ADP"), ("more", "ADJ"), ("info", "NOUN")]],
                             min_words=3, max_words=6)
        assert got != {}
