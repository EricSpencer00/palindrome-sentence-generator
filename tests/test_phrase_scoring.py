"""Tests for scoring multi-word units.

Putting phrases in the trie achieved nothing on its own: a first run placed
zero of 20000 available phrases, because `CoherentScorer` looked each unit up
in the bigram model as if it were one word. "new york" is not a key, so every
phrase took the unseen-pair backoff — the inventory was in the trie and priced
out of the beam.

A unit joins its neighbours at its EDGES. What sits next to the preceding text
is the unit's first word, what the following text sees is its last, and the
join in between belongs to the unit itself.
"""
import pytest

from llm_palindrome.bigram import BigramModel
from llm_palindrome.scoring import CoherentScorer, first_word, last_word


COUNTS = {
    ("new", "york"): 1000,
    ("york", "city"): 800,
    ("in", "new"): 900,
    ("the", "city"): 700,
    ("zebra", "york"): 1,
}
UNI = {"new": 6000, "york": 1800, "city": 800, "in": 5000, "the": 9000,
       "zebra": 1, "quartz": 1}


def model():
    return BigramModel(COUNTS, UNI)


class TestUnitEdges:
    def test_single_word_is_both_its_edges(self):
        assert first_word("york") == "york" and last_word("york") == "york"

    def test_phrase_edges_are_its_outer_words(self):
        assert first_word("new york") == "new"
        assert last_word("new york") == "york"


class TestPhraseJoins:
    def _scorer(self):
        return CoherentScorer(model())

    def test_appended_phrase_joins_on_its_first_word(self):
        """'in' + 'new york' must see the attested pair (in, new)."""
        s = self._scorer()
        good = s.word_delta((), ("in", "new york"), "R", "new york", "append")
        bad = s.word_delta((), ("in", "quartz york"), "R", "quartz york", "append")
        assert good > bad

    def test_prepended_phrase_joins_on_its_last_word(self):
        """'new york' + 'city' must see the attested pair (york, city)."""
        s = self._scorer()
        good = s.word_delta(("new york", "city"), (), "L", "new york", "prepend")
        bad = s.word_delta(("new quartz", "city"), (), "L", "new quartz", "prepend")
        assert good > bad

    def test_a_phrase_is_not_penalised_for_not_being_a_bigram_key(self):
        """The bug that made the inventory inert.

        A phrase joined to an attested neighbour must not score below the same
        join made with its first word alone — the phrase carries strictly more
        attested structure.
        """
        s = self._scorer()
        phrase = s.word_delta((), ("in", "new york"), "R", "new york", "append")
        single = s.word_delta((), ("in", "new"), "R", "new", "append")
        assert phrase >= single

    def test_the_join_inside_a_phrase_counts_toward_its_score(self):
        s = self._scorer()
        attested = s.word_delta((), ("in", "new york"), "R", "new york", "append")
        unattested = s.word_delta((), ("in", "new zebra"), "R", "new zebra", "append")
        assert attested > unattested


class TestRepetitionSeesInsidePhrases:
    def test_a_word_repeated_via_a_phrase_is_penalised(self):
        """"york" then "new york" is a repeat, even though the units differ."""
        s = self._scorer() if False else CoherentScorer(model())
        fresh = s.word_delta((), ("in", "new york"), "R", "new york", "append")
        repeat = s.word_delta(("york",), ("in", "new york"), "R", "new york", "append")
        assert repeat < fresh


class _FlatBigrams:
    """Every join costs the same, so only the length term can move a score."""

    def forward(self, a, b):
        return 0.0

    def backward(self, a, b):
        return 0.0


class TestLengthTermCountsLetters:
    def test_a_phrase_is_measured_by_its_letters_not_its_characters(self):
        """Otherwise a phrase is paid for its separator, which is not in the
        palindrome — the mirror never sees a space."""
        s = CoherentScorer(_FlatBigrams(), length_weight=1.0, freq_weight=0.0)
        score = s.word_delta((), ("in", "new york"), "R", "new york", "append")
        assert score == len("newyork")

    def test_a_single_word_is_measured_the_same_way(self):
        s = CoherentScorer(_FlatBigrams(), length_weight=1.0, freq_weight=0.0)
        assert s.word_delta((), ("in", "york"), "R", "york", "append") == len("york")


class TestPhraseWeightIsTunable:
    """How much a unit is paid for the joins it carries internally.

    At 1.0 a phrase is paid in full, and a first run showed that lets a good
    internal join offset a bad join at the unit's edge: phrases got picked, and
    attested-bigram coverage fell from 0.757 to 0.52. At 0.0 a phrase is still
    reachable and still not penalised, but it cannot buy its way past a bad
    boundary. The right value is an empirical question, so it is a dial.
    """

    def test_zero_weight_drops_the_internal_join_entirely(self):
        s = CoherentScorer(model(), phrase_weight=0.0, freq_weight=0.0,
                           length_weight=0.0)
        attested = s.word_delta((), ("in", "new york"), "R", "new york", "append")
        unattested = s.word_delta((), ("in", "new zebra"), "R", "new zebra", "append")
        assert attested == unattested

    def test_full_weight_counts_it(self):
        s = CoherentScorer(model(), phrase_weight=1.0, freq_weight=0.0,
                           length_weight=0.0)
        attested = s.word_delta((), ("in", "new york"), "R", "new york", "append")
        unattested = s.word_delta((), ("in", "new zebra"), "R", "new zebra", "append")
        assert attested > unattested

    def test_a_phrase_is_still_not_penalised_at_zero_weight(self):
        """The original bug must not come back when the dial is turned down."""
        s = CoherentScorer(model(), phrase_weight=0.0)
        phrase = s.word_delta((), ("in", "new york"), "R", "new york", "append")
        single = s.word_delta((), ("in", "new"), "R", "new", "append")
        assert phrase >= single


class TestLongUnitBonus:
    """Long units are placeable — forcing them closed 8/8 searches with 6-word
    units in the output — but they lose on score under the default weights: a
    6-gram of common words earns less than the six short frequent words the
    search would otherwise take. So how much a unit is worth for being LONG is
    its own dial, separate from how much its internal joins are worth.
    """

    def test_zero_bonus_leaves_scoring_unchanged(self):
        plain = CoherentScorer(model(), long_bonus=0.0)
        assert plain.word_delta((), ("in", "new york"), "R", "new york", "append") \
            == CoherentScorer(model()).word_delta((), ("in", "new york"), "R",
                                                  "new york", "append")

    def test_bonus_scales_with_words_beyond_the_first(self):
        a = CoherentScorer(model(), long_bonus=0.0)
        b = CoherentScorer(model(), long_bonus=10.0)
        delta = (b.word_delta((), ("in", "new york"), "R", "new york", "append")
                 - a.word_delta((), ("in", "new york"), "R", "new york", "append"))
        assert delta == pytest.approx(10.0)

    def test_a_single_word_earns_no_bonus(self):
        a = CoherentScorer(model(), long_bonus=0.0)
        b = CoherentScorer(model(), long_bonus=10.0)
        assert (a.word_delta((), ("in", "york"), "R", "york", "append")
                == b.word_delta((), ("in", "york"), "R", "york", "append"))


class TestShortWordPenalty:
    """Banning junk short words is not enough on its own.

    "of to in is it" are all real English and all fit any overhang, so a search
    with the fragments removed will simply lean harder on the legitimate short
    words. Real English runs 18.5% one- and two-letter words; the generator ran
    52.6%. The list fixes WHICH short words; this fixes HOW MANY.
    """

    def test_a_short_word_is_penalised(self):
        s = CoherentScorer(_FlatBigrams(), freq_weight=0.0, length_weight=0.0,
                           short_penalty=5.0)
        assert s.word_delta((), ("in", "it"), "R", "it", "append") == -5.0

    def test_a_long_word_is_not(self):
        s = CoherentScorer(_FlatBigrams(), freq_weight=0.0, length_weight=0.0,
                           short_penalty=5.0)
        assert s.word_delta((), ("in", "elephant"), "R", "elephant", "append") == 0.0

    def test_a_multi_word_unit_is_exempt(self):
        """Superseded by TestShortPenaltyExemptsValidatedUnits — kept here so
        the original intent, per-word charging, is visibly retired."""
        s = CoherentScorer(_FlatBigrams(), freq_weight=0.0, length_weight=0.0,
                           short_penalty=5.0)
        assert s.word_delta((), ("in", "it is"), "R", "it is", "append") == 0.0

    def test_zero_penalty_restores_the_old_behaviour(self):
        s = CoherentScorer(_FlatBigrams(), freq_weight=0.0, length_weight=0.0,
                           short_penalty=0.0)
        assert s.word_delta((), ("in", "it"), "R", "it", "append") == 0.0


class TestShortPenaltyExemptsValidatedUnits:
    """The penalty is aimed at FILLER, not at English.

    Charging it inside a composed sentence punishes the sentence for being
    English: "i do not know how" carries three short words and paid 36 while
    the search's own filler paid 12 a word. Judged-coherent sentences fell from
    18/20 to 1/20 as the beam moved to fragments like "look for more
    information", which are short-word-poor and sentence-poor together.

    A multi-word unit was validated before it entered the trie. Its short words
    are what English is 18.5% made of.
    """

    def test_a_multi_word_unit_pays_no_short_penalty(self):
        s = CoherentScorer(_FlatBigrams(), freq_weight=0.0, length_weight=0.0,
                           short_penalty=5.0, long_bonus=0.0)
        assert s.word_delta((), ("in", "it is"), "R", "it is", "append") == 0.0

    def test_a_lone_short_word_still_pays(self):
        s = CoherentScorer(_FlatBigrams(), freq_weight=0.0, length_weight=0.0,
                           short_penalty=5.0, long_bonus=0.0)
        assert s.word_delta((), ("in", "it"), "R", "it", "append") == -5.0

    def test_a_lone_long_word_pays_nothing(self):
        s = CoherentScorer(_FlatBigrams(), freq_weight=0.0, length_weight=0.0,
                           short_penalty=5.0, long_bonus=0.0)
        assert s.word_delta((), ("in", "elephant"), "R", "elephant", "append") == 0.0


class TestUnitQualityBonus:
    """The inventory is ranked and the search never saw the ranking.

    1500 composed sentences go into the trie ordered by GPT-2, and the search
    picks among them by bigrams, frequency and length — none of which is the
    thing that ordered them. So it placed "A little bit of time to get" while
    "a little more information is available" sat unused, and a judge passed 2
    of 20. Quality has to travel with the unit.
    """

    def test_a_better_unit_scores_higher(self):
        s = CoherentScorer(_FlatBigrams(), freq_weight=0.0, length_weight=0.0,
                           long_bonus=0.0,
                           unit_bonus={"new york": 5.0, "new jersey": 1.0})
        good = s.word_delta((), ("in", "new york"), "R", "new york", "append")
        worse = s.word_delta((), ("in", "new jersey"), "R", "new jersey", "append")
        assert good - worse == pytest.approx(4.0)

    def test_a_unit_absent_from_the_table_gets_nothing(self):
        s = CoherentScorer(_FlatBigrams(), freq_weight=0.0, length_weight=0.0,
                           long_bonus=0.0, unit_bonus={"new york": 5.0})
        assert s.word_delta((), ("in", "york"), "R", "york", "append") == 0.0

    def test_no_table_is_the_old_behaviour(self):
        a = CoherentScorer(_FlatBigrams(), freq_weight=0.0, length_weight=0.0,
                           long_bonus=0.0)
        b = CoherentScorer(_FlatBigrams(), freq_weight=0.0, length_weight=0.0,
                           long_bonus=0.0, unit_bonus={})
        args = ((), ("in", "new york"), "R", "new york", "append")
        assert a.word_delta(*args) == b.word_delta(*args)
