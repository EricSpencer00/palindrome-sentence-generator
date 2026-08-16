"""Tests for scoring that can see the debt it is creating.

Every scorer in this project takes (left, right, placement, word, growth) — the
text placed so far — and none of them takes the overhang. So a word that reads
beautifully and leaves a run of letters the other half can never spell scores
exactly as well as one that leaves a clean debt. The search is subject to a
constraint its scorers cannot see, which is why neither a larger vocabulary nor
a better judge moved anything: both improve ranking on the visible half, and
the cost is always paid on the half that is not yet written.

One-step lookahead fixes the blindness: before taking a word, ask how many ways
the debt it leaves could be repaid.
"""
import pytest

from llm_palindrome.overhang import DebtIndex, OverhangAware
from llm_palindrome.search import WordTries


VOCAB = ["stop", "pots", "star", "rats", "on", "no", "live", "evil", "a", "i",
         "step", "pets", "was", "saw"]


class _Flat:
    """Scores every word the same, so only the debt term can move a ranking."""

    wants_overhang = False

    def word_delta(self, left, right, placement, word, growth):
        return 1.0


class TestDebtIndex:
    def _index(self):
        return DebtIndex(WordTries(VOCAB))

    def test_an_unspellable_debt_has_no_options(self):
        assert self._index().options("qqzz") == 0

    def test_a_spellable_debt_has_options(self):
        assert self._index().options("pots") > 0

    def test_an_empty_debt_is_free(self):
        assert self._index().options("") > 0

    def test_repeated_lookups_agree(self):
        idx = self._index()
        assert idx.options("pots") == idx.options("pots")

    def test_caches_rather_than_rewalking_the_trie(self):
        idx = self._index()
        idx.options("pots")
        assert "pots" in idx.cache

    def test_a_prefix_of_a_word_is_still_repayable(self):
        """'sta' is not a word, but 'star' begins with it."""
        assert self._index().options("sta") > 0


class TestOverhangAware:
    def _scorer(self, weight=1.0):
        return OverhangAware(_Flat(), DebtIndex(WordTries(VOCAB)), debt_weight=weight)

    def test_it_advertises_that_it_wants_the_overhang(self):
        assert self._scorer().wants_overhang is True

    def test_a_repayable_debt_beats_a_dead_one(self):
        s = self._scorer()
        good = s.word_delta((), ("stop",), "R", "stop", "append", overhang="pots")
        dead = s.word_delta((), ("stop",), "R", "stop", "append", overhang="qqzz")
        assert good > dead

    def test_zero_weight_ignores_the_debt_entirely(self):
        s = self._scorer(weight=0.0)
        a = s.word_delta((), ("stop",), "R", "stop", "append", overhang="pots")
        b = s.word_delta((), ("stop",), "R", "stop", "append", overhang="qqzz")
        assert a == b

    def test_it_still_works_when_no_overhang_is_supplied(self):
        """The search must be able to call it the old way."""
        s = self._scorer()
        assert isinstance(s.word_delta((), ("stop",), "R", "stop", "append"), float)

    def test_it_passes_the_base_score_through(self):
        s = self._scorer(weight=0.0)
        assert s.word_delta((), ("stop",), "R", "stop", "append") == 1.0


class TestSearchSuppliesTheOverhang:
    def test_the_search_hands_the_new_overhang_to_a_scorer_that_wants_it(self):
        from llm_palindrome.centerout import centerout_search

        seen = []

        class Spy:
            wants_overhang = True

            def word_delta(self, left, right, placement, word, growth, overhang=None):
                seen.append(overhang)
                return 1.0

        centerout_search(WordTries(VOCAB), Spy(), min_letters=4, beam_width=8,
                         seed=0, max_steps=6, candidate_limit=50)
        assert seen and all(o is not None for o in seen)

    def test_a_scorer_that_does_not_want_it_is_called_the_old_way(self):
        from llm_palindrome.centerout import centerout_search

        class Old:
            def word_delta(self, left, right, placement, word, growth):
                return 1.0    # would TypeError if handed an overhang

        words = centerout_search(WordTries(VOCAB), Old(), min_letters=4,
                                 beam_width=8, seed=0, max_steps=6,
                                 candidate_limit=50)
        assert isinstance(words, list)
