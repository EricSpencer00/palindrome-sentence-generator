"""Scoring a half against its own shuffles, so common words stop paying twice."""
import pytest

from llm_palindrome.wordorder import order_gain, rank_halves, shuffles


class TestShuffles:
    def test_it_never_returns_the_given_order(self):
        for trial in shuffles(["a", "b", "c"], 5):
            assert trial != ["a", "b", "c"]

    def test_it_does_not_repeat_itself(self):
        out = shuffles(["a", "b", "c", "d"], 6)
        assert len({tuple(t) for t in out}) == len(out)

    def test_it_stops_when_the_words_run_out_of_orders(self):
        """Three words have five other arrangements, not eight."""
        assert len(shuffles(["a", "b", "c"], 8)) == 5

    def test_it_is_deterministic(self):
        assert shuffles(["a", "b", "c", "d"], 3) == shuffles(["a", "b", "c", "d"], 3)

    def test_a_different_seed_gives_a_different_sample(self):
        assert (shuffles(list("abcdef"), 3, seed=1)
                != shuffles(list("abcdef"), 3, seed=2))


class TestOrderGain:
    def test_it_is_the_given_score_minus_the_mean_of_the_rest(self):
        assert order_gain([1.0, 0.0, -1.0]) == 1.5

    def test_nothing_to_compare_against_is_zero_not_infinity(self):
        assert order_gain([5.0]) == 0.0
        assert order_gain([]) == 0.0

    def test_a_bag_of_words_gains_nothing(self):
        assert order_gain([0.0, 0.0, 0.0]) == 0.0


class TestRankHalves:
    def test_it_scores_every_half_once(self):
        halves = [["the", "dog", "sat"], ["sat", "the", "dog"]]
        out = rank_halves(halves, lambda texts: [0.0] * len(texts), n=2)
        assert len(out) == len(halves)

    def test_the_model_sees_every_variant_in_one_call(self):
        calls = []

        def score(texts):
            calls.append(len(texts))
            return [0.0] * len(texts)

        rank_halves([["a", "b", "c"], ["d", "e", "f"]], score, n=2)
        assert calls == [6]

    def test_word_order_is_what_it_measures(self):
        """A scorer that only likes one arrangement gives it the whole gain."""
        def score(texts):
            return [1.0 if t == "The dog sat." else 0.0 for t in texts]

        gain, = rank_halves([["the", "dog", "sat"]], score, n=3)
        assert gain == pytest.approx(1.0)

    def test_a_scorer_blind_to_order_gives_zero(self):
        def score(texts):
            return [len(t) for t in texts]

        gain, = rank_halves([["the", "dog", "sat"]], score, n=3)
        assert gain == pytest.approx(0.0)
