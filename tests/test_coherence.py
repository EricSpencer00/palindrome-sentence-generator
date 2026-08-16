"""Tests for the global coherence metric.

Every other number in this project has a horizon of two words. This one asks
whether a text's own opening informs its own ending, which is the thing
"holds a topic" actually means — so the test that matters most is the one
where the metric must report ZERO: a scorer that ignores its prefix entirely
describes a text with no long-range structure, and the metric has to say so
rather than reporting whatever fluency happens to be lying around.
"""
import math

import pytest

from llm_palindrome.coherence import CoherenceMetric, split_at_word


CONTROLS = ["the quarterly earnings report was filed late again",
            "she walked the dog around the reservoir at dusk"]


class MemorylessScorer:
    """Assigns each token a fixed cost regardless of what precedes it."""

    def __init__(self, per_token: float = -3.0):
        self.per_token = per_token

    def conditional_logprobs(self, prefix, tail):
        return [self.per_token] * len(tail.split())


class TopicScorer:
    """Rewards tail words that already appeared in the prefix.

    A crude stand-in for a model that uses long-range context: text that
    reuses its own vocabulary gets cheaper, foreign context does not help.
    """

    def conditional_logprobs(self, prefix, tail):
        seen = set(prefix.split())
        return [-1.0 if w in seen else -5.0 for w in tail.split()]


class TestSplitAtWord:
    def test_splits_into_two_nonempty_halves(self):
        head, tail = split_at_word("one two three four five six")
        assert head.split() and tail.split()

    def test_split_preserves_every_word_in_order(self):
        text = "one two three four five six seven"
        head, tail = split_at_word(text)
        assert f"{head} {tail}".split() == text.split()

    def test_split_lands_near_the_middle(self):
        head, tail = split_at_word("a b c d e f g h i j")
        assert abs(len(head.split()) - len(tail.split())) <= 1


class TestCoherenceMetric:
    def test_memoryless_scorer_reports_no_coherence(self):
        """The load-bearing test. No prefix-dependence must read as zero."""
        m = CoherenceMetric(MemorylessScorer(), controls=CONTROLS, skip_tokens=0)
        result = m.score("alpha beta gamma delta epsilon zeta eta theta")
        assert result.gain == pytest.approx(0.0, abs=1e-9)

    def test_memoryless_gain_is_zero_whatever_the_fluency(self):
        """Gain must not inherit the absolute score, only the difference."""
        a = CoherenceMetric(MemorylessScorer(-1.0), controls=CONTROLS, skip_tokens=0)
        b = CoherenceMetric(MemorylessScorer(-9.0), controls=CONTROLS, skip_tokens=0)
        text = "alpha beta gamma delta epsilon zeta eta theta"
        assert a.score(text).gain == pytest.approx(b.score(text).gain, abs=1e-9)

    def test_self_referring_text_scores_positive(self):
        m = CoherenceMetric(TopicScorer(), controls=CONTROLS, skip_tokens=0)
        result = m.score("otter otter otter otter otter otter otter otter")
        assert result.gain > 0

    def test_text_sharing_nothing_with_its_own_opening_scores_no_better(self):
        m = CoherenceMetric(TopicScorer(), controls=CONTROLS, skip_tokens=0)
        result = m.score("alpha beta gamma delta epsilon zeta eta theta")
        assert result.gain <= 0

    def test_skip_tokens_excludes_the_junction(self):
        """The first tail tokens sit against a seam and must not be counted."""
        m = CoherenceMetric(TopicScorer(), controls=CONTROLS, skip_tokens=2)
        result = m.score("otter otter otter otter alpha beta otter otter")
        assert result.scored_tokens == 2

    def test_too_short_to_split_returns_none_rather_than_a_number(self):
        m = CoherenceMetric(MemorylessScorer(), controls=CONTROLS, skip_tokens=4)
        assert m.score("one two three").gain is None

    def test_result_reports_how_many_tokens_it_averaged_over(self):
        m = CoherenceMetric(MemorylessScorer(), controls=CONTROLS, skip_tokens=1)
        result = m.score("a b c d e f g h i j")
        assert result.scored_tokens == len(result_tail_words(result))


def result_tail_words(result):
    return result.tail.split()[1:]


class TestGainIsFiniteAndSigned:
    def test_gain_is_a_finite_float(self):
        m = CoherenceMetric(TopicScorer(), controls=CONTROLS, skip_tokens=0)
        assert math.isfinite(m.score("otter otter otter otter otter otter").gain)


class TestGPT2ConditionalScorer:
    """The one property the metric cannot survive being wrong about.

    `CoherenceMetric` subtracts the control run from the own run elementwise,
    so if the tail retokenized when the prefix changed, the two runs would be
    over different units and the difference would be noise.
    """

    def _scorer(self):
        torch = pytest.importorskip("torch")
        pytest.importorskip("transformers")
        from llm_palindrome.lm_scoring import GPT2ConditionalScorer
        return GPT2ConditionalScorer("gpt2", device="cpu")

    def test_tail_length_does_not_depend_on_the_prefix(self):
        s = self._scorer()
        tail = "the reservoir was frozen over by the end of january"
        short = s.conditional_logprobs("a", tail)
        long = s.conditional_logprobs(
            "the quarterly earnings report was filed late again and nobody noticed", tail)
        assert len(short) == len(long) > 0

    def test_a_relevant_prefix_beats_an_irrelevant_one(self):
        s = self._scorer()
        tail = "the reservoir was frozen over by the end of january"
        related = s.conditional_logprobs(
            "we walked out to the reservoir every winter to see the ice", tail)
        unrelated = s.conditional_logprobs(
            "quarterly earnings were restated after the audit committee met", tail)
        assert sum(related) / len(related) > sum(unrelated) / len(unrelated)


class TestPerTextControls:
    """Foreign controls conflate topic with vocabulary reuse.

    Calibration showed the foreign-prefix gain rising when a paragraph's words
    were SHUFFLED, because a control drawn from another text differs in
    vocabulary as well as in order — so the metric rewarded a text for reusing
    its own words and could not see structure at all. Holding the vocabulary
    fixed and varying only the order needs the control to be derived from the
    text being scored, which means it cannot be fixed at construction.
    """

    def test_controls_can_be_supplied_per_call(self):
        m = CoherenceMetric(TopicScorer(), controls=CONTROLS, skip_tokens=0)
        result = m.score("alpha beta gamma delta epsilon zeta eta theta",
                         controls=["epsilon zeta eta theta"])
        assert result.gain < 0     # the control saw the tail; the head did not

    def test_per_call_controls_override_the_constructor_controls(self):
        m = CoherenceMetric(TopicScorer(), controls=CONTROLS, skip_tokens=0)
        text = "otter otter otter otter otter otter otter otter"
        assert m.score(text).gain != m.score(text, controls=["otter otter"]).gain
