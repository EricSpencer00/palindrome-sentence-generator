"""Tests for the bidirectional bigram scorer.

The left half of a palindrome is built by prepending and the right half by
appending, so a single direction of bigram is only ever half useful.
"""
import math

from llm_palindrome.bigram import BigramModel


COUNTS = {
    ("new", "york"): 1000,
    ("york", "city"): 800,
    ("new", "jersey"): 50,
    ("the", "new"): 5000,
    ("cat", "sat"): 10,
}
UNI = {"new": 6000, "york": 1800, "city": 800, "jersey": 50, "the": 5000,
       "cat": 10, "sat": 10, "zebra": 1}


class TestBigramModel:
    def _m(self):
        return BigramModel(COUNTS, UNI)

    def test_forward_prefers_the_common_continuation(self):
        m = self._m()
        assert m.forward("new", "york") > m.forward("new", "jersey")

    def test_backward_prefers_the_common_predecessor(self):
        """Given 'york' comes next, 'new' should beat an unrelated word."""
        m = self._m()
        assert m.backward("new", "york") > m.backward("cat", "york")

    def test_unseen_pair_backs_off_rather_than_failing(self):
        m = self._m()
        s = m.forward("zebra", "york")
        assert isinstance(s, float) and math.isfinite(s)

    def test_unseen_pair_scores_below_a_seen_pair(self):
        m = self._m()
        assert m.forward("zebra", "york") < m.forward("new", "york")

    def test_no_context_falls_back_to_unigram_frequency(self):
        m = self._m()
        assert m.forward(None, "the") > m.forward(None, "zebra")


class TestLoader:
    def test_loads_norvig_format(self, tmp_path):
        f = tmp_path / "c2w.txt"
        f.write_text("new york\t1000\nyork city\t800\nof the\t5\n")
        m = BigramModel.from_file(str(f))
        assert m.forward("new", "york") > m.forward("new", "city")
