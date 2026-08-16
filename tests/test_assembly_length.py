"""Choosing how many pairs to place, and in what order."""
from llm_palindrome.paragraphs import (adjacent_links, enough_pairs,
                                       order_pairs, paragraph_words, render)

PAIRS = [
    (["draw", "no", "tip"], ["pit", "on", "ward"]),
    (["step", "on", "no", "pets"], ["step", "on", "no", "pets"]),
]


class TestLength:
    def test_counts_both_halves(self):
        assert paragraph_words(PAIRS) == 14

    def test_counts_the_centre(self):
        assert paragraph_words(PAIRS[:1], center=["racecar", "won"]) == 8

    def test_takes_the_shortest_prefix_that_reaches_the_floor(self):
        pairs = [(["a", "b", "c"], ["d", "e", "f"])] * 10
        taken = enough_pairs(pairs, min_words=20)
        assert paragraph_words(taken) >= 20
        assert paragraph_words(taken[:-1]) < 20

    def test_stops_at_the_pair_cap_even_below_the_floor(self):
        pairs = [(["a", "b", "c"], ["d", "e", "f"])] * 10
        assert len(enough_pairs(pairs, min_words=200, max_pairs=3)) == 3

    def test_takes_nothing_from_nothing(self):
        assert enough_pairs([], min_words=100) == []


class TestOrdering:
    def test_links_are_counted_over_reading_order_not_pair_order(self):
        """Pair k is read twice, at position k and at its mirror."""
        pairs = [(["rain", "fell"], ["gone", "far"]),
                 (["rain", "rose"], ["rose", "gone"])]
        # Read in order: rain fell | rain rose | rose gone | gone far — three
        # links, none of which is visible in the order the pairs are listed.
        assert adjacent_links(pairs) == 3

    def test_ordering_never_loses_material(self):
        pairs = [(["a", "b"], ["c", "d"]), (["e", "f"], ["g", "h"]),
                 (["b", "i"], ["j", "k"])]
        assert sorted(order_pairs(pairs)) == sorted(
            [[list(l), list(r)] and (list(l), list(r)) for l, r in pairs])

    def test_ordering_does_not_lower_the_score(self):
        pairs = [(["rain", "fell"], ["hard", "wind"]),
                 (["dust", "sat"], ["cold", "iron"]),
                 (["wind", "rose"], ["rain", "again"])]
        assert adjacent_links(order_pairs(pairs)) >= adjacent_links(pairs)

    def test_reordering_keeps_the_mirror(self):
        """Any order of pairs assembles; the test is that render still asserts."""
        pairs = [(["draw", "no"], ["on", "ward"]),
                 (["step", "on"], ["no", "pets"])]
        assert render(order_pairs(pairs))
