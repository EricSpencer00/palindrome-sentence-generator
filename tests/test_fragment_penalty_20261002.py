import itertools
import unittest

from experiments.fragment_penalty_20261002 import FragmentScorer


class UnitBase:
    def word_delta(self, *args):
        return 1.0


class FragmentPenaltyTests(unittest.TestCase):
    def test_local_difference_matches_full_potential_for_every_insertion(self):
        fragments = [('a', 'b', 'a'), ('b', 'a', 'b'), ('a', 'a', 'b')]
        scorer = FragmentScorer(UnitBase(), fragments, 4)
        for n in range(7):
            for original in itertools.product(('a', 'b'), repeat=n):
                for split in range(n+1):
                    left, right = original[:split], original[split:]
                    for side, growth, word in itertools.product(('L', 'R'), ('append', 'prepend'), ('a', 'b')):
                        half = left if side == 'L' else right
                        changed = half+(word,) if growth == 'append' else (word,)+half
                        new_left, new_right = (changed, right) if side == 'L' else (left, changed)
                        expected = scorer.potential(new_left+new_right)-scorer.potential(original)
                        self.assertEqual(scorer.change(new_left, new_right, side, growth), expected)
                        self.assertEqual(scorer.word_delta(new_left, new_right, side, word, growth), 1-4*expected)

    def test_disappearing_join_is_refunded_and_zero_strength_preserves_baseline(self):
        scorer = FragmentScorer(UnitBase(), [('one', 'poem', 'association')], 4)
        self.assertEqual(scorer.word_delta(('one', 'new'), ('poem', 'association'), 'L', 'new', 'append'), 5)
        zero = FragmentScorer(UnitBase(), [('one', 'poem', 'association')], 0)
        self.assertEqual(zero.word_delta(('one', 'poem'), ('association',), 'L', 'poem', 'append'), 1)


if __name__ == '__main__':
    unittest.main()
