import unittest
from experiments.block_seam_comparison_20261009 import seeded, WordAdditiveScorer
from llm_palindrome.search import WordTries, beam_search


class ComparisonHarnessTests(unittest.TestCase):
    def test_equivalent_seeded_existing_baseline(self):
        s=seeded('No rider','red iron')
        self.assertEqual((s.overhang,s.side),('', 'L'))
        result=beam_search(WordTries(['sees']),WordAdditiveScorer(['sees']),
                           initial_state=s,min_letters=18,max_steps=2,diversity=0)
        self.assertEqual(' '.join(result),'no rider sees red iron')

    def test_right_anchored_seed_debt_orientation(self):
        s=seeded('', 'the map')
        self.assertEqual((s.overhang,s.side),('pameht','R'))

    def test_phrase_score_equals_wordwise_left(self):
        scorer=WordAdditiveScorer(['a','pot','holds'])
        phrase=scorer.word_delta(('holds','a pot'),(), 'L','a pot','append')
        singles=(scorer.word_delta(('holds','a'),(),'L','a','append')+
                 scorer.word_delta(('holds','a','pot'),(),'L','pot','append'))
        self.assertAlmostEqual(phrase,singles)

    def test_phrase_score_equals_wordwise_right(self):
        scorer=WordAdditiveScorer(['a','pot','holds'])
        phrase=scorer.word_delta((),('a pot','holds'),'R','a pot','prepend')
        singles=(scorer.word_delta((),('pot','holds'),'R','pot','prepend')+
                 scorer.word_delta((),('a','pot','holds'),'R','a','prepend'))
        self.assertAlmostEqual(phrase,singles)


if __name__=='__main__':unittest.main()
