import unittest
from experiments.typed_partial_bank_20261009 import Cursor,choices,advance,run,FRAMES,SUBJECTS,DETERMINERS,YieldTrie
from llm_palindrome.admission import normalize_letters


class TypedBankTests(unittest.TestCase):
    def test_prefix_index_keeps_shorter_and_longer_yields(self):
        index=YieldTrie(('a','airs','airship','boat'),'left')
        self.assertEqual(set(index.matching('airs')),{'a','airs','airship'})
        self.assertEqual(set(index.matching('air')),{'a','airs','airship'})
        self.assertEqual(set(index.matching('ax')),{'a'})
        self.assertEqual(YieldTrie(('permit',),'right').matching('tim'),['permit'])

    def test_reverse_slots_preserve_transitive_valency(self):
        c=advance(Cursor(),'right','permit')
        self.assertEqual(choices(c,'right'),list(DETERMINERS))
        c=advance(c,'right','a')
        self.assertIn('signs',choices(c,'right'))
        self.assertNotIn('reads',choices(c,'right'))
        c=advance(c,'right','signs');self.assertEqual(choices(c,'right'),list(SUBJECTS))

    def test_conditioned_chart_and_failure_certificates(self):
        r=run(seconds=1,max_states=100)
        self.assertEqual(r['status'],'exhausted')
        self.assertEqual(r['endpoint_suggestions'],[dict(subject='Tim',right_object='permit'),dict(subject='Eva',right_object='wave')])
        self.assertTrue(r['failed_joins']);self.assertFalse(r['exact_constructions'])
        for s in r['residual_conditioned_suggestions']:
            t=normalize_letters(s['compatible_phrase'])
            if s['side']=='right':t=t[::-1]
            self.assertTrue(t.startswith(s['open_residual']) or s['open_residual'].startswith(t))
        self.assertIsNone(r['human_ratings'])
        self.assertFalse(r['options']['shared_entities_required'])
        self.assertFalse(r['options']['require_nonpalindromic_blocks'])
        self.assertFalse(r['options']['require_unique_sentences'])


if __name__=='__main__':unittest.main()
