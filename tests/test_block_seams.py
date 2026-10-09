import itertools
import unittest
from llm_palindrome.block_seams import Piece,Seam,grammar_complete,paragraph_gates
from llm_palindrome.admission import normalize_letters
from experiments.grammar_block_seams_20261009 import BANK,run


class SeamTests(unittest.TestCase):
    def test_exhaustive_invariant_matches_global_reversal(self):
        strings=['']+[''.join(x) for n in range(1,5) for x in itertools.product('ab',repeat=n)]
        for a,b in itertools.product(strings,repeat=2):
            s=Seam((Piece('toy',0,a),),(Piece('toy2',0,b),))
            self.assertEqual(s.exact(),bool(a+b) and a+b==(a+b)[::-1])
            for side,c in itertools.product(('left','right'),('a','b','ab','ba')):
                child=s.add(side,Piece('toy3',0,c))
                a2,b2=(a+c,b) if side=='left' else (a,c+b)
                x,y=a2,b2[::-1]
                viable=x.startswith(y) or y.startswith(x)
                self.assertEqual(child is not None,viable)

    def test_unequal_partial_debt_and_failure(self):
        s=Seam().add('left',Piece('carry-gem',0,'Tara'))
        s=s.add('right',Piece('weigh-gem',3,'carat.'))
        self.assertEqual((s.debt()['owner'],s.debt()['residual']),('right','c'))
        s=s.add('left',Piece('carry-gem',1,'carries'))
        self.assertEqual(s.debt()['residual'],'arries')
        s=s.add('right',Piece('weigh-gem',2,'a'))
        self.assertEqual(s.debt()['residual'],'rries')
        self.assertIsNone(s.add('right',Piece('weigh-gem',1,'weighs')))
        self.assertFalse(paragraph_gates(s,BANK)['sentence_grammar_complete'])

    def test_complete_grammar_is_independent_of_seam_and_topic(self):
        pieces=tuple(Piece(sid,i,t) for sid in ('carry-gem','weigh-gem') for i,t in enumerate(BANK[sid]['parts']))
        state=Seam(pieces,())
        gates=paragraph_gates(state,BANK)
        self.assertTrue(gates['sentence_grammar_complete'])
        self.assertTrue(gates['discourse_topic_linkage'])
        self.assertFalse(gates['exact_palindrome'])
        self.assertFalse(gates['human_coherence_verified'])
        self.assertFalse(grammar_complete((Piece('carry-gem',0,'Nora'),),BANK)[0])

    def test_rejects_palindromic_block_concatenation_and_duplicates(self):
        bank={'one':dict(parts=['aba.'],entities=['x']), 'two':dict(parts=['aba.'],entities=['x'])}
        s=Seam((Piece('one',0,'aba.'),),(Piece('two',0,'aba.'),))
        self.assertTrue(s.exact());self.assertFalse(paragraph_gates(s,bank)['independent_nonpalindromic_blocks'])
        self.assertTrue(paragraph_gates(s,bank)['experiment_options_pass'])
        self.assertFalse(paragraph_gates(s,bank,require_nonpalindromic_blocks=True)['experiment_options_pass'])
        s=Seam((Piece('one',0,'aba.'),),(Piece('one',0,'aba.'),))
        self.assertFalse(paragraph_gates(s,bank)['no_duplicated_sentences'])

    def test_finite_chart_preserves_failures_without_claiming_success(self):
        r=run();self.assertIn(r['status'],{'state_cap','finite_chart_exhausted'})
        self.assertTrue(r['failed_joins']);self.assertEqual(r['constructions'],[])
        self.assertTrue(any(p['debt']['viable'] for p in r['endpoint_pairs']))


if __name__=='__main__':unittest.main()
