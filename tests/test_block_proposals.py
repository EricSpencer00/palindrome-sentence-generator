import unittest
from llm_palindrome.palindrome_islands import lexical_atom
from llm_palindrome.block_proposals import Proposal,UnconnectedTransformer,checked_beam,context_for
from llm_palindrome.block_seams import Seam,Piece


class ProposalTests(unittest.TestCase):
    def test_real_word_may_be_all_debt(self):
        r=lexical_atom('carry','carries','synthetic lexical test',{'carries'},'finite-3sg',['subject and patient'])
        self.assertEqual(r['core'],'');self.assertEqual(r['left_edge'],'carries')
        with self.assertRaises(ValueError):lexical_atom('bad','nonsenseword','test',{'carries'},'verb',['patient'])

    def test_unconnected_interface_cannot_claim_inference(self):
        with self.assertRaises(RuntimeError):checked_beam(Seam(),UnconnectedTransformer(),lambda s,p:True,
                                                          meaning='toy',grammar_slots=['toy'])

    def test_decoded_feasibility_multiple_paths_and_context(self):
        # Synthetic strings validate the adapter only; not model or language output.
        class Fake:
            connected=True;identity='synthetic-test-proposer'
            def propose(self,context):
                self.context=context
                return [Proposal('right','ababa','good',1,'synthetic'),Proposal('right','aba','good2',.5,'synthetic'),
                        Proposal('right','aca','bad',9,'synthetic'),Proposal('left','Été','unicode',10,'synthetic')]
        f=Fake();st=Seam((Piece('P',0,'aba'),),())
        r=checked_beam(st,f,lambda s,p:True,meaning='not a story',grammar_slots=lambda s:['toy'],width=2)
        self.assertEqual(len(r['states']),2)
        self.assertEqual(f.context['left_surface'],'aba');self.assertEqual(f.context['right_surface'],'')
        self.assertEqual(f.context['intended_meaning'],'not a story')
        self.assertIn(('bad','letter infeasible'),r['rejections']);self.assertIn(('unicode','letter infeasible'),r['rejections'])
        self.assertEqual(r['model_identity'],'synthetic-test-proposer')
        self.assertFalse(r['human_readability_verified'])


if __name__=='__main__':unittest.main()
