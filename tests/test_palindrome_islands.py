import itertools,unittest
from llm_palindrome.palindrome_islands import full_island,partial_island
from llm_palindrome.block_seams import Seam,Piece
from llm_palindrome.admission import normalize_letters


class IslandTests(unittest.TestCase):
    def test_unequal_full_blocks_and_short_seam(self):
        # Pure toy strings, no grammar or linguistic claim.
        s=Seam((Piece('P',0,'aba'),),(Piece('Q',0,'ababa'),))
        self.assertEqual(s.debt()['residual'],'ba')
        s=s.add('left',Piece('seam',0,'b'))
        self.assertTrue(s.exact());self.assertEqual(s.text(),'aba b ababa')
        self.assertIsNone(Seam((Piece('P',0,'aba'),),()).add('right',Piece('Q',0,'aca')))

    def test_every_toy_span_obeys_edge_core_identity(self):
        for n in range(2,5):
            for letters in itertools.product('ab',repeat=n):
                h=''.join(letters);parent=h+h[::-1]
                for start in range(len(parent)):
                    for end in range(start+1,len(parent)+1):
                        try:r=partial_island('toy',parent,(start,end),'synthetic-test','not language',['toy completion'],min_core=2)
                        except ValueError:continue
                        self.assertEqual(r['left_edge']+r['core']+r['right_edge'],parent[start:end])
                        self.assertEqual(r['core'],r['core'][::-1])
                        reconstructed=r['mirror_completion_left']+parent[start:end]+r['mirror_completion_right']
                        self.assertEqual(reconstructed,reconstructed[::-1])

    def test_real_control_partial_and_grammar_debt(self):
        parent='Step on no pets.'
        r=partial_island('step-prefix',parent,(0,len('Step on no')),'known-control',
                         'imperative with incomplete quantified NP',['noun after no; source completion pets'])
        self.assertEqual((r['core'],r['left_edge'],r['mirror_completion_right']),('onno','step','pets'))
        self.assertFalse(r['human_readability_verified'])
        with self.assertRaises(ValueError):partial_island('bad','ordinary text',(0,5),'test','none',['none'])
        with self.assertRaises(ValueError):partial_island('bad',parent,(0,4),'test','fragment',['none'])

    def test_full_island_is_not_automatic_composition(self):
        full_island('control','Dennis sinned.','known-control','subject + past intransitive')
        with self.assertRaises(ValueError):full_island('bad','The cat sleeps.','test','ordinary sentence')


if __name__=='__main__':unittest.main()
