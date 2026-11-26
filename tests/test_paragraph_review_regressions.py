import unittest
from llm_palindrome.block_seams import Piece,grammar_complete
from llm_palindrome.typed_constituents import default_grammar,words


class ParagraphReviewTests(unittest.TestCase):
    def test_declared_fallback_preserves_punctuation(self):
        bank={'control':{'parts':['Liam sees mail.'],'entities':['Liam','mail']}}
        self.assertFalse(grammar_complete((Piece('A',0,'Liam.'),Piece('B',0,'sees mail.')),bank)[0])

    def test_frontier_respects_combined_clause_budget(self):
        g=default_grammar();two=words('Liam reads mail Leon reads a book')
        self.assertIsNone(g.paragraph(two+two,3))
        self.assertFalse(g.paragraph_frontier(two,two,3))
        self.assertTrue(g.paragraph_frontier(two,two,4))

    def test_runon_is_not_claimed_grammatical_as_written(self):
        g=default_grammar()
        self.assertIsNone(g.text_paragraph('Liam reads mail Leon reads a book.'))
        self.assertIsNotNone(g.text_paragraph('Liam reads mail. Leon reads a book.'))

    def test_ambiguous_read_tense_retained(self):
        g=default_grammar();t=words('I read mail');f=g.clause_features(t,g.complete(t))
        self.assertEqual(set(f['tense_options']),{'present','past'})

    def test_patient_and_location_are_separate(self):
        g=default_grammar();t=words('Liam places a gem inside the case');f=g.clause_features(t,g.complete(t))
        self.assertEqual(f['patient'],['a','gem'])
        self.assertEqual(f['object'],['a','gem'])
        self.assertEqual(f['location'],['the','case'])
        self.assertIn('gem',f['entities']);self.assertIn('case',f['entities'])

    def test_index_allowance_allocated_symmetrically(self):
        from experiments.block_seam_comparison_20261009 import allocate_cells
        self.assertEqual(allocate_cells(60,12,12),4)
        self.assertEqual(allocate_cells(60,0,12),5)
        self.assertEqual(allocate_cells(60,65,12),0)

    def test_paragraph_fixture_matches_proposal(self):
        from experiments.block_seam_comparison_20261009 import paragraph_cells
        cells=paragraph_cells()
        self.assertEqual(len(cells),12)
        self.assertEqual({tuple(x['band']) for x in cells},{(60,119),(120,239)})
        self.assertEqual({x['seed'] for x in cells},{921,922,923})
        self.assertTrue(all(x['left']==x['right']=='' for x in cells))

    def test_provenance_lookup_preserves_order_and_multiplicity(self):
        from experiments.block_seam_comparison_20261009 import contiguous_sources
        records=[{'words':('a','book','a'),'lineage':'one'}]
        self.assertEqual(contiguous_sources('book a',records),['one'])
        self.assertEqual(contiguous_sources('book book',records),[])
        self.assertEqual(contiguous_sources('book a a',records),[])

    def test_rendering_is_explicit_and_preserves_letter_tape(self):
        from experiments.block_seam_comparison_20261009 import render_paragraph,source_records
        from llm_palindrome.admission import normalize_letters
        g=default_grammar();raw='liam reads mail leon reads a book'
        rendered=render_paragraph(g.paragraph(words(raw),4))
        self.assertEqual(rendered,'Liam reads mail. Leon reads a book.')
        self.assertEqual(normalize_letters(raw),normalize_letters(rendered))
        self.assertIsNotNone(g.text_paragraph(rendered,4))
        records=[{'words':('a','book','a'),'lineage':'one','source':'local-row'}]
        entry=source_records('book a',records)[0]
        self.assertEqual(entry['word_span'],[1,3])
        self.assertEqual(entry['source_record'],records[0])

    def test_existing_source_attribution_gaps_are_repaired(self):
        from experiments.block_seam_comparison_20261009 import build_config
        c,_=build_config()
        for unit in ('ate','basil','dennis sin','lisa'):
            self.assertTrue(c['unit_provenance'][unit],unit)
        partial=c['unit_provenance']['dennis sin'][0]['source_record']
        self.assertIn('unfinished lexeme',partial['attribution_type'])


if __name__=='__main__':unittest.main()
