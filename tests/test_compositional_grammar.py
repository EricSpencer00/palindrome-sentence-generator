import unittest
from llm_palindrome.block_seams import Piece, Seam, grammar_complete, paragraph_gates
from experiments.block_inventory_coverage_20261009 import parse
from experiments.expanded_grammar_seams_20261009 import grammar
from llm_palindrome.typed_constituents import TypedGrammar,default_grammar,words


class CompositionalRegressionTests(unittest.TestCase):
    def test_cross_source_grammar_is_not_source_identity(self):
        pieces=(Piece('source-A',87,'Liam sees'),Piece('source-B',12,'mail.'))
        s=Seam(pieces,())
        self.assertTrue(s.exact())
        self.assertTrue(grammar_complete(pieces,{})[0])
        self.assertTrue(paragraph_gates(s,{})['sentence_grammar_complete'])

    def test_rider_cross_source_control(self):
        pieces=(Piece('outer-A',0,'No rider'),Piece('middle-B',0,'sees'),Piece('outer-C',0,'red iron.'))
        self.assertTrue(grammar_complete(pieces,{})[0])

    def test_first_person_agreement(self):
        self.assertIsNone(parse(('i','is','ill')))
        self.assertIsNone(parse(('i','were','ill')))
        self.assertIsNotNone(parse(('i','am','ill')))
        self.assertIsNotNone(parse(('you','are','ill')))

    def test_article_allomorphy_expanded_paths(self):
        paths=grammar()
        bad=False;good=False
        for p in paths:
            for i,role in enumerate(p['roles'][:-2]):
                if role=='determiner' and p['roles'][i+1]=='modifier' and 'old' in p['slots'][i+1] and 'book' in p['slots'][i+2]:
                    bad|='a' in p['slots'][i]
                    good|='an' in p['slots'][i]
        self.assertFalse(bad)
        self.assertTrue(good)

    def test_composes_new_object_without_stored_sentence(self):
        g=TypedGrammar({'the','pilot','reads','a','note','map'})
        self.assertTrue(g.complete(words('the pilot reads a note')))
        self.assertTrue(g.compatible(words('the pilot reads'),words('a note')))

    def test_source_independent_multiclause_paragraph(self):
        pieces=(Piece('A',12,'Liam reads mail.'),Piece('B',7,'Leon reads a book.'))
        gates=paragraph_gates(Seam(pieces,()),{})
        self.assertTrue(gates['sentence_grammar_complete'])
        self.assertEqual(len(gates['clause_features']),2)
        self.assertFalse(gates['exact_palindrome'])
        self.assertFalse(gates['human_coherence_verified'])
        self.assertEqual(gates['source_lineages'],['A','B'])

    def test_partial_frontier_across_sentence_boundary(self):
        g=default_grammar()
        self.assertTrue(g.paragraph_frontier(words('Liam reads mail Leon'),words('a book'),3))
        self.assertTrue(g.paragraph_frontier(words('No rider'),words('red iron'),3))

    def test_typed_person_number_and_articles(self):
        g=default_grammar()
        for bad in ('i is ill','i were ill','you is ill','they sees mail','Liam reads a old book','Liam reads an new book'):
            self.assertFalse(g.complete(words(bad)),bad)
        for good in ('i am ill','i was ill','you are ill','they see mail','Liam reads an old book','Liam reads a new book'):
            self.assertTrue(g.complete(words(good)),good)

    def test_does_not_parse_across_explicit_sentence_boundary(self):
        self.assertFalse(grammar_complete((Piece('A',0,'Liam.'),Piece('B',0,'sees mail.')),{})[0])


if __name__=='__main__':unittest.main()
