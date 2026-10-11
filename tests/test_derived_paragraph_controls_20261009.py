import unittest

from experiments.derived_paragraph_controls_20261009 import CANDIDATES, check_representation
from experiments.block_boundary_gate_20261009 import boundary_certificate
from llm_palindrome.admission import normalize_letters
from llm_palindrome.typed_constituents import TypedGrammar, words


class DerivedParagraphControlsTests(unittest.TestCase):
    def test_supplied_controls_are_exact_distinct_complete_clauses(self):
        grammar = TypedGrammar(set(words(CANDIDATES[1])))
        for text, count, letters in zip(CANDIDATES, (2, 4), (30, 60)):
            n = normalize_letters(text)
            self.assertEqual(len(n), letters)
            self.assertEqual(n, n[::-1])
            parsed = grammar.text_paragraph(text, 4)
            self.assertEqual(len(parsed), count)
            self.assertEqual(len({clause for clause, _ in parsed}), count)

    def test_ordinary_grammar_restrictions_are_retained(self):
        grammar = TypedGrammar()
        for text in ('No rider see mail.', 'Liam sees a mail.'):
            self.assertIsNone(grammar.text_paragraph(text, 4))

    def test_boundary_gate_and_cross_np_engine_path(self):
        grammar = TypedGrammar(set(words(CANDIDATES[1])))
        self.assertTrue(boundary_certificate(grammar)['compatible_pairs'])
        receipt = check_representation(grammar)
        self.assertEqual(receipt['rendered_terminal'], CANDIDATES[1])
        self.assertEqual(receipt['attempted_actions'], 168)
        self.assertFalse(receipt['truncated'])


if __name__ == '__main__':
    unittest.main()
