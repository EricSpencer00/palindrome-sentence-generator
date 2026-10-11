import unittest
from unittest.mock import patch

from experiments.licensed_control_inventory_v2_20261009 import build_inventory
from experiments.derived_paragraph_controls_20261009 import CANDIDATES, UNITS
from llm_palindrome.admission import normalize_letters
from llm_palindrome.typed_constituents import words


class LicensedInventoryTests(unittest.TestCase):
    def test_components_are_source_bound_and_controls_remain_excluded(self):
        inventory, grammar, _ = build_inventory()
        records = {r['id']: r for r in inventory['source_records']}
        self.assertTrue(set(UNITS) <= set(inventory['blocks']))
        for block in inventory['added_blocks']:
            sources = [r for r in inventory['unit_provenance'][block] if 'source_id' in r]
            self.assertTrue(sources)
            for source in sources:
                start, stop = source['word_span']
                self.assertEqual(tuple(records[source['source_id']]['words'][start:stop]), words(block))
        for text in CANDIDATES + ('No rider sees red iron.', 'Leon sees Noel.'):
            self.assertIn(normalize_letters(text), inventory['excluded_normalized'])
        self.assertEqual(len(grammar.text_paragraph(CANDIDATES[1], 4)), 4)
        self.assertIsNone(grammar.text_paragraph('No rider see mail.', 4))
        self.assertIsNone(grammar.text_paragraph('Liam sees a mail.', 4))

    def test_supplied_success_texts_do_not_determine_inventory(self):
        original, _, _ = build_inventory()
        with patch('experiments.licensed_control_inventory_v2_20261009.CANDIDATES', ()):
            independent, _, _ = build_inventory()
        for key in ('words', 'blocks', 'unit_provenance', 'source_records'):
            self.assertEqual(original[key], independent[key])


if __name__ == '__main__':
    unittest.main()
