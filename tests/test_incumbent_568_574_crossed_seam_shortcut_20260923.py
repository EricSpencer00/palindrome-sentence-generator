from __future__ import annotations

import unittest

from experiments.audit_incumbent_568_574_crossed_seam_shortcut_20260923 import build_payload


class ExactNearWordwiseMirrorAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = build_payload()

    def test_child_is_exact_but_only_six_letters_longer(self) -> None:
        candidate = self.payload["candidate"]
        self.assertEqual(candidate["letters"], 574)
        self.assertEqual(candidate["growth_over_parent"], 6)
        self.assertTrue(candidate["audit"]["independent_outside_in_exact"])
        self.assertTrue(candidate["audit"]["project_validator_exact"])
        self.assertEqual(candidate["normalized_sha256"], "4c2d2eec9a2bcf0490d760bbe411c58e4f40f323e7e373ddda31dd4800d9d25e")

    def test_exact_equation_is_factored_by_reflected_word_boundaries(self) -> None:
        audit = self.payload["anti_shortcut_audit"]
        self.assertFalse(audit["whole_token_sequence_is_reverse"])
        self.assertEqual(audit["reflected_right_boundary_coverage"], 1.0)
        self.assertGreaterEqual(len(audit["aligned_semordnilap_word_pairs"]), 6)
        self.assertTrue(any(unit["self_palindromic_unit"] for unit in audit["exact_local_decomposition"]))
        self.assertTrue(all(unit["left_equals_reverse_of_right_unit"] for unit in audit["exact_local_decomposition"]))
        self.assertFalse(self.payload["admission"]["admitted"])

    def test_no_literal_clause_collision_and_no_reader_claim(self) -> None:
        self.assertEqual(self.payload["novelty_preflight"]["status"], "no_literal_hits")
        self.assertFalse(self.payload["readability_review"]["human_reader_evidence"])
        self.assertFalse(self.payload["readability_review"]["certified_readable"])


if __name__ == "__main__":
    unittest.main()
