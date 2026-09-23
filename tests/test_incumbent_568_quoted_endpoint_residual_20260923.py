from __future__ import annotations

import unittest

from experiments.audit_incumbent_568_quoted_endpoint_residual_20260923 import build_payload


class QuotedEndpointResidualAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = build_payload()

    def test_pinned_parent_and_live_endpoint_residual(self) -> None:
        self.assertEqual(self.payload["parent"]["letters"], 568)
        self.assertEqual(
            self.payload["parent"]["sha256_normalized"],
            "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380",
        )
        probe = self.payload["endpoint_probe"]
        self.assertEqual(probe["matched_letters"], 5)
        self.assertEqual(probe["residual_after_prefix"], "swardraila")
        self.assertEqual(probe["right_word_boundaries"], [1, 5, 10, 11])
        self.assertEqual(probe["reflected_right_word_boundaries"], [4, 5, 10, 14])
        self.assertEqual(probe["left_word_boundaries"], [4, 5, 10, 14])
        self.assertEqual(probe["boundary_alignment_fraction"], 1.0)
        self.assertEqual(probe["novelty_preflight"]["status"], "no_literal_hits")

    def test_no_candidate_was_emitted(self) -> None:
        self.assertFalse(self.payload["candidate"]["emitted"])
        self.assertIsNone(self.payload["candidate"]["rendered"])
        self.assertFalse(self.payload["admission"]["reader_evidence"])


if __name__ == "__main__":
    unittest.main()
