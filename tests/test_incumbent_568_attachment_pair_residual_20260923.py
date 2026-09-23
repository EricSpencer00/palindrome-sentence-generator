from __future__ import annotations

import unittest

from experiments.audit_incumbent_568_attachment_pair_residual_20260923 import build_payload


class AttachmentPairResidualAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = build_payload()

    def test_equal_growth_pair_fails_on_first_character(self) -> None:
        candidate = self.payload["candidate"]
        equation = candidate["local_equation"]
        self.assertEqual(candidate["letters"], 584)
        self.assertEqual(candidate["growth_over_parent"], 16)
        self.assertEqual(equation["left_letters"], 24)
        self.assertEqual(equation["right_letters"], 24)
        self.assertEqual(equation["matched_prefix_letters"], 0)
        self.assertEqual(equation["first_mismatch"], {"cursor": 0, "left": "m", "required": "k"})
        self.assertEqual(candidate["audit"]["first_mismatch"], [204, "m", 379, "k"])
        self.assertFalse(candidate["audit"]["independent_outside_in_exact"])
        self.assertFalse(candidate["audit"]["project_validator_exact"])
        self.assertNotEqual(candidate["audit"]["sha256_forward"], candidate["audit"]["sha256_reverse"])

    def test_parent_and_phrase_preflight(self) -> None:
        self.assertEqual(self.payload["parent"]["letters"], 568)
        self.assertEqual(
            self.payload["parent"]["sha256_normalized"],
            "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380",
        )
        self.assertEqual(self.payload["novelty_preflight"]["status"], "no_literal_hits")
        self.assertFalse(self.payload["admission"]["reader_evidence"])


if __name__ == "__main__":
    unittest.main()
