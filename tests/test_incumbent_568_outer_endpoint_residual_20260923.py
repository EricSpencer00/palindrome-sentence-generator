from __future__ import annotations

import unittest

from experiments.audit_incumbent_568_outer_endpoint_residual_20260923 import build_payload


class OuterEndpointResidualAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = build_payload()

    def test_reconstructs_long_nonexact_child(self) -> None:
        candidate = self.payload["candidate"]
        self.assertEqual(candidate["letters"], 641)
        self.assertEqual(candidate["growth_over_parent"], 73)
        self.assertEqual(candidate["normalized_sha256"], "ad68f263f7d1ee681ef60a5d1cbcf80734fb0fbfbabf3f7a96b2d825ae40fb4c")
        self.assertFalse(candidate["audit"]["independent_outside_in_exact"])
        self.assertFalse(candidate["audit"]["project_validator_exact"])
        self.assertEqual(candidate["audit"]["first_mismatch"], [3, "i", 637, "e"])
        self.assertEqual(candidate["local_equation"]["left_letters"], 184)
        self.assertEqual(candidate["local_equation"]["right_letters"], 185)
        self.assertEqual(candidate["local_equation"]["matched_prefix_letters"], 3)

    def test_parent_and_provenance_preflight(self) -> None:
        self.assertEqual(self.payload["parent"]["letters"], 568)
        self.assertEqual(self.payload["parent"]["sha256_normalized"], "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380")
        self.assertEqual(self.payload["novelty_preflight"]["status"], "no_literal_hits")
        self.assertFalse(self.payload["admission"]["reader_evidence"])


if __name__ == "__main__":
    unittest.main()
