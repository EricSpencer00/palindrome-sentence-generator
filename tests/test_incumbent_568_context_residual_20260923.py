from __future__ import annotations

import unittest

from experiments.audit_incumbent_568_context_residual_20260923 import build_payload


class ContextResidualAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = build_payload()

    def test_reconstructs_nonexact_equal_length_splice(self) -> None:
        candidate = self.payload["candidate"]
        self.assertEqual(candidate["letters"], 634)
        self.assertEqual(candidate["growth_over_parent"], 66)
        self.assertEqual(candidate["normalized_sha256"], "7dc2b275bca05aa6027ded1834f8f2264edb0ea1abd9cc326963f3983f0e9965")
        self.assertFalse(candidate["audit"]["independent_outside_in_exact"])
        self.assertFalse(candidate["audit"]["project_validator_exact"])
        self.assertEqual(candidate["audit"]["first_mismatch"], [54, "i", 579, "u"])
        self.assertEqual(candidate["local_equation"]["left_letters"], 45)
        self.assertEqual(candidate["local_equation"]["right_letters"], 45)
        self.assertEqual(candidate["local_equation"]["matched_prefix_letters"], 4)

    def test_context_traps_and_parallel_obstructions_are_retained(self) -> None:
        self.assertEqual(self.payload["novelty_preflight"]["status"], "no_literal_hits")
        reports = self.payload["parallel_lane_obstructions"]
        self.assertEqual(len(reports), 2)
        self.assertTrue(all(report["candidate_hash"] is None for report in reports))
        self.assertEqual(reports[0]["fresh_units"], 16632)
        self.assertEqual(reports[0]["exact_closures"], 0)
        self.assertEqual(reports[1]["retained_continuation"], "is reviled")
        self.assertFalse(self.payload["admission"]["reader_evidence"])

    def test_parent_is_the_pinned_568_artifact(self) -> None:
        self.assertEqual(self.payload["parent"]["letters"], 568)
        self.assertEqual(self.payload["parent"]["sha256_normalized"], "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380")


if __name__ == "__main__":
    unittest.main()
