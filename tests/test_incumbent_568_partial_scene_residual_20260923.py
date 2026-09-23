from __future__ import annotations

import unittest

from experiments.audit_incumbent_568_partial_scene_residual_20260923 import build_payload


class PartialSceneResidualAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = build_payload()

    def test_reconstructs_nonexact_576_candidate(self) -> None:
        candidate = self.payload["candidate"]
        self.assertEqual(candidate["letters"], 576)
        self.assertEqual(candidate["growth_over_parent"], 8)
        self.assertFalse(candidate["audit"]["independent_outside_in_exact"])
        self.assertFalse(candidate["audit"]["project_validator_exact"])
        self.assertEqual(candidate["local_equation"]["left_letters"], 28)
        self.assertEqual(candidate["local_equation"]["right_letters"], 32)
        self.assertEqual(candidate["local_equation"]["matched_prefix_letters"], 2)
        self.assertEqual(candidate["local_equation"]["first_mismatch"], {
            "cursor": 2,
            "left": "m",
            "required": "s",
        })

    def test_parallel_reports_are_obstructions_not_candidates(self) -> None:
        reports = self.payload["parallel_lane_obstructions"]
        self.assertEqual(len(reports), 3)
        self.assertTrue(all(item["candidate_hash"] is None for item in reports))
        self.assertEqual(reports[0]["exact_closures"], 0)
        self.assertEqual(reports[1]["normalized_spans"], [[73, 100], [468, 495]])
        self.assertEqual(reports[1]["fresh_structures"], 12000)
        self.assertEqual(reports[2]["obstruction_after_forced_suffix"], "eht")

    def test_parent_and_phrase_preflight(self) -> None:
        self.assertEqual(self.payload["parent"]["letters"], 568)
        self.assertEqual(self.payload["parent"]["sha256_normalized"], "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380")
        self.assertEqual(self.payload["novelty_preflight"]["status"], "no_literal_hits")
        self.assertFalse(self.payload["admission"]["reader_evidence"])


if __name__ == "__main__":
    unittest.main()
