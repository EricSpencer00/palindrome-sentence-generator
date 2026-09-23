from __future__ import annotations

import unittest

from experiments.audit_incumbent_568_wide_scene_residual_20260923 import build_payload


class WideSceneResidualAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = build_payload()

    def test_reconstructs_568_nonexact_scene(self) -> None:
        candidate = self.payload["candidate"]
        self.assertEqual(candidate["letters"], 568)
        self.assertEqual(candidate["growth_over_parent"], 0)
        self.assertEqual(candidate["normalized_sha256"], "88f122fa227833415684eebc68f31cf25edac55a316a86ad747a97fd54669c4e")
        self.assertFalse(candidate["audit"]["independent_outside_in_exact"])
        self.assertFalse(candidate["audit"]["project_validator_exact"])
        self.assertEqual(candidate["audit"]["first_mismatch"], [23, "i", 544, "e"])
        self.assertEqual(candidate["local_equation"]["left_letters"], 128)
        self.assertEqual(candidate["local_equation"]["right_letters"], 128)
        self.assertEqual(candidate["local_equation"]["matched_prefix_letters"], 3)

    def test_testimony_lane_is_report_only_obstruction(self) -> None:
        report = self.payload["parallel_lane_obstructions"][0]
        self.assertEqual(report["normalized_spans"], [[148, 204], [364, 420]])
        self.assertEqual(report["complete_dialogue_states"], 18816)
        self.assertEqual(report["exact_closures"], 0)
        self.assertIsNone(report["candidate_hash"])
        self.assertFalse(self.payload["admission"]["reader_evidence"])

    def test_pinned_parent_and_preflight(self) -> None:
        self.assertEqual(self.payload["parent"]["letters"], 568)
        self.assertEqual(self.payload["parent"]["sha256_normalized"], "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380")
        self.assertEqual(self.payload["novelty_preflight"]["status"], "no_literal_hits")


if __name__ == "__main__":
    unittest.main()
