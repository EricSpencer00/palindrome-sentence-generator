from __future__ import annotations

import unittest

from experiments.incumbent_568_boundary_event_residual_20260923 import build_payload


class BoundaryEventResidualAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = build_payload()

    def test_reconstructs_visible_602_letter_nonexact_child(self) -> None:
        candidate = self.payload["candidate"]
        self.assertEqual(candidate["letters"], 602)
        self.assertEqual(candidate["growth_over_parent"], 34)
        self.assertEqual(candidate["exact_audit"]["independent_outside_in_exact"], False)
        self.assertEqual(candidate["exact_audit"]["project_validator_exact"], False)
        self.assertEqual(candidate["local_residual"]["matched_prefix_letters"], 4)
        self.assertEqual(candidate["local_residual"]["first_mismatch"], {
            "cursor": 4,
            "left": "c",
            "required": "r",
        })

    def test_records_historical_phrase_collision_and_dead_repair(self) -> None:
        self.assertEqual(self.payload["novelty_preflight"]["status"], "collision")
        self.assertTrue(any(
            "endpoint_seeded_scene_inward_20260920.py" in line
            for line in self.payload["novelty_preflight"]["hits"]
        ))
        repair = self.payload["repair_attempt"]["lane_report"]
        self.assertEqual(repair["complete_grammatical_parses"], 0)
        self.assertEqual(repair["dead_residual_after_det"], "iawaronnretnaladeirracnora")

    def test_parent_provenance_is_the_pinned_568_artifact(self) -> None:
        self.assertEqual(self.payload["parent"]["letters"], 568)
        self.assertEqual(
            self.payload["parent"]["sha256_normalized"],
            "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380",
        )
        self.assertFalse(self.payload["admission"]["readability_certified"])

    def test_parallel_lane_reports_remain_explicitly_separate(self) -> None:
        lanes = self.payload["parallel_lane_reports"]
        self.assertEqual(len(lanes), 2)
        self.assertTrue(all(lane["exact_closures"] == 0 for lane in lanes))
        self.assertTrue(all("not independently rerun" in lane["source"] for lane in lanes))


if __name__ == "__main__":
    unittest.main()
