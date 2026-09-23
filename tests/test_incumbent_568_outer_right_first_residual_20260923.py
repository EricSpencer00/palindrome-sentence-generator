from __future__ import annotations

import unittest

from experiments.incumbent_568_outer_right_first_residual_20260923 import build_payload


class OuterRightFirstResidualAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = build_payload()

    def test_candidate_is_visible_but_not_exact(self) -> None:
        candidate = self.payload["candidate"]
        self.assertEqual(candidate["letters"], 577)
        self.assertEqual(candidate["growth_over_parent"], 9)
        self.assertEqual(candidate["audit"]["independent_outside_in_exact"], False)
        self.assertEqual(candidate["audit"]["project_validator_exact"], False)
        self.assertEqual(candidate["local_equation"]["matched_prefix_letters"], 4)
        self.assertEqual(candidate["local_equation"]["first_mismatch"], {
            "cursor": 4,
            "left": "f",
            "required": "h",
        })
        self.assertEqual(candidate["local_equation"]["left_letters"], 52)
        self.assertEqual(candidate["local_equation"]["right_letters"], 53)

    def test_repeated_event_and_phrase_family_collision_are_rejected(self) -> None:
        repeated = self.payload["candidate"]["same_turn_repeated_event"]
        self.assertEqual(repeated["occurrences_in_insertions"], 2)
        self.assertEqual(self.payload["novelty_preflight"]["status"], "family_collision")
        self.assertTrue(any("opens the garden gate" in hit for hit in self.payload["novelty_preflight"]["hits"]))
        self.assertFalse(self.payload["admission"]["readability_certified"])


if __name__ == "__main__":
    unittest.main()
