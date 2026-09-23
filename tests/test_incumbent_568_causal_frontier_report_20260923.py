from __future__ import annotations

import unittest

from experiments.audit_incumbent_568_causal_frontier_report_20260923 import build_payload


class CausalFrontierReportTest(unittest.TestCase):
    def test_pinned_seam_and_reported_frontier(self) -> None:
        payload = build_payload()
        self.assertEqual(payload["parent"]["letters"], 568)
        self.assertEqual(
            payload["parent"]["sha256_normalized"],
            "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380",
        )
        self.assertEqual(payload["operator"]["source_letters_each"], 31)
        self.assertEqual(payload["reported_search"]["furthest_prefix_letters"], 1)
        self.assertEqual(
            payload["reported_search"]["verified_left_residual_after_prefix"],
            "ftermararescuedmaramararescuedwater",
        )
        self.assertFalse(payload["candidate"]["emitted"])


if __name__ == "__main__":
    unittest.main()
