from __future__ import annotations

import unittest

from experiments.audit_incumbent_568_shared_event_chain_residual_20260923 import build_payload


class SharedEventChainResidualTest(unittest.TestCase):
    def test_reconstructs_and_rejects_the_594_letter_diagnostic(self) -> None:
        payload = build_payload()
        self.assertEqual(payload["parent"]["letters"], 568)
        self.assertEqual(
            payload["parent"]["sha256_normalized"],
            "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380",
        )
        self.assertEqual(payload["operator"]["normalized_spans"], [[48, 91], [477, 520]])
        self.assertEqual(payload["operator"]["raw_spans"], [[63, 119], [662, 721]])
        self.assertEqual(payload["candidate"]["letters"], 594)
        self.assertEqual(payload["candidate"]["growth_over_parent"], 26)
        self.assertEqual(
            payload["candidate"]["normalized_sha256"],
            "9b961f4c1a8fa61008541be9cf37066a8bd321752100c5cf138f970fc29c4f86",
        )
        self.assertEqual(payload["candidate"]["local_equation"]["matched_prefix_letters"], 3)
        self.assertEqual(
            payload["candidate"]["local_equation"]["first_mismatch"],
            {"cursor": 3, "left": "a", "required": "e"},
        )
        self.assertEqual(payload["candidate"]["audit"]["first_mismatch"], [51, "a", 542, "e"])
        self.assertFalse(payload["candidate"]["audit"]["independent_outside_in_exact"])
        self.assertFalse(payload["candidate"]["audit"]["project_validator_exact"])
        self.assertFalse(payload["admission"]["admitted"])
        self.assertEqual(payload["novelty_preflight"]["status"], "no_literal_hits")


if __name__ == "__main__":
    unittest.main()
