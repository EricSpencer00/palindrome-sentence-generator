import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "audit_incumbent_568_identity_witness_growth_20260923.py"
SPEC = importlib.util.spec_from_file_location("identity_witness_growth", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class IdentityWitnessGrowthAuditTest(unittest.TestCase):
    def test_reconstructs_exact_growth_and_rejects_shortcut(self):
        result = MODULE.build_payload()
        candidate = result["candidate"]
        self.assertEqual(candidate["letters"], 590)
        self.assertEqual(candidate["growth_over_parent"], 22)
        self.assertEqual(
            candidate["normalized_sha256"],
            "9f94e2ea6ab8976495eea3ab93bbc9c3fdebd70949c5de236e61e2ba184b670b",
        )
        self.assertTrue(candidate["independent_validation"]["outside_in_exact"])
        self.assertTrue(candidate["independent_validation"]["project_validator_exact"])
        self.assertTrue(candidate["independent_validation"]["hashes_equal"])
        self.assertEqual(result["status"], "exact_growth_diagnostic_rejected_by_shortcut_audit")
        audit = result["anti_shortcut_audit"]
        self.assertEqual(audit["reflected_right_boundary_coverage"], 1.0)
        self.assertTrue(any(row["self_palindromic_unit"] for row in audit["equation_decomposition"]))
        self.assertFalse(result["readability"]["human_reader_evidence"])


if __name__ == "__main__":
    unittest.main()
