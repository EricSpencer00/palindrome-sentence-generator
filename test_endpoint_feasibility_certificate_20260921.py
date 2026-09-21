import unittest

from experiments.endpoint_feasibility_certificate_20260921 import audit, certificate, paired_prefix, run


class EndpointCertificateTests(unittest.TestCase):
    def test_word_boundaries_do_not_have_to_align(self):
        # Synthetic strings exercise segmentation, not reader evidence.
        result = paired_prefix(["ab", "cde"], ["ed", "cba"])
        self.assertEqual(result, {"matched": 5, "mismatch": None, "exhausted_side": "both"})

    def test_actual_right_order_is_required(self):
        self.assertIsNone(paired_prefix(["abcd"], ["dc", "ba"])["mismatch"])
        self.assertEqual(paired_prefix(["abcd"], ["ba", "dc"])["matched"], 0)

    def test_exterior_certificate_matches_full_scan(self):
        for result in run().values():
            if not isinstance(result, dict):
                continue
            self.assertEqual(result["domain_derivations"], 54)
            self.assertEqual(result["exterior_pairs_surviving"], 0)
            self.assertEqual(result["exact"], 0)
            self.assertFalse(result["diagnostic_prose"]["audit"]["pointer_exact"])

    def test_independent_audits_agree(self):
        for text in ["ab c ba", "ab cd", "", "a"]:
            result = audit(text)
            self.assertEqual(result["pointer_exact"], bool(result["letters"]) and
                             result["forward_sha256"] == result["reverse_sha256"])


if __name__ == "__main__":
    unittest.main()
