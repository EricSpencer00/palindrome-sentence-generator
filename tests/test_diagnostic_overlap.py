"""Check the manuscript cross-tab against frozen data and reject broken joins."""
from pathlib import Path
import json
import shutil
import tempfile
import unittest

from paper.audit_diagnostic_overlap import ROOT, RUN, audit


class DiagnosticOverlapTests(unittest.TestCase):
    def test_frozen_counts_and_duplicate_collapse(self):
        result = audit()
        self.assertEqual(result["selected_records"], 66)
        self.assertEqual(result["distinct_texts"], 52)
        self.assertEqual(result["record_cross_tab"], {
            "brown_positive__gpt2_positive": 21,
            "brown_positive__gpt2_nonpositive": 19,
            "brown_nonpositive__gpt2_positive": 16,
            "brown_nonpositive__gpt2_nonpositive": 10,
        })
        self.assertEqual(result["distinct_text_cross_tab"], {
            "brown_positive__gpt2_positive": 15,
            "brown_positive__gpt2_nonpositive": 12,
            "brown_nonpositive__gpt2_positive": 16,
            "brown_nonpositive__gpt2_nonpositive": 9,
        })
        self.assertEqual(result["qwen3_candidates_zero_on_both_dimensions"], 52)
        self.assertEqual(result["qwen3_joint_passes_excluding_repeats"]["heldout_brown_prose"], 11)

    def test_rejects_stale_scorer_and_mismatched_judge_text(self):
        files = [f"{RUN}/{name}" for name in (
            "static-verifier-input-aws.json", "brown-v4-aws.json", "gpt2-v4-aws.json",
        )] + ["runs/aws-qwen3-32b-readability-audit-20260926.json"]
        cases = (
            (files[1], lambda d: d.update(input_manifest_sha256="0" * 64), "brown input hash mismatch"),
            (files[3], lambda d: d["responses"][0].update(text_sha256="0" * 64), "Qwen text hash mismatch"),
        )
        for filename, mutate, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                for name in files:
                    (root / name).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ROOT / name, root / name)
                target = root / filename
                data = json.loads(target.read_bytes())
                mutate(data)
                target.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, message):
                    audit(root)


if __name__ == "__main__":
    unittest.main()
