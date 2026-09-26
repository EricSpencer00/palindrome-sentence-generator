"""Regression checks for the anonymized v1 public reproduction archive."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from paper import build_v1_supplement as supplement


class V1SupplementPrivacyTests(unittest.TestCase):
    def test_export_is_anonymous_relinked_and_deterministic(self) -> None:
        root = supplement.ROOT
        original_v3_manifest_bytes = (root / supplement.V3_MANIFEST).read_bytes()
        original_v4_manifest_bytes = (root / supplement.V4_MANIFEST).read_bytes()
        original_qwen_packet_bytes = (root / supplement.QWEN_PACKET).read_bytes()
        original_brown_manifest_bytes = (root / supplement.QWEN_BROWN_SOURCE).read_bytes()
        original_qwen_main_result = (root / supplement.QWEN_MAIN_RESULT).read_bytes()
        original_qwen_followup_result = (root / supplement.QWEN_FOLLOWUP_RESULT).read_bytes()
        raw_v4_manifest = json.loads((root / supplement.V4_MANIFEST).read_text())
        author_host = raw_v4_manifest["provenance"]["manifest_preparation_host"]
        project_commit = raw_v4_manifest["provenance"]["project_git_commit_at_freeze"]
        original_input_hash = supplement.sha256((root / supplement.V4_INPUT).read_bytes())
        original_summary_hash = supplement.sha256((root / supplement.V4_STATIC_SUMMARY).read_bytes())

        with tempfile.TemporaryDirectory(prefix="v1-supplement-test-") as temporary:
            first = Path(temporary) / "first.zip"
            second = Path(temporary) / "second.zip"
            first_result = supplement.build(first)
            second_result = supplement.build(second)

            self.assertEqual(first_result["sha256"], second_result["sha256"])
            self.assertEqual(first_result["privacy_screen"], "passed")
            self.assertEqual(first_result["internal_hash_links"], "passed")
            self.assertEqual(
                supplement.verify_archive(first)["member_count"],
                len(set(supplement.FILES)) + 3,
            )

            with zipfile.ZipFile(first) as archive:
                members = {name: archive.read(name) for name in archive.namelist()}
            all_bytes = b"\n".join(members.values())
            self.assertNotIn(author_host.encode("utf-8"), all_bytes)
            self.assertNotIn(project_commit.encode("utf-8"), all_bytes)
            self.assertNotIn(b"hst-bench", all_bytes)

            for name in (
                supplement.QWEN_MAIN_PROTOCOL,
                supplement.QWEN_FOLLOWUP_PROTOCOL,
                supplement.QWEN_REPORT,
                supplement.QWEN_MAIN_RUNNER,
                supplement.QWEN_FOLLOWUP_RUNNER,
                supplement.QWEN_MAIN_RESULT,
                supplement.QWEN_FOLLOWUP_RESULT,
            ):
                self.assertIn(name, members)

            exported_packet = json.loads(members[supplement.QWEN_PACKET])
            exported_brown = json.loads(members[supplement.QWEN_BROWN_SOURCE])
            export_provenance = json.loads(
                members["supplement/QWEN-EXPORT-PROVENANCE.json"]
            )
            self.assertEqual(len(exported_packet["candidates"]), 52)
            self.assertEqual(
                exported_packet["source_input_sha256"],
                supplement.sha256(members[supplement.V4_INPUT]),
            )
            self.assertFalse(supplement.has_key(exported_brown, "surface"))
            self.assertFalse(supplement.has_key(exported_brown, "text"))
            self.assertEqual(
                export_provenance["historical_input_sha256"]["full_brown_control_manifest"],
                supplement.sha256(original_brown_manifest_bytes),
            )
            self.assertEqual(members[supplement.QWEN_MAIN_RESULT], original_qwen_main_result)
            self.assertEqual(members[supplement.QWEN_FOLLOWUP_RESULT], original_qwen_followup_result)

            exported_v3_manifest = json.loads(members[supplement.V3_MANIFEST])
            exported_v4_manifest = json.loads(members[supplement.V4_MANIFEST])
            self.assertFalse(supplement.contains_forbidden_metadata(exported_v3_manifest))
            self.assertFalse(supplement.contains_forbidden_metadata(exported_v4_manifest))
            self.assertEqual(
                exported_v4_manifest["provenance"]["wordfreq_required_version"],
                raw_v4_manifest["provenance"]["wordfreq_required_version"],
            )
            self.assertEqual(
                json.loads(members[supplement.V4_GPT2])["model"]["revision"],
                "607a30d783dfa663caf39e06633721c8d4cfcd7e",
            )
            input_hash = supplement.sha256(members[supplement.V4_INPUT])
            summary_hash = supplement.sha256(members[supplement.V4_STATIC_SUMMARY])
            exported_report = members[supplement.V4_REPORT].decode("utf-8")
            self.assertIn(input_hash, exported_report)
            self.assertIn(summary_hash, exported_report)
            self.assertNotIn(original_input_hash, exported_report)
            self.assertNotIn(original_summary_hash, exported_report)

            # The source files behind the frozen protocol remain byte-identical.
            for manifest_name, manifest in (
                (supplement.V3_MANIFEST, exported_v3_manifest),
                (supplement.V4_MANIFEST, exported_v4_manifest),
            ):
                for source_key, source_path in manifest["sources"].items():
                    expected = manifest["source_sha256"][source_key]
                    self.assertEqual(
                        supplement.sha256(members[source_path]), expected, source_path
                    )

        # Building an anonymous derivative must not rewrite raw, private records.
        self.assertEqual(
            (root / supplement.V3_MANIFEST).read_bytes(), original_v3_manifest_bytes,
        )
        self.assertEqual(
            (root / supplement.V4_MANIFEST).read_bytes(), original_v4_manifest_bytes,
        )
        self.assertEqual((root / supplement.QWEN_PACKET).read_bytes(), original_qwen_packet_bytes)
        self.assertEqual((root / supplement.QWEN_BROWN_SOURCE).read_bytes(), original_brown_manifest_bytes)


if __name__ == "__main__":
    unittest.main()
