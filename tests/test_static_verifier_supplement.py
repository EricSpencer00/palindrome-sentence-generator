import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "static_supplement", ROOT / "paper/build_static_verifier_supplement.py"
)
supplement = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(supplement)


class StaticVerifierSupplementTests(unittest.TestCase):
    def test_archive_is_deterministic_hash_checked_and_excludes_brown_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.zip"
            second = Path(tmp) / "second.zip"
            supplement.build_archive(first)
            supplement.build_archive(second)
            manifest = supplement.verify_archive(first)

            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(len(manifest["files_sha256"]), len(supplement.FILES))
            self.assertIn("readme_sha256", manifest)
            self.assertNotIn("runs/static-language-verifier-20260925/input.json",
                             supplement.FILES)


if __name__ == "__main__":
    unittest.main()
