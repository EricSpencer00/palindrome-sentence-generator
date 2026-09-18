"""Validate the double-anonymous review evidence in a clean extraction."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper.validate_release import run, verify_reports


RELEASE = ROOT / "paper/releases/naacl2027-2026-09-11"
ARCHIVE = RELEASE / "anonymous-review-evidence.zip"
MANIFEST = RELEASE / "ANONYMOUS-REVIEW-MANIFEST.json"
META = {"MANIFEST-SHA256.json", "RELEASE-MANIFEST.json", "README.txt"}
IDENTITY = re.compile(
    rb"Eric[\s]+Spencer|EricSpencer00|0009-0003-2592-8075|"
    rb"@luc\.edu|@gmail\.com|github\.com/EricSpencer00",
    re.IGNORECASE,
)


def main() -> None:
    raise RuntimeError(
        "anonymous-review release validation is disabled with the legacy release. "
        "There is no admissible current paper package to validate."
    )
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["release_id"].endswith("-anonymous-review")
    assert manifest["review_anonymization"]["identity_scan_passed"] is True
    with zipfile.ZipFile(ARCHIVE) as archive:
        names = set(archive.namelist())
        payload = names - META
        assert "licenses/PROJECT-LICENSE.txt" not in payload
        assert "licenses/PROJECT-LICENSE-REDACTED.txt" in payload
        hashes = json.loads(archive.read("MANIFEST-SHA256.json"))
        assert set(hashes) == payload
        assert hashes == manifest["evidence_files_sha256"]
        assert json.loads(archive.read("RELEASE-MANIFEST.json")) == manifest
        for name in payload:
            data = archive.read(name)
            assert hashlib.sha256(data).hexdigest() == hashes[name]
            if IDENTITY.search(data):
                raise RuntimeError(f"Identity-bearing content remains: {name}")

        with tempfile.TemporaryDirectory(prefix="palindrome-review-validate-") as tmp:
            root = Path(tmp)
            archive.extractall(root)
            provenance = json.loads((root / "SOURCE-SNAPSHOT.json").read_text())
            assert provenance["git"] is None
            for program, output in (
                ("verify_structural_draft.py", "structural-evidence.json"),
                ("critique_evidence.py", "critique-evidence.json"),
            ):
                run(root, "paper/" + program, "--output", output)
            run(
                root,
                "-m",
                "experiments.audit_controlled_pos_pruning",
                "data/controlled",
                "--output",
                "controlled-audit.json",
            )
            run(
                root,
                "-m",
                "experiments.audit_mirror_cost",
                "data/mirror-cost/results.json",
                "--output",
                "mirror-cost-audit.json",
                isolated=False,
            )
            run(
                root,
                "-m",
                "experiments.audit_long_form_examples",
                "data/long-form/examples.json",
                "--root",
                ".",
                "--output",
                "long-form-audit.json",
            )
            verify_reports(root)

    print(
        "Passed anonymous review release: identity scan, payload hashes, "
        "clean-extraction audits, and scientific invariants"
    )


if __name__ == "__main__":
    main()
