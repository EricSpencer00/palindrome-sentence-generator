"""Build an identity-scrubbed evidence archive for double-anonymous review."""

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
from paper.build_release import write_archive


RELEASE = ROOT / "paper/releases/naacl2027-2026-09-11"
STANDARD_MANIFEST = RELEASE / "RELEASE-MANIFEST.json"
STANDARD_EVIDENCE = RELEASE / "evidence.zip"
REVIEW_MANIFEST = RELEASE / "ANONYMOUS-REVIEW-MANIFEST.json"
REVIEW_EVIDENCE = RELEASE / "anonymous-review-evidence.zip"
META = {"MANIFEST-SHA256.json", "RELEASE-MANIFEST.json", "README.txt"}
IDENTITY = re.compile(
    rb"Eric[\s]+Spencer|EricSpencer00|0009-0003-2592-8075|"
    rb"@luc\.edu|@gmail\.com|github\.com/EricSpencer00",
    re.IGNORECASE,
)

REDACTED_LICENSE = """Project source notice for double-anonymous review

The project-specific copyright and license notice is intentionally withheld
from this confidential review archive because it identifies the author. The
complete notice is retained in the archival release and will be restored for
public distribution. This temporary omission does not grant additional rights.
All third-party attribution and license notices remain included.
"""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    raise RuntimeError(
        "anonymous-review release building is disabled with the legacy release. "
        "There is no admissible current paper package to anonymize."
    )
    base = json.loads(STANDARD_MANIFEST.read_text())
    with tempfile.TemporaryDirectory(prefix="palindrome-anonymous-review-") as tmp:
        stage = Path(tmp)
        with zipfile.ZipFile(STANDARD_EVIDENCE) as archive:
            archive.extractall(stage)
        for name in META:
            (stage / name).unlink()

        (stage / "licenses/PROJECT-LICENSE.txt").unlink()
        (stage / "licenses/PROJECT-LICENSE-REDACTED.txt").write_text(REDACTED_LICENSE)

        terms = stage / "TERMS.md"
        terms.write_text(
            terms.read_text().replace(
                "with the copyright notice in `licenses/PROJECT-LICENSE.txt`.",
                "with its identifying copyright notice temporarily withheld in "
                "`licenses/PROJECT-LICENSE-REDACTED.txt` for double-anonymous review.",
            )
        )
        readme = stage / "README.md"
        readme.write_text(
            readme.read_text().replace(
                "hashes of included source plus checkout provenance.",
                "hashes of included source; identifying checkout provenance is withheld.",
            )
        )
        source_snapshot = stage / "SOURCE-SNAPSHOT.json"
        provenance = json.loads(source_snapshot.read_text())
        provenance["git"] = None
        provenance["review_anonymization"] = (
            "Identifying Git revision withheld for double-anonymous review."
        )
        source_snapshot.write_text(json.dumps(provenance, indent=2) + "\n")

        evidence = {
            str(path.relative_to(stage)): path
            for path in stage.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        }
        leaks = [name for name, path in evidence.items() if IDENTITY.search(path.read_bytes())]
        if leaks:
            raise RuntimeError("Identity-bearing files remain: " + ", ".join(leaks))

        manifest = dict(base)
        manifest.update(
            {
                "release_id": "naacl2027-2026-09-11-anonymous-review",
                "evidence_bundle": REVIEW_EVIDENCE.name,
                "evidence_files_sha256": {
                    name: digest(path) for name, path in sorted(evidence.items())
                },
                "review_anonymization": {
                    "identity_scan_passed": True,
                    "git_revision_withheld": True,
                    "project_license_notice_withheld": True,
                    "third_party_notices_retained": True,
                    "scientific_payload_unchanged": True,
                    "public_archival_release": "evidence.zip",
                },
                "deposit_status": (
                    "prepared for the unpublished double-anonymous review preview"
                ),
            }
        )
        REVIEW_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
        write_archive(
            REVIEW_EVIDENCE,
            evidence,
            manifest,
            "Double-anonymous review evidence. Extract and follow README.md.\n",
        )

    print(
        json.dumps(
            {
                "archive": str(REVIEW_EVIDENCE),
                "manifest": str(REVIEW_MANIFEST),
                "evidence_files": len(manifest["evidence_files_sha256"]),
                "archive_sha256": digest(REVIEW_EVIDENCE),
                "manifest_sha256": digest(REVIEW_MANIFEST),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
