"""Prepare the exact, versioned files that belong in the Dataverse draft."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import shutil


ROOT = Path(__file__).resolve().parents[1]
RELEASE_ID = "naacl2027-2026-09-11"
RELEASE = ROOT / "paper" / "releases" / RELEASE_ID
OUTPUT = ROOT / "paper" / "out" / "dataverse-upload"
DATASET_IDENTIFIER = "doi:10.7910/DVN/UOTHMD"

UPLOADS = {
    "naacl2027-mirror-cost-paper-2026-09-11.pdf": ROOT / "paper" / "naacl2027.pdf",
    "naacl2027-mirror-cost-source-2026-09-11.zip": RELEASE / "source.zip",
    "naacl2027-mirror-cost-evidence-2026-09-11.zip": RELEASE / "evidence.zip",
    "naacl2027-mirror-cost-release-manifest-2026-09-11.json": RELEASE / "RELEASE-MANIFEST.json",
    "naacl2027-mirror-cost-anonymous-review-evidence-2026-09-11.zip": (
        RELEASE / "anonymous-review-evidence.zip"
    ),
    "naacl2027-mirror-cost-anonymous-review-manifest-2026-09-11.json": (
        RELEASE / "ANONYMOUS-REVIEW-MANIFEST.json"
    ),
}

METADATA = {
    "title": (
        "Measuring Reversal Cost in English for Exact Palindrome Search: "
        "Paper and Reproducibility Data"
    ),
    "description": (
        "Reproducibility artifacts for the NAACL 2027 submission Measuring "
        "Reversal Cost in English for Exact Palindrome Search. The release "
        "contains the double-anonymous manuscript PDF and source; 900 frozen "
        "WikiText-2 spans with derived segmentations and model scores across "
        "36 experimental cells; controlled POS-shape search trials; exact "
        "long-form outputs; independent audit programs and reports; pinned "
        "model, corpus, vocabulary, and software provenance; licenses; "
        "manifests; and rerun instructions. The files distinguish structural "
        "validity and search yield from human language quality."
    ),
    "subject": "Computer and Information Science",
    "keywords": [
        "palindromes",
        "natural language processing",
        "language modeling",
        "constrained generation",
        "reproducibility",
        "reversal cost",
        "exact search",
    ],
}


def digests(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "bytes": len(data),
        "md5": hashlib.md5(data).hexdigest(),  # Dataverse displays MD5.
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def main() -> None:
    raise RuntimeError(
        "Dataverse upload preparation is disabled: it packages quarantined legacy "
        "claims, and no current-evidence paper release exists."
    )
    release_manifest = json.loads((RELEASE / "RELEASE-MANIFEST.json").read_text())
    if release_manifest["release_id"] != RELEASE_ID:
        raise RuntimeError("Release directory and manifest disagree")
    if release_manifest["title"] not in METADATA["title"]:
        raise RuntimeError("Paper and deposit titles disagree")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest_name = "naacl2027-mirror-cost-upload-manifest-2026-09-11-v2.json"
    expected = set(UPLOADS) | {
        "naacl2027-mirror-cost-upload-manifest-2026-09-11.json",
        manifest_name,
    }
    for path in OUTPUT.iterdir():
        if path.is_file() and path.name not in expected:
            raise RuntimeError(f"Unexpected file in upload directory: {path.name}")

    records = []
    for remote_name, source in UPLOADS.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = OUTPUT / remote_name
        shutil.copyfile(source, destination)
        records.append(
            {
                "remote_name": remote_name,
                "source": str(source.relative_to(ROOT)),
                **digests(destination),
            }
        )

    manifest = {
        "dataset_identifier": DATASET_IDENTIFIER,
        "dataset_state": "unpublished draft",
        "release_id": RELEASE_ID,
        "metadata": METADATA,
        "files": records,
        "notes": (
            "This manifest covers the six payload files, including the "
            "identity-scrubbed review archive. It is intentionally not self-hashed."
        ),
    }
    manifest_path = OUTPUT / manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "output": str(OUTPUT),
        "files": [*records, {"remote_name": manifest_path.name, **digests(manifest_path)}],
    }, indent=2))


if __name__ == "__main__":
    main()
