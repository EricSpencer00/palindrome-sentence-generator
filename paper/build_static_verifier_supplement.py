"""Build and validate an anonymized, deterministic paper audit supplement."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "paper/static-verifier-supplement.zip"
STAMP = (2026, 9, 25, 0, 0, 0)
FILES = (
    "experiments/STATIC-LANGUAGE-VERIFIER-20260925.md",
    "experiments/audit_programmatic_readability.py",
    "experiments/infer_deepseek_static_verifier_20260925.py",
    "experiments/score_gpt2_static_verifier_20260925.py",
    "experiments/score_length_stratified_readability.py",
    "experiments/score_week_results_readability.py",
    "experiments/static-verifier-bedrock-requirements.txt",
    "experiments/static-verifier-nuc-requirements.txt",
    "experiments/static_language_verifier_20260925.py",
    "paper/acl.sty",
    "paper/acl_natbib.bst",
    "paper/build_readability_table.py",
    "paper/build_static_verifier_table.py",
    "paper/naacl2027.tex",
    "paper/readability-requirements.txt",
    "paper/readability_table.tex",
    "paper/refs.bib",
    "paper/static_verifier_table.tex",
    "paper/week_results.json",
    "runs/readability-length-stratified-sentence-aware-20260925.json",
    "runs/static-language-verifier-20260925/deepseek-r1-preflight.json",
    "runs/static-language-verifier-20260925/deepseek.json",
    "runs/static-language-verifier-20260925/gpt2.json",
    "runs/static-language-verifier-20260925/report.json",
    "tests/test_static_language_verifier_20260925.py",
)
IDENTITY_MARKERS = (
    re.compile(rb"Eric\s+Spencer", re.IGNORECASE),
    re.compile(rb"EricSpencer00", re.IGNORECASE),
    re.compile(rb"0009-0003-2592-8075", re.IGNORECASE),
    re.compile(rb"/Users/eric/", re.IGNORECASE),
    re.compile(rb"@luc\.edu", re.IGNORECASE),
    re.compile(rb"@gmail\.com", re.IGNORECASE),
    re.compile(rb"github\.com/EricSpencer00", re.IGNORECASE),
)
MANIFEST_NAME = "SUPPLEMENT-MANIFEST.json"
README_NAME = "README.txt"
README = """NAACL 2027 static language-model audit supplement

The source manuscript is paper/naacl2027.tex. Start with
experiments/STATIC-LANGUAGE-VERIFIER-20260925.md for the protocol, sources,
limits, and exact rerun commands. The frozen score reports and generated table
are included. Rebuilding Brown controls requires the NLTK Brown Corpus; raw
Brown passages and GPT-2 weights are not redistributed. GPT-2 can be fetched
from the pinned public model revision. Re-running DeepSeek requires authorized
Amazon Bedrock access and may return different choices because the provider
does not expose a deterministic seed. Recorded responses and request-level
token use are preserved for audit.

All source and result files are listed with SHA-256 hashes in
SUPPLEMENT-MANIFEST.json.
The frozen input.json is intentionally excluded because it materializes Brown
passages; the prepare command reconstructs it from the recorded source offsets
and checks the expected passage hashes.
"""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def payload(root: Path = ROOT) -> dict[str, bytes]:
    files = {}
    for name in FILES:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"required supplement file is missing: {name}")
        data = path.read_bytes()
        if any(marker.search(data) for marker in IDENTITY_MARKERS):
            raise ValueError(f"identity marker found in supplement file: {name}")
        files[name] = data
    return files


def zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=STAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def build_archive(output: Path = DEFAULT_OUTPUT, root: Path = ROOT) -> dict:
    files = payload(root)
    manifest = {
        "format": "naacl-static-verifier-supplement-v1",
        "created_date": "2026-09-25",
        "readme_sha256": sha256(README.encode()),
        "files_sha256": {name: sha256(data) for name, data in sorted(files.items())},
        "excluded_material": [
            "runs/static-language-verifier-20260925/input.json (contains reconstructed Brown passages)",
            "GPT-2 model weights (retrievable from the pinned public model revision)",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr(zip_info(README_NAME), README.encode())
        for name, data in sorted(files.items()):
            archive.writestr(zip_info(name), data)
        archive.writestr(zip_info(MANIFEST_NAME), (json.dumps(manifest, indent=2) + "\n").encode())
    verify_archive(output)
    return {
        "archive": str(output),
        "payload_file_count": len(files),
        "archive_sha256": sha256(output.read_bytes()),
        "payload_sha256": manifest["files_sha256"],
    }


def verify_archive(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if names != [README_NAME, *sorted(FILES), MANIFEST_NAME]:
            raise ValueError("supplement inventory differs from the frozen file list")
        manifest = json.loads(archive.read(MANIFEST_NAME))
        if manifest.get("readme_sha256") != sha256(archive.read(README_NAME)):
            raise ValueError("supplement README hash mismatch")
        expected = {name: sha256(archive.read(name)) for name in sorted(FILES)}
        if manifest.get("files_sha256") != expected:
            raise ValueError("supplement payload hash mismatch")
        if "runs/static-language-verifier-20260925/input.json" in names:
            raise ValueError("Brown passage input must not be distributed")
        for name in names:
            if any(marker.search(archive.read(name)) for marker in IDENTITY_MARKERS):
                raise ValueError(f"identity marker found in archive member: {name}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify", action="store_true", help="verify an existing supplement archive")
    args = parser.parse_args()
    if args.verify:
        result = {"verified": str(args.output), "payload_file_count": len(verify_archive(args.output)["files_sha256"])}
    else:
        result = build_archive(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
