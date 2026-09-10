"""Validate named release archives, optionally compiling the source in a clean directory."""
from argparse import ArgumentParser
from pathlib import Path
import json
import shutil
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE_ID = "revision-2026-09-10"
SOURCE_REQUIRED = {
    "eric_evidence_release_draft.md",
    "naacl2027.tex",
    "refs.bib",
    "revision-agreement.tex",
    "revision-conservation-table.tex",
    "revision-judge-table.tex",
    "revision-punctuation-table.tex",
    "revision-seam-results.tex",
    "RELEASE-MANIFEST.json",
}
EVIDENCE_REQUIRED = {
    "paper/eric_evidence_release_draft.md",
    "paper/SOURCE-AUDIT.md",
    "paper/build_release.py",
    "paper/validate_release.py",
    "paper/verify_structural_draft.py",
    "artifacts/norvig-v3/palindrome.txt",
    "experiments/audit_norvig_result.py",
    "experiments/verify_revision.py",
    "runs/polaris/sentence_plan_20260904_204815/aggregate.json",
    "runs/punct/after_20b.json",
    "runs/punct/after_120b.json",
}


def archive_names(path):
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("MANIFEST-SHA256.json"))
        for name, expected in manifest.items():
            if name not in names:
                raise RuntimeError(f"{path}: manifest entry missing from archive: {name}")
            import hashlib
            actual = hashlib.sha256(archive.read(name)).hexdigest()
            if actual != expected:
                raise RuntimeError(f"{path}: manifest hash mismatch: {name}")
    return names


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--release-id", default=DEFAULT_RELEASE_ID)
    parser.add_argument("--compile", action="store_true",
                        help="compile the extracted source bundle with Tectonic")
    args = parser.parse_args()
    release_dir = ROOT / "paper" / "releases" / args.release_id
    source_zip = release_dir / "source.zip"
    evidence_zip = release_dir / "evidence.zip"
    source_names = archive_names(source_zip)
    evidence_names = archive_names(evidence_zip)
    missing_source = SOURCE_REQUIRED - source_names
    missing_evidence = EVIDENCE_REQUIRED - evidence_names
    if missing_source or missing_evidence:
        raise RuntimeError(
            f"missing source={sorted(missing_source)} evidence={sorted(missing_evidence)}"
        )
    release_manifest = json.loads((release_dir / "RELEASE-MANIFEST.json").read_text())
    if release_manifest["release_id"] != args.release_id:
        raise RuntimeError("release manifest has the wrong release id")
    if args.compile:
        tectonic = shutil.which("tectonic")
        if tectonic is None:
            raise RuntimeError("Tectonic is required for --compile")
        with tempfile.TemporaryDirectory(prefix="palindrome-release-") as directory:
            extracted = Path(directory) / "source"
            output = Path(directory) / "out"
            with zipfile.ZipFile(source_zip) as archive:
                archive.extractall(extracted)
            output.mkdir()
            subprocess.run(
                [tectonic, "--outdir", str(output), "naacl2027.tex"],
                cwd=extracted,
                check=True,
            )
            if not (output / "naacl2027.pdf").is_file():
                raise RuntimeError("clean source build did not produce naacl2027.pdf")
    print(f"Passed: {args.release_id}; source={len(source_names)} evidence={len(evidence_names)}")


if __name__ == "__main__":
    main()
