"""Build a named source and evidence release; no network calls."""
from argparse import ArgumentParser
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE_ID = "revision-2026-09-10"


def run(*args):
    subprocess.run(args, cwd=ROOT, check=True)


def file_manifest(files):
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }


def write_archive(destination, files, *, flatten, release_manifest, readme):
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        hashes = {}
        for path in files:
            name = path.name if flatten else str(path.relative_to(ROOT))
            data = path.read_bytes()
            archive.writestr(name, data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        archive.writestr("MANIFEST-SHA256.json", json.dumps(hashes, indent=2) + "\n")
        archive.writestr("RELEASE-MANIFEST.json", json.dumps(release_manifest, indent=2) + "\n")
        archive.writestr("README.txt", readme)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--release-id", default=DEFAULT_RELEASE_ID)
    args = parser.parse_args()
    release_dir = ROOT / "paper" / "releases" / args.release_id
    release_dir.mkdir(parents=True, exist_ok=True)

    run("python3", "paper/build_revision_tables.py")
    run("python3", "experiments/verify_revision.py")

    source = [
        ROOT / "paper" / "eric_evidence_release_draft.md",
        ROOT / "paper" / "naacl2027.tex",
        ROOT / "paper" / "refs.bib",
        *sorted((ROOT / "paper").glob("revision-*.tex")),
    ]
    evidence = []
    patterns = [
        "artifacts/norvig-v3/*",
        "runs/revision-2026-09-07/*.json",
        "runs/revision-2026-09-07/*.sha256",
        "runs/revision-2026-09-07/*.csv",
        "runs/revision-2026-09-07/*.txt",
        "runs/punct/after_*.json",
        "experiments/revision_*.py",
        "experiments/verify_revision.py",
        "experiments/audit_norvig_result.py",
        "experiments/sentence_intersection-results.json",
        "experiments/RESULTS-*.md",
        "runs/polaris/scale_20260823/summaries.jsonl",
        "runs/polaris/sentence_plan_20260904_204815/aggregate.json",
        "runs/polaris/sentence_quality_20260905_011235/aggregate.json",
        "runs/sentence_quality*120b.json",
        "paper/SOURCE-AUDIT.md",
        "paper/eric_evidence_release_draft.md",
        "paper/build_revision_tables.py",
        "paper/build_release.py",
        "paper/validate_release.py",
        "paper/verify_structural_draft.py",
    ]
    for pattern in patterns:
        evidence.extend(path for path in ROOT.glob(pattern) if path.is_file())
    evidence = sorted(set(evidence))

    inputs = {}
    for pattern in [
        "runs/norvig/npdict.txt",
        "runs/norvig/pal21txt.html",
        "runs/norvig/pal3.py",
        "data/v3_bank.json",
        "data/*2w*",
    ]:
        for path in ROOT.glob(pattern):
            if path.is_file():
                inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()

    release_manifest = {
        "release_id": args.release_id,
        "current_working_draft": "paper/eric_evidence_release_draft.md",
        "archival_typeset_revision": "paper/naacl2027.tex",
        "source_bundle": "source.zip",
        "evidence_bundle": "evidence.zip",
        "source_build": "tectonic --outdir out naacl2027.tex",
        "evidence_checks": [
            "python3 paper/verify_structural_draft.py",
            "python3 experiments/verify_revision.py",
            f"python3 paper/validate_release.py --release-id {args.release_id} --compile",
        ],
        "external_inputs_sha256": inputs,
        "source_files_sha256": file_manifest(source),
        "evidence_files_sha256": file_manifest(evidence),
    }
    readme = (
        "Current working paper: eric_evidence_release_draft.md.\n"
        "Archival typeset revision: naacl2027.tex. Compile it with "
        "tectonic --outdir out naacl2027.tex.\n"
        "The evidence archive retains repository-relative paths. Run "
        "python3 paper/validate_release.py --release-id " + args.release_id +
        " --compile from the repository root to check archive membership and a clean source build.\n"
        "External corpora are not bundled. SOURCE-AUDIT.md records input versions, "
        "hashes, provenance, and missing evidence.\n"
    )
    (release_dir / "RELEASE-MANIFEST.json").write_text(
        json.dumps(release_manifest, indent=2) + "\n"
    )
    write_archive(release_dir / "source.zip", source, flatten=True,
                  release_manifest=release_manifest, readme=readme)
    write_archive(release_dir / "evidence.zip", evidence, flatten=False,
                  release_manifest=release_manifest, readme=readme)
    print(f"{args.release_id}: source={len(source)} evidence={len(evidence)}")


if __name__ == "__main__":
    main()
