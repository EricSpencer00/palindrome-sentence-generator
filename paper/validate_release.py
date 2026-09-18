"""Check hashes, portable paths, clean extraction, saved audits, and rerun entry points."""
from argparse import ArgumentParser
from pathlib import Path, PurePosixPath
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE_ID = "naacl2027-2026-09-11"
QUARANTINED_RELEASE_IDS = {"naacl2027-2026-09-11"}
SOURCE_REQUIRED = {"naacl2027.tex", "refs.bib", "acl.sty", "acl_natbib.bst",
                   "fig/mirror-cost.pdf"}
EVIDENCE_REQUIRED = {"paper/naacl2027.tex", "paper/verify_structural_draft.py",
    "paper/critique_evidence.py", "paper/rerun_search.py", "SOURCE-SNAPSHOT.json",
    "experiments/controlled_pos_pruning.py", "experiments/audit_controlled_pos_pruning.py",
    "inputs/brown.json.gz", "inputs/vocab30k.txt", "inputs/norvig/npdict.txt",
    "inputs/norvig/pal3.py", "inputs/norvig/pal21txt.html",
    "data/structural/aggregate.json", "data/length/palindrome.txt",
    "data/length/phrases.json", "data/length/result.json", "TERMS.md", "README.md"}
EVIDENCE_REQUIRED |= {"data/mirror-cost/results.json", "data/mirror-cost/audit.json",
    "data/mirror-cost/RESULTS.md", "experiments/mirror_cost.py",
    "experiments/audit_mirror_cost.py", "data/lexicon.txt", "requirements.txt",
    "data/long-form/examples.json", "data/long-form/audit.json",
    "experiments/audit_long_form_examples.py", "experiments/freeze_long_form_examples.py",
    "data/novel_pairs.json", "data/mirror_units.json", "data/centres.json",
    "data/known_palindromes.json"}
EVIDENCE_REQUIRED |= {"data/controlled/provenance.json", "data/controlled/summary.json",
    "data/controlled/trials.jsonl", "data/controlled/RESULTS.md",
    "data/controlled/audit.json"}
PRIVATE_PATH = re.compile(rb"(?:tools|runs)/polaris/|/(?:Users|home|lus|eagle|gpfs)/|polaris\.alcf", re.I)
META = {"MANIFEST-SHA256.json", "RELEASE-MANIFEST.json", "README.txt"}


def inspect_archive(path, manifest, key):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("Duplicate archive entries")
        for name in names:
            parts = PurePosixPath(name)
            if parts.is_absolute() or ".." in parts.parts or "\\" in name:
                raise RuntimeError(f"Unsafe archive member: {name}")
            data = archive.read(name)
            if name.endswith(".gz"):
                data = gzip.decompress(data)
            if PRIVATE_PATH.search(name.encode()) or PRIVATE_PATH.search(data):
                raise RuntimeError(f"Machine-specific path in {path.name}: {name}")
        hashes = json.loads(archive.read("MANIFEST-SHA256.json"))
        if set(hashes) != set(names) - META or hashes != manifest[key]:
            raise RuntimeError("Archive payload differs from the release manifest")
        for name, expected in hashes.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise RuntimeError(f"Hash mismatch: {name}")
        if json.loads(archive.read("RELEASE-MANIFEST.json")) != manifest:
            raise RuntimeError("Inner and outer release manifests differ")
        paper = "naacl2027.tex" if key == "source_files_sha256" else "paper/naacl2027.tex"
        if archive.read(paper) != (ROOT / "paper/naacl2027.tex").read_bytes():
            raise RuntimeError("Bundled manuscript differs from current manuscript")
        return set(names) - META


def run(directory, *args, isolated=True):
    env = dict(os.environ, PYTHONPATH="", PYTHONNOUSERSITE="1")
    command = [sys.executable] + (["-S"] if isolated else []) + list(args)
    result = subprocess.run(command, cwd=directory, env=env,
                            text=True, capture_output=True, timeout=240)
    if result.returncode:
        raise RuntimeError(f"Execution failed: {' '.join(args)}\n{result.stderr[-3000:]}")
    return result


def verify_reports(directory):
    report = json.loads((directory / "structural-evidence.json").read_text())
    diversity = json.loads((directory / "critique-evidence.json").read_text())
    assert report["structure"]["word_types"] == 49815
    assert report["structure"]["word_tag_associations"] == 53548
    assert report["output"]["letters"] == 90937
    assert report["output"]["unique_phrases"] == 16168
    assert diversity["shared_pairs"] == 16766
    assert diversity["exclusive_pairs"]["planned"] == {"pairs": 69745, "junction_families": 29}
    assert diversity["arms"]["planned"]["junction_families"] == 30
    assert diversity["arms"]["terminal"]["verified_pairs"] == 20989
    assert diversity["arms"]["planned"]["verified_pairs"] == 86511
    assert report["source_snapshot"]["git"] is None
    controlled = json.loads((directory / "controlled-audit.json").read_text())
    assert controlled["status"] == "ok"
    assert controlled["paired_trials"] == 50
    assert controlled["checked_accepted_pair_rows"] == 2544
    assert controlled["terminal_pairs_missing_from_matched_incremental_arm"] == 0
    assert controlled["totals"]["terminal"]["accepted"] == 407
    assert controlled["totals"]["incremental"]["accepted"] == 2137
    mirror = json.loads((directory / "mirror-cost-audit.json").read_text())
    assert mirror["status"] == "pass"
    assert mirror["cells"] == 36
    assert mirror["spans_per_length"] == 150
    assert round(mirror["mirror_cost_min"], 2) == 2.14
    assert round(mirror["mirror_cost_max"], 2) == 3.34
    assert mirror["checks"]["reported_arithmetic_consistent"] is True
    long_form = json.loads((directory / "long-form-audit.json").read_text())
    assert long_form["status"] == "pass"
    assert long_form["examples"]["generated"]["words"] == 101
    assert long_form["examples"]["generated"]["letters"] == 342
    assert long_form["examples"]["catalogue"]["words"] == 72
    assert long_form["examples"]["catalogue"]["letters"] == 237
    origin = json.loads((directory / "SOURCE-SNAPSHOT.json").read_text())
    for name, expected in origin["files_sha256"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected, name


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--release-id", default=DEFAULT_RELEASE_ID)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="also run short structural and dictionary execution checks")
    args = parser.parse_args()
    raise RuntimeError(
        "release validation is disabled: no current-evidence submission bundle exists. "
        "Historical archives are quarantined until shared-gate survivors have "
        "blinded human-reader evidence."
    )
    if args.release_id in QUARANTINED_RELEASE_IDS:
        raise RuntimeError("historical release is quarantined; no current submission release exists")
    if sys.flags.optimize:
        parser.error("Run without Python optimization")
    release = ROOT / "paper/releases" / args.release_id
    manifest = json.loads((release / "RELEASE-MANIFEST.json").read_text())
    if manifest["release_id"] != args.release_id:
        raise RuntimeError("Wrong release ID")
    sources = inspect_archive(release / "source.zip", manifest, "source_files_sha256")
    evidence = inspect_archive(release / "evidence.zip", manifest, "evidence_files_sha256")
    assert sources == SOURCE_REQUIRED
    for name, expected in manifest["source_files_sha256"].items():
        current = ROOT / "paper" / name
        assert hashlib.sha256(current.read_bytes()).hexdigest() == expected, f"Stale source: {name}"
    assert EVIDENCE_REQUIRED <= evidence
    with tempfile.TemporaryDirectory(prefix="palindrome-release-") as temporary:
        directory = Path(temporary)
        with zipfile.ZipFile(release / "evidence.zip") as archive:
            archive.extractall(directory / "evidence")
        extracted = directory / "evidence"
        for program, output in (("verify_structural_draft.py", "structural-evidence.json"),
                                ("critique_evidence.py", "critique-evidence.json")):
            run(extracted, "paper/" + program, "--output", output)
        run(extracted, "-m", "experiments.audit_controlled_pos_pruning",
            "data/controlled", "--output", "controlled-audit.json")
        run(extracted, "-m", "experiments.audit_mirror_cost",
            "data/mirror-cost/results.json", "--output", "mirror-cost-audit.json",
            isolated=False)
        run(extracted, "-m", "experiments.audit_long_form_examples",
            "data/long-form/examples.json", "--root", ".",
            "--output", "long-form-audit.json")
        verify_reports(extracted)
        if args.smoke:
            run(extracted, "paper/rerun_search.py", "--shards", "2", "--seconds-per-arm", "1",
                "--node-budget", "4096", "--max-hits", "5", "--out-dir", "rerun/structural")
            summaries = list((extracted / "rerun/structural").glob("summary_*.json"))
            assert len(summaries) == 2
            for path in summaries:
                row = json.loads(path.read_text())
                assert [arm["arm"] for arm in row["arms"]] == ["terminal", "planned"]
                assert all(arm["stop_reason"] in {"node_budget", "deadline", "max_hits", "exhausted"} for arm in row["arms"])
            run(extracted, "-m", "experiments.norvig_letters", "--seconds", "2", "--dynamic", "--feasible",
                "--max-content-uses", "3", "--out", "rerun/length")
            run(extracted, "experiments/audit_norvig_result.py", "rerun/length", "--dictionary", "inputs/norvig/npdict.txt",
                "--reference", "inputs/norvig/pal21txt.html")
            audit = json.loads((extracted / "rerun/length/audit.json").read_text())
            assert audit["maximum_content_word_uses"] <= 3
            assert (extracted / "rerun/length/provenance.json").is_file()
        if args.compile:
            tectonic = shutil.which("tectonic")
            if not tectonic:
                raise RuntimeError("Tectonic is required for --compile")
            with zipfile.ZipFile(release / "source.zip") as archive:
                archive.extractall(directory / "source")
            subprocess.run([tectonic, "naacl2027.tex"], cwd=directory / "source", check=True,
                           stdout=subprocess.DEVNULL, timeout=180)
            assert (directory / "source/naacl2027.pdf").is_file()
    print(f"Passed: {args.release_id}; source={len(sources)} evidence={len(evidence)}; portable paths and clean-extraction audits" +
          ("; both rerun entry points" if args.smoke else "") + ("; extracted source compiled" if args.compile else ""))


if __name__ == "__main__":
    main()
