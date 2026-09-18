"""Build portable source and evidence archives from the current working files."""
from argparse import ArgumentParser
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper.provenance import snapshot

QUARANTINED_RELEASE_IDS = {"naacl2027-2026-09-11"}
STAGE = ROOT / "paper/out/evidence"
MODULES = ("__init__", "exhaustive", "pairs", "search", "sentence_plan", "syntax",
           "centerout", "lexicon", "hierarchy", "bigram", "validator", "shortwords")
PAPER_CODE = ("verify_structural_draft", "critique_evidence", "rerun_search", "evidence_paths", "provenance")
INVENTORY_RUNS = ("static", "dynamic", "dynamic-long", "feasible", "comparable", "feasible-300")
CONTROLLED_RUN = ROOT / "runs/controlled-pos-pruning-openings-2026-09-11"
MIRROR_RUN = ROOT / "runs/mirror-cost-2026-09-11"
LONG_FORM_RUN = ROOT / "runs/long-form-examples-2026-09-11"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy(source, name):
    destination = STAGE / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / source, destination)
    return destination


def write_archive(path, files, manifest, readme):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        hashes = {}
        for name, source in sorted(files.items()):
            data = source.read_bytes()
            archive.writestr(name, data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        archive.writestr("MANIFEST-SHA256.json", json.dumps(hashes, indent=2) + "\n")
        archive.writestr("RELEASE-MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
        archive.writestr("README.txt", readme)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--release-id", required=True)
    args = parser.parse_args()
    raise RuntimeError(
        "release building is disabled: no current-evidence submission bundle exists. "
        "Historical archives are quarantined until shared-gate survivors have "
        "blinded human-reader evidence."
    )
    if Path(args.release_id).name != args.release_id or args.release_id in (".", ".."):
        parser.error("release-id must be a directory name")
    if args.release_id in QUARANTINED_RELEASE_IDS:
        parser.error("the historical release is quarantined and cannot be rebuilt")
    release = ROOT / "paper/releases" / args.release_id
    release.mkdir(parents=True, exist_ok=True)
    # This is a generated staging directory, never a source or saved run.
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    mapping = {
        "inputs/brown.json.gz": "tools/polaris/payload/brown.json.gz",
        "inputs/vocab30k.txt": "tools/polaris/payload/vocab30k.txt",
        "data/structural/aggregate.json": "runs/polaris/sentence_plan_20260904_204815/aggregate.json",
        "data/mirror-cost/results.json": "runs/mirror-cost-2026-09-11/results.json",
        "data/mirror-cost/audit.json": "runs/mirror-cost-2026-09-11/audit.json",
        "data/mirror-cost/RESULTS.md": "runs/mirror-cost-2026-09-11/RESULTS.md",
        "data/long-form/examples.json": "runs/long-form-examples-2026-09-11/examples.json",
        "data/long-form/audit.json": "runs/long-form-examples-2026-09-11/audit.json",
        "data/lexicon.txt": "data/lexicon.txt",
        "data/novel_pairs.json": "data/novel_pairs.json",
        "data/mirror_units.json": "data/mirror_units.json",
        "data/centres.json": "data/centres.json",
        "data/known_palindromes.json": "data/known_palindromes.json",
        "requirements.txt": "paper/MIRROR-COST-REQUIREMENTS.txt",
        "TERMS.md": "paper/DATA-TERMS.md",
        "README.md": "paper/EVIDENCE-README.md",
        "paper/REVISION-RESPONSE.md": "paper/REVISION-RESPONSE.md",
        "paper/naacl2027.tex": "paper/naacl2027.tex",
        "paper/refs.bib": "paper/refs.bib",
        "licenses/PROJECT-LICENSE.txt": "LICENSE",
        "server/v3.py": "server/v3.py",
    }
    for name in ("npdict.txt", "pal3.py", "pal21txt.html"):
        mapping[f"inputs/norvig/{name}"] = f"runs/norvig/{name}"
    for name in ("palindrome.txt", "phrases.json", "result.json", "audit.json"):
        mapping[f"data/length/{name}"] = f"artifacts/norvig-v3/{name}"
    for run in INVENTORY_RUNS:
        mapping[f"data/inventory/{run}.json"] = f"runs/norvig-letter-{run}/result.json"
    for name in ("provenance.json", "summary.json", "trials.jsonl", "RESULTS.md", "audit.json"):
        mapping[f"data/controlled/{name}"] = str((CONTROLLED_RUN / name).relative_to(ROOT))
    for module in MODULES:
        mapping[f"llm_palindrome/{module}.py"] = f"llm_palindrome/{module}.py"
    for module in PAPER_CODE:
        mapping[f"paper/{module}.py"] = f"paper/{module}.py"
    for module in ("norvig_letters", "norvig_long", "audit_norvig_result",
                   "controlled_pos_pruning", "audit_controlled_pos_pruning",
                   "mirror_cost", "audit_mirror_cost",
                   "freeze_long_form_examples", "audit_long_form_examples"):
        mapping[f"experiments/{module}.py"] = f"experiments/{module}.py"
    mapping["tests/test_mirror_cost.py"] = "tests/test_mirror_cost.py"
    mapping["tests/test_novel_bank.py"] = "tests/test_novel_bank.py"
    for path in (ROOT / "paper/licenses").iterdir():
        if path.is_file():
            mapping[f"licenses/{path.name}"] = str(path.relative_to(ROOT))
    for name, source in mapping.items():
        copy(source, name)
    # Retain exact current bytes and the dirty checkout status, not HEAD alone.
    code = [STAGE / name for name in mapping if name.endswith((".py", ".tex", ".bib"))]
    origin = snapshot(STAGE, code)
    origin["git"] = snapshot(ROOT, [])["git"]
    (STAGE / "SOURCE-SNAPSHOT.json").write_text(json.dumps(origin, indent=2) + "\n")
    for program, output in (("verify_structural_draft.py", "structural-evidence.json"),
                            ("critique_evidence.py", "critique-evidence.json")):
        with (STAGE / "paper" / output).open("w") as report:
            subprocess.run([sys.executable, "paper/" + program], cwd=STAGE, stdout=report, check=True)
        shutil.copyfile(STAGE / "paper" / output, ROOT / "paper" / output)
    subprocess.run([sys.executable, "-m", "experiments.audit_controlled_pos_pruning",
                    "data/controlled", "--output", "paper/controlled-audit.json"],
                   cwd=STAGE, check=True, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, "-m", "experiments.audit_mirror_cost",
                    "data/mirror-cost/results.json", "--output", "paper/mirror-cost-audit.json"],
                   cwd=STAGE, check=True, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, "-m", "experiments.audit_long_form_examples",
                    "data/long-form/examples.json", "--root", ".",
                    "--output", "paper/long-form-audit.json"],
                   cwd=STAGE, check=True, stdout=subprocess.DEVNULL)
    source = {name: ROOT / "paper" / name for name in ("naacl2027.tex", "refs.bib", "acl.sty", "acl_natbib.bst")}
    source["fig/mirror-cost.pdf"] = ROOT / "paper/fig/mirror-cost.pdf"
    evidence = {str(path.relative_to(STAGE)): path for path in STAGE.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts}
    manifest = {
        "release_id": args.release_id,
        "title": "Measuring Reversal Cost in English for Exact Palindrome Search",
        "current_working_draft": "paper/naacl2027.tex",
        "source_build": "tectonic naacl2027.tex",
        "source_bundle": "source.zip", "evidence_bundle": "evidence.zip",
        "reproduction_context": "Extract evidence.zip and run the commands in README.md. The structural and long-form audits use the standard library; the mirror-cost audit additionally requires the pinned wordfreq package. Full model rescoring uses all packages in requirements.txt and the recorded model revisions.",
        "evidence_checks": ["python3 -m experiments.audit_mirror_cost data/mirror-cost/results.json --output mirror-cost-audit.json",
                            "python3 -m experiments.audit_long_form_examples data/long-form/examples.json --root . --output long-form-audit.json",
                            "python3 paper/verify_structural_draft.py --output structural-evidence.json",
                            "python3 paper/critique_evidence.py --output critique-evidence.json",
                            "python3 -m experiments.audit_controlled_pos_pruning data/controlled --output controlled-audit.json"],
        "source_files_sha256": {name: digest(path) for name, path in sorted(source.items())},
        "evidence_files_sha256": {name: digest(path) for name, path in sorted(evidence.items())},
        "historical_provenance": "Incomplete: source revision, full environment, and per-process structural traces were not frozen.",
        "deposit_status": "prepared_locally; private repository upload is not recorded by this build",
    }
    (release / "RELEASE-MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    readme = (ROOT / "paper/EVIDENCE-README.md").read_text()
    write_archive(release / "source.zip", source, manifest,
                  "Extract this archive and run tectonic naacl2027.tex. The evidence commands are in evidence.zip.\n")
    write_archive(release / "evidence.zip", evidence, manifest, readme)
    # Do not leave an importable duplicate package tree under paper/out: test
    # discovery can otherwise import staged modules instead of the checkout.
    shutil.rmtree(STAGE)
    print(f"{args.release_id}: source={len(source)} evidence={len(evidence)}")


if __name__ == "__main__":
    main()
