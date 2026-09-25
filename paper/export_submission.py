"""Build allowlisted Overleaf source and anonymous evidence bundles.

Git history and unrelated raw runs are excluded. The selected-output audit,
matched-candidate archive, and length-stratified Brown diagnostic are included
because they directly support manuscript results.
"""
from __future__ import annotations

import ast
import gzip
import hashlib
import json
import re
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = ROOT / "output/naacl-submission"
PRIVATE_PATTERNS = (
    r"/Users/", r"/home/", r"(?i)ericspencer", r"(?i)overleaf\.com/project/",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----", r"\bgh[pousr]_[A-Za-z0-9]{20,}",
)


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def rename_candidate_digest_fields(value: object) -> object:
    if isinstance(value, dict):
        return {
            ("candidate_set_digest" if key == "candidate_key_sha256" else key):
            rename_candidate_digest_fields(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [rename_candidate_digest_fields(item) for item in value]
    return value


def audit_bytes(name: str, payload: bytes) -> None:
    source = gzip.decompress(payload) if name.endswith(".gz") else payload
    text = source.decode("utf-8")
    for pattern in PRIVATE_PATTERNS:
        if re.search(pattern, text):
            raise AssertionError(f"Privacy screen rejected {name}; pattern {pattern}")


def write_bundle(name: str, files: dict[str, bytes]) -> None:
    directory = OUT / name
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {path: hashlib.sha256(content).hexdigest() for path, content in sorted(files.items())}
    payloads = {**files, "manifest.json": json_bytes(manifest)}
    with zipfile.ZipFile(OUT / f"{name}.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename, payload in sorted(payloads.items()):
            if Path(filename).name != filename:
                raise ValueError("Bundle paths must be flat allowlisted filenames")
            audit_bytes(filename, payload)
            (directory / filename).write_bytes(payload)
            info = zipfile.ZipInfo(filename, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)


def bibliography_for(source: str) -> str:
    needed = set()
    for match in re.finditer(r"\\cite\w*\s*(?:\[[^\]]*\]\s*)*\{([^}]+)\}", source):
        needed.update(key.strip() for key in match.group(1).split(","))
    bib = (PAPER / "refs.bib").read_text()
    starts = list(re.finditer(r"^@\w+\{\s*([^,]+),", bib, re.MULTILINE))
    entries = {}
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(bib)
        entries[match.group(1)] = bib[match.start():end].strip()
    missing = needed - entries.keys()
    if missing:
        raise AssertionError(f"Unresolved bibliography keys: {sorted(missing)}")
    return "\n\n".join(entries[key] for key in sorted(needed)) + "\n"


def build() -> dict[str, object]:
    source = (PAPER / "naacl2027.tex").read_text()
    for part in ("week_results_table", "readability_table"):
        token = "\\input{" + part + "}"
        if source.count(token) != 1:
            raise AssertionError(f"Expected one manuscript input: {part}")
        source = source.replace(token, (PAPER / f"{part}.tex").read_text())
    if "\\input{" in source:
        raise AssertionError("Unresolved manuscript input in upload source")
    write_bundle("overleaf-source", {
        "naacl2027.tex": source.encode(),
        "refs.bib": bibliography_for(source).encode(),
        "acl.sty": (PAPER / "acl.sty").read_bytes(),
        "acl_natbib.bst": (PAPER / "acl_natbib.bst").read_bytes(),
    })

    selected = json.loads((PAPER / "week_results.json").read_text())
    selected.pop("snapshot", None)
    for row in selected["results"]:
        row["source"].pop("git_revision", None)
    index = json.loads((ROOT / "runs/incumbent-672-global-novelty-snapshot-20260922.json").read_text())
    module = ast.parse((ROOT / "experiments/luna6_god_dog_live_residual_growth_20260923.py").read_text())
    constants = {}
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"LEFT_INSERT", "RIGHT_INSERT"}:
                    constants[target.id] = ast.literal_eval(node.value)
    fixture = {
        "raw_cursors": [20, 758], "right_punctuation_skip": 1,
        "left_insert": constants["LEFT_INSERT"], "right_insert": constants["RIGHT_INSERT"],
        "scope": "Authored paired insertion; exact raw rendering replay, not a new generation run.",
    }
    readme = """# Anonymous construction evidence

Run `python3 verify_anonymous_evidence.py` after extracting this archive.
Python 3.10 or later and its standard library are sufficient. No network,
Git checkout, credentials, model API, or additional corpus is required.

The verifier checks all eleven selected renderings with a raw-text scan and
normalized reversal, including the complete 752-letter endpoint, reconstructs
the 630-letter edit, independently replays
the 672-letter first-success clause search using its fixed relation index,
checks every candidate in the matched operator-equivalence archive against the
parent seam and frozen relation index, and runs the bounded seam-algebra check.
It also recomputes the five-stage lineage's word and repetition diagnostics.
The verifier cross-checks the eleven Brown diagnostic records against the
selected exact texts, cross-checks the recorded nine long-output pairs and the
108 length-control summaries, and checks the bundle manifest.
In the comparison archive, `candidate_key_sha256` field names are normalized
to `candidate_set_digest`; every digest value and candidate row is unchanged.

Selection is retrospective, not exhaustive. These outputs are mechanically
exact construction results; no human-study results are included. The fixed
relation index supports replay, not a new historical novelty investigation.
The selected-results file preserves source filenames and source-file digests
for provenance. Original source files are not bundled, so those original-file
digests are provenance identifiers here, not independently rechecked inputs.
No Git history, host metadata, account information, or raw execution logs are
included. The two imported modules also have repository-specific entry points;
use the verifier command above for this standalone archive.

`readability-calibration.json` records a separate Brown word-bigram diagnostic:
ten selected project outputs of 54--752 letters plus an inherited 38-letter
reference, matched held-out prose spans, and 108 additional length controls.
It includes scores, hashes, split metadata, and exactness checks, but does not
redistribute the Brown control passages. The score measures local word order,
not human readability. To rerun it from the repository, install `nltk` and
`wordfreq`, make the NLTK Brown corpus available, and run
`python3 experiments/score_length_stratified_readability.py`.
"""
    comparison_run = ROOT / "runs/comparison-568-online-residual-vs-offline-reverse-index-20260924.json.gz"
    comparison_audit = ROOT / "runs/comparison-568-online-residual-vs-offline-reverse-index-20260924.audit.json"
    comparison_audit_data = json.loads(comparison_audit.read_text())
    comparison_audit_data = rename_candidate_digest_fields(comparison_audit_data)
    with gzip.open(comparison_run, "rt", encoding="utf-8") as stream:
        comparison_data = rename_candidate_digest_fields(json.load(stream))
    comparison_payload = gzip.compress(
        json.dumps(comparison_data, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        mtime=0,
    )
    files = {
        "selected-results.json": json_bytes(selected),
        "relation-index.json": json_bytes(index["relation_counts"]),
        "seam-fixture.json": json_bytes(fixture),
        "comparison-candidates.json.gz": comparison_payload,
        "comparison-audit.json": json_bytes(comparison_audit_data),
        "readability-calibration.json": json_bytes(json.loads(
            (ROOT / "runs/readability-length-stratified-20260925.json").read_text())),
        "README.md": readme.encode(),
    }
    for name in ("verify_anonymous_evidence.py", "check_seam_invariant.py", "replay_clause_search.py"):
        files[name] = (PAPER / name).read_bytes()
    write_bundle("anonymous-evidence", files)
    return {"source_bundle": str((OUT / "overleaf-source.zip").relative_to(ROOT)),
            "evidence_bundle": str((OUT / "anonymous-evidence.zip").relative_to(ROOT)),
            "privacy_screen": "passed on every allowlisted text file"}


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
