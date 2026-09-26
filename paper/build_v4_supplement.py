"""Build a deterministic, self-contained v4 evidence and source archive."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "paper" / "v4-reproducibility.zip"

FILES = (
    "data/known_palindromes.json",
    "data/lexicon.txt",
    "experiments/__init__.py",
    "experiments/DIRECTIONAL-SEARCH-RESOURCE-SENSITIVITY-20260925.md",
    "experiments/DIRECTIONAL-SEARCH-BENCHMARK-20260925.md",
    "experiments/analyze_directional_search_resource_sensitivity_20260925.py",
    "experiments/analyze_paired_directional_search_benchmark_20260925.py",
    "experiments/audit_programmatic_readability.py",
    "experiments/infer_deepseek_static_verifier_20260925.py",
    "experiments/paired_directional_search_benchmark_20260925.py",
    "experiments/paired_directional_search_resource_sensitivity_20260925.py",
    "experiments/prepare_directional_static_verifier_20260925.py",
    "experiments/prepare_directional_static_verifier_aws_20260925.py",
    "experiments/score_directional_search_brown_20260925.py",
    "experiments/score_gpt2_static_verifier_20260925.py",
    "experiments/score_gpt2_static_verifier_aws_20260925.py",
    "experiments/score_gpt2_static_verifier_20260925.py",
    "experiments/score_length_stratified_readability.py",
    "experiments/static-verifier-bedrock-requirements.txt",
    "experiments/static-verifier-nuc-requirements.txt",
    "experiments/static_language_verifier_20260925.py",
    "experiments/summarize_directional_static_verifier_20260925.py",
    "llm_palindrome/__init__.py",
    "llm_palindrome/admission.py",
    "llm_palindrome/centerout.py",
    "llm_palindrome/generate.py",
    "llm_palindrome/lexicon.py",
    "llm_palindrome/overhang.py",
    "llm_palindrome/scoring.py",
    "llm_palindrome/search.py",
    "llm_palindrome/safe_vocab.py",
    "llm_palindrome/shortwords.py",
    "llm_palindrome/textify.py",
    "llm_palindrome/validator.py",
    "paper/acl.sty",
    "paper/acl_natbib.bst",
    "paper/build_v4_supplement.py",
    "paper/naacl2027.tex",
    "paper/readability-requirements.txt",
    "paper/refs.bib",
    "runs/directional-search-benchmark-20260925/confirmatory-v3-native-closure.json",
    "runs/directional-search-benchmark-20260925/manifest.json",
    "runs/directional-search-resource-sensitivity-20260925/brown-v4-aws.json",
    "runs/directional-search-resource-sensitivity-20260925/confirmatory-v4-resource-sensitivity.json",
    "runs/directional-search-resource-sensitivity-20260925/deepseek-v4.json",
    "runs/directional-search-resource-sensitivity-20260925/gpt2-v4-aws.json",
    "runs/directional-search-resource-sensitivity-20260925/manifest.json",
    "runs/directional-search-resource-sensitivity-20260925/static-verifier-input-aws.json",
    "runs/directional-search-resource-sensitivity-20260925/static-verifier-summary-v4-aws.json",
    "runs/directional-search-resource-sensitivity-20260925/summary-v4.json",
    "tests/test_initial_state_search.py",
    "tools/polaris/payload/vocab30k.txt",
)

README = """# v4 supplementary evidence and reproduction

This archive accompanies `paper/naacl2027.tex`. Extract it at a clean checkout
of the repository root, preserving its paths. It contains the frozen v4
manifest and raw results, v3 comparison used to motivate the adaptive follow-
up, all 66 selected texts and their 32 fixed shuffles, model outputs, source
code, vocabulary/lexicon inputs, and pinned dependency lists. `MANIFEST.sha256`
records each archived file hash.

Author-specific machine names and repository commit IDs are omitted from the
exported run metadata. Source-file hashes, model revisions, and cross-file
reproduction hashes are retained and checked.

## Search replay

Use Python 3.10+ and install the package versions recorded in the requirements
files. From the repository root, run:

```sh
python3 experiments/paired_directional_search_resource_sensitivity_20260925.py run \\
  --manifest runs/directional-search-resource-sensitivity-20260925/manifest.json \\
  --phase confirmatory --output /tmp/replayed-v4.json --workers 6
python3 experiments/analyze_directional_search_resource_sensitivity_20260925.py \\
  --manifest runs/directional-search-resource-sensitivity-20260925/manifest.json \\
  --results /tmp/replayed-v4.json \\
  --static-verifier-summary runs/directional-search-resource-sensitivity-20260925/static-verifier-summary-v4-aws.json \\
  --json-output /tmp/v4-summary.json --markdown-output /tmp/v4-summary.md
```

The search runner checks every source hash and the pinned `wordfreq` version
before it runs. Its random seeds, task order, vocabulary, gates, and resource
settings are frozen in the manifest. Execution time can vary by host; no run
in the reported experiment reached its cooperative deadline.

## Language diagnostics

Install `paper/readability-requirements.txt` and the Brown corpus to rerun the
Brown scorer; install `experiments/static-verifier-nuc-requirements.txt` and
use the pinned GPT-2 revision shown in the paper to rerun GPT-2. Example:

```sh
NLTK_DATA=/path/to/nltk_data python3 -m experiments.score_directional_search_brown_20260925 \\
  --input runs/directional-search-resource-sensitivity-20260925/static-verifier-input-aws.json \\
  --output /tmp/brown-v4.json
HF_HOME=/path/to/hf-cache OMP_NUM_THREADS=4 python3 -m experiments.score_gpt2_static_verifier_aws_20260925 \\
  --input runs/directional-search-resource-sensitivity-20260925/static-verifier-input-aws.json \\
  --output /tmp/gpt2-v4.json --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e \\
  --batch-size 8 --execution-host 'authorized CPU host'
python3 experiments/infer_deepseek_static_verifier_20260925.py \\
  --input runs/directional-search-resource-sensitivity-20260925/static-verifier-input-aws.json \\
  --output /tmp/deepseek-v4.json --budget-usd 0.20 --interval-seconds 0.75
python3 experiments/summarize_directional_static_verifier_20260925.py \\
  --input runs/directional-search-resource-sensitivity-20260925/static-verifier-input-aws.json \\
  --brown /tmp/brown-v4.json --gpt2 /tmp/gpt2-v4.json --deepseek /tmp/deepseek-v4.json \\
  --output /tmp/static-summary-v4.json
```

These scores are deterministic for the archived inputs and model/corpus
versions. A fresh
DeepSeek V3.2 request requires an authorized Bedrock account and
`experiments/static-verifier-bedrock-requirements.txt`; the archived requests
and responses can be re-parsed without a new call. To obtain new responses,
run the included cost-capped scorer with `--budget-usd 0.20`; Bedrock did not provide a
model seed, so a new response is not guaranteed to match. The reported call
used 11,538 input tokens and 1,410 output tokens. Its estimated cost was USD
0.00976206, computed from recorded usage and published [Bedrock rates](https://aws.amazon.com/bedrock/pricing/).

The outputs are automatic diagnostics, not readability labels. No human
ratings are in this archive.
"""

V3_MANIFEST = "runs/directional-search-benchmark-20260925/manifest.json"
V3_RESULT = "runs/directional-search-benchmark-20260925/confirmatory-v3-native-closure.json"
V4_MANIFEST = "runs/directional-search-resource-sensitivity-20260925/manifest.json"
V4_RESULT = "runs/directional-search-resource-sensitivity-20260925/confirmatory-v4-resource-sensitivity.json"
V4_INPUT = "runs/directional-search-resource-sensitivity-20260925/static-verifier-input-aws.json"
V4_BROWN = "runs/directional-search-resource-sensitivity-20260925/brown-v4-aws.json"
V4_GPT2 = "runs/directional-search-resource-sensitivity-20260925/gpt2-v4-aws.json"
V4_DEEPSEEK = "runs/directional-search-resource-sensitivity-20260925/deepseek-v4.json"
V4_STATIC_SUMMARY = "runs/directional-search-resource-sensitivity-20260925/static-verifier-summary-v4-aws.json"
V4_SUMMARY = "runs/directional-search-resource-sensitivity-20260925/summary-v4.json"
V4_REPORT = "experiments/DIRECTIONAL-SEARCH-RESOURCE-SENSITIVITY-20260925.md"

# These fields disclose a specific author machine or repository revision.
# Model `revision` and content SHA-256 fields are intentionally unaffected.
DROP_METADATA_FIELDS = {
    "manifest_preparation_host",
    "project_git_commit_at_freeze",
    "git_revision",
    "snapshot_revision",
    "snapshot_commit",
    "novelty_snapshot_commit",
    "host",
}
ANONYMITY_FIELD_NAMES = DROP_METADATA_FIELDS
PRIVATE_TEXT_PATTERNS = (
    re.compile(rb"/Users/[A-Za-z0-9._-]+", re.IGNORECASE),
    re.compile(rb"/home/[A-Za-z0-9._-]+", re.IGNORECASE),
    re.compile(rb"EricSpencer00", re.IGNORECASE),
    re.compile(rb"overleaf\.com/project/", re.IGNORECASE),
    re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{20,}"),
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def sanitize_metadata(value: object) -> tuple[object, bool]:
    """Remove author-specific machine and repository fields recursively."""
    if isinstance(value, dict):
        result = {}
        changed = False
        for key, item in value.items():
            if key in DROP_METADATA_FIELDS:
                changed = True
                continue
            sanitized, item_changed = sanitize_metadata(item)
            result[key] = sanitized
            changed = changed or item_changed
        return result, changed
    if isinstance(value, list):
        result, changed = [], False
        for item in value:
            sanitized, item_changed = sanitize_metadata(item)
            result.append(sanitized)
            changed = changed or item_changed
        return result, changed
    return value, False


def parse_json(content: dict[str, bytes], name: str) -> dict:
    try:
        value = json.loads(content[name].decode("utf-8"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"required JSON archive member is invalid: {name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {name}")
    return value


def store_json(content: dict[str, bytes], name: str, value: dict) -> None:
    content[name] = json_bytes(value)


def refresh_path_hashes(value: object, hashes_by_path: dict[str, str]) -> None:
    """Refresh embedded provenance links for archive members that were scrubbed."""
    if isinstance(value, dict):
        manifest_path = value.get("manifest_path_from_repository_root")
        if manifest_path in hashes_by_path and "manifest_sha256" in value:
            value["manifest_sha256"] = hashes_by_path[manifest_path]
        result_path = value.get("result_path_from_repository_root")
        if result_path in hashes_by_path and "result_sha256" in value:
            value["result_sha256"] = hashes_by_path[result_path]
        for item in value.values():
            refresh_path_hashes(item, hashes_by_path)
    elif isinstance(value, list):
        for item in value:
            refresh_path_hashes(item, hashes_by_path)


def export_anonymized_json(content: dict[str, bytes]) -> None:
    """Scrub author metadata and rebind every included v3/v4 hash edge."""
    # Scrub metadata in every JSON source without rewriting unrelated JSON bytes.
    for name, payload in list(content.items()):
        if not name.endswith(".json"):
            continue
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid JSON archive member: {name}") from exc
        sanitized, changed = sanitize_metadata(value)
        if changed:
            content[name] = json_bytes(sanitized)

    # The v3 result is cryptographically tied to its archived manifest.
    v3_manifest = parse_json(content, V3_MANIFEST)
    v3_manifest_sha = sha256(content[V3_MANIFEST])
    v3_result = parse_json(content, V3_RESULT)
    v3_result["protocol_manifest_sha256"] = v3_manifest_sha
    store_json(content, V3_RESULT, v3_result)
    v3_result_sha = sha256(content[V3_RESULT])

    # The v4 manifest carries the included v3 manifest/result digests in its
    # prior-protocol records. Refresh only paths that are part of this archive.
    v4_manifest = parse_json(content, V4_MANIFEST)
    refresh_path_hashes(v4_manifest, {
        V3_MANIFEST: v3_manifest_sha,
        V3_RESULT: v3_result_sha,
    })
    store_json(content, V4_MANIFEST, v4_manifest)
    v4_manifest_sha = sha256(content[V4_MANIFEST])

    v4_result = parse_json(content, V4_RESULT)
    v4_result["protocol_manifest_sha256"] = v4_manifest_sha
    store_json(content, V4_RESULT, v4_result)
    v4_result_sha = sha256(content[V4_RESULT])

    # Rebind the saved scorer input to the sanitized search artifacts. Candidate
    # and shuffle text are unchanged, so model output rows remain valid.
    v4_input = parse_json(content, V4_INPUT)
    v4_input["input_manifest_sha256"] = v4_manifest_sha
    v4_input["search_result_sha256"] = v4_result_sha
    for item in v4_input.get("items", []):
        source = item.get("source")
        if not isinstance(source, dict):
            continue
        if "search_manifest_sha256" in source:
            source["search_manifest_sha256"] = v4_manifest_sha
        if "search_result_sha256" in source:
            source["search_result_sha256"] = v4_result_sha
    store_json(content, V4_INPUT, v4_input)
    v4_input_sha = sha256(content[V4_INPUT])

    # Scorer data values are unchanged; only their frozen-input checksum changes.
    score_files = (V4_BROWN, V4_GPT2, V4_DEEPSEEK)
    score_hashes = {}
    for name in score_files:
        score = parse_json(content, name)
        score["input_manifest_sha256"] = v4_input_sha
        store_json(content, name, score)
        score_hashes[name] = sha256(content[name])

    static_summary = parse_json(content, V4_STATIC_SUMMARY)
    static_summary["input_manifest_sha256"] = v4_input_sha
    inputs = static_summary.get("inputs", {})
    for key, name in (
        ("brown_result_sha256", V4_BROWN),
        ("gpt2_result_sha256", V4_GPT2),
        ("deepseek_result_sha256", V4_DEEPSEEK),
    ):
        if key in inputs:
            inputs[key] = score_hashes[name]
    store_json(content, V4_STATIC_SUMMARY, static_summary)
    static_summary_sha = sha256(content[V4_STATIC_SUMMARY])

    # The report prints the frozen-input and summary digests, so bind those
    # claims to the anonymized exported copies rather than the private originals.
    report = content[V4_REPORT].decode("utf-8")
    original_input_sha = sha256((ROOT / V4_INPUT).read_bytes())
    original_summary_sha = sha256((ROOT / V4_STATIC_SUMMARY).read_bytes())
    report = report.replace(original_input_sha, v4_input_sha)
    report = report.replace(original_summary_sha, static_summary_sha)
    content[V4_REPORT] = report.encode("utf-8")

    # Refresh the combined report's direct links and its embedded v3 history.
    summary = parse_json(content, V4_SUMMARY)
    summary["manifest_sha256"] = v4_manifest_sha
    summary["result_sha256"] = v4_result_sha
    refresh_path_hashes(summary, {
        V3_MANIFEST: v3_manifest_sha,
        V3_RESULT: v3_result_sha,
    })
    static = summary.get("static_verifier", {})
    if isinstance(static, dict):
        static["summary_sha256"] = static_summary_sha
        static["input_manifest_sha256"] = v4_input_sha
        static_inputs = static.get("inputs", {})
        for key, name in (
            ("brown_result_sha256", V4_BROWN),
            ("gpt2_result_sha256", V4_GPT2),
            ("deepseek_result_sha256", V4_DEEPSEEK),
        ):
            if key in static_inputs:
                static_inputs[key] = score_hashes[name]
    store_json(content, V4_SUMMARY, summary)

    validate_internal_hashes(content)


def validate_internal_hashes(content: dict[str, bytes]) -> None:
    """Fail closed if anonymization left any archived provenance link stale."""
    hashes = {name: sha256(payload) for name, payload in content.items()}
    m3 = parse_json(content, V3_MANIFEST)
    r3 = parse_json(content, V3_RESULT)
    m4 = parse_json(content, V4_MANIFEST)
    r4 = parse_json(content, V4_RESULT)
    frozen_input = parse_json(content, V4_INPUT)
    brown = parse_json(content, V4_BROWN)
    gpt2 = parse_json(content, V4_GPT2)
    deepseek = parse_json(content, V4_DEEPSEEK)
    static_summary = parse_json(content, V4_STATIC_SUMMARY)
    summary = parse_json(content, V4_SUMMARY)

    require_hash(r3.get("protocol_manifest_sha256"), hashes[V3_MANIFEST], V3_RESULT)
    require_hash(r4.get("protocol_manifest_sha256"), hashes[V4_MANIFEST], V4_RESULT)
    require_hash(frozen_input.get("input_manifest_sha256"), hashes[V4_MANIFEST], V4_INPUT)
    require_hash(frozen_input.get("search_result_sha256"), hashes[V4_RESULT], V4_INPUT)
    prior_v3 = m4.get("design", {}).get("prior_v3_result", {})
    require_hash(prior_v3.get("manifest_sha256"), hashes[V3_MANIFEST], V4_MANIFEST)
    require_hash(prior_v3.get("result_sha256"), hashes[V3_RESULT], V4_MANIFEST)

    archived_hashes = {
        V3_MANIFEST: hashes[V3_MANIFEST],
        V3_RESULT: hashes[V3_RESULT],
        V4_MANIFEST: hashes[V4_MANIFEST],
        V4_RESULT: hashes[V4_RESULT],
    }
    validate_path_hashes(m4, archived_hashes, V4_MANIFEST)
    validate_path_hashes(summary, archived_hashes, V4_SUMMARY)

    for item in frozen_input.get("items", []):
        if item.get("kind") != "candidate":
            continue
        source = item.get("source", {})
        require_hash(source.get("search_manifest_sha256"), hashes[V4_MANIFEST], V4_INPUT)
        require_hash(source.get("search_result_sha256"), hashes[V4_RESULT], V4_INPUT)

    for score in (brown, gpt2, deepseek):
        require_hash(score.get("input_manifest_sha256"), hashes[V4_INPUT], "scorer output")
    require_hash(static_summary.get("input_manifest_sha256"), hashes[V4_INPUT], V4_STATIC_SUMMARY)
    for key, name in (
        ("brown_result_sha256", V4_BROWN),
        ("gpt2_result_sha256", V4_GPT2),
        ("deepseek_result_sha256", V4_DEEPSEEK),
    ):
        require_hash(static_summary.get("inputs", {}).get(key), hashes[name], V4_STATIC_SUMMARY)

    require_hash(summary.get("manifest_sha256"), hashes[V4_MANIFEST], V4_SUMMARY)
    require_hash(summary.get("result_sha256"), hashes[V4_RESULT], V4_SUMMARY)
    static = summary.get("static_verifier", {})
    require_hash(static.get("summary_sha256"), hashes[V4_STATIC_SUMMARY], V4_SUMMARY)
    require_hash(static.get("input_manifest_sha256"), hashes[V4_INPUT], V4_SUMMARY)
    for key, name in (
        ("brown_result_sha256", V4_BROWN),
        ("gpt2_result_sha256", V4_GPT2),
        ("deepseek_result_sha256", V4_DEEPSEEK),
    ):
        require_hash(static.get("inputs", {}).get(key), hashes[name], V4_SUMMARY)
    report = content[V4_REPORT].decode("utf-8")
    if hashes[V4_INPUT] not in report or hashes[V4_STATIC_SUMMARY] not in report:
        raise ValueError("v4 report does not cite the exported input and summary hashes")

    # Source and vocabulary hashes remain useful for exact reproduction.
    for manifest_name in (V3_MANIFEST, V4_MANIFEST):
        manifest = m3 if manifest_name == V3_MANIFEST else m4
        for source_key, source_path in manifest.get("sources", {}).items():
            expected = manifest.get("source_sha256", {}).get(source_key)
            if source_path in content and expected:
                require_hash(sha256(content[source_path]), expected, source_path)


def require_hash(actual: str | None, expected: str, context: str) -> None:
    if actual != expected:
        raise ValueError(f"stale or invalid provenance hash in {context}")


def validate_path_hashes(value: object, hashes_by_path: dict[str, str], context: str) -> None:
    """Check path-addressed historical references for any archived member."""
    if isinstance(value, dict):
        manifest_path = value.get("manifest_path_from_repository_root")
        if manifest_path in hashes_by_path and "manifest_sha256" in value:
            require_hash(value["manifest_sha256"], hashes_by_path[manifest_path], context)
        result_path = value.get("result_path_from_repository_root")
        if result_path in hashes_by_path and "result_sha256" in value:
            require_hash(value["result_sha256"], hashes_by_path[result_path], context)
        for item in value.values():
            validate_path_hashes(item, hashes_by_path, context)
    elif isinstance(value, list):
        for item in value:
            validate_path_hashes(item, hashes_by_path, context)


def validate_privacy(content: dict[str, bytes]) -> None:
    """Screen exported text and metadata without rejecting public model hashes."""
    for name, payload in content.items():
        # This source file necessarily contains its own privacy-pattern literals.
        if name != "paper/build_v4_supplement.py":
            for pattern in PRIVATE_TEXT_PATTERNS:
                if pattern.search(payload):
                    raise ValueError(f"privacy screen rejected {name}: {pattern.pattern.decode('ascii')}")
        if name.endswith(".json"):
            try:
                value = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid JSON archive member: {name}") from exc
            if contains_forbidden_metadata(value):
                raise ValueError(f"privacy screen rejected identifying metadata in {name}")


def contains_forbidden_metadata(value: object) -> bool:
    if isinstance(value, dict):
        if set(value) & ANONYMITY_FIELD_NAMES:
            return True
        return any(contains_forbidden_metadata(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_forbidden_metadata(item) for item in value)
    return False


def zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(2026, 9, 25, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def build(output: Path = DEFAULT_OUTPUT) -> dict:
    missing = [name for name in FILES if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError("supplement inputs missing: " + ", ".join(missing))

    content = {name: (ROOT / name).read_bytes() for name in FILES}
    export_anonymized_json(content)
    validate_privacy(content)
    content["supplement/README.md"] = README.encode("utf-8")
    hashes = "".join(
        f"{sha256(content[name])}  {name}\n" for name in sorted(content)
    ).encode("ascii")
    content["supplement/MANIFEST.sha256"] = hashes

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for name in sorted(content):
            archive.writestr(zip_info(name), content[name])
    verify_archive(output)
    return {
        "output": str(output),
        "member_count": len(content),
        "uncompressed_bytes": sum(map(len, content.values())),
        "sha256": sha256(output.read_bytes()),
        "privacy_screen": "passed",
        "internal_hash_links": "passed",
    }


def verify_archive(path: Path) -> dict:
    """Verify ZIP integrity, per-member hashes, privacy, and linked provenance."""
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("supplement ZIP CRC check failed")
        names = archive.namelist()
        if names != sorted(names):
            raise ValueError("supplement members are not in deterministic sorted order")
        if "supplement/MANIFEST.sha256" not in names:
            raise ValueError("supplement checksum manifest is missing")
        checksum_text = archive.read("supplement/MANIFEST.sha256").decode("ascii")
        expected = {}
        for line in checksum_text.splitlines():
            try:
                digest, name = line.split("  ", 1)
            except ValueError as exc:
                raise ValueError("malformed supplement checksum line") from exc
            if name in expected:
                raise ValueError(f"duplicate checksum entry: {name}")
            expected[name] = digest
        observed_names = set(names) - {"supplement/MANIFEST.sha256"}
        if set(expected) != observed_names:
            raise ValueError("supplement checksum inventory does not match ZIP members")
        payloads = {name: archive.read(name) for name in observed_names}
        for name, digest in expected.items():
            if sha256(payloads[name]) != digest:
                raise ValueError(f"supplement member checksum mismatch: {name}")
        validate_privacy(payloads)
        validate_internal_hashes(payloads)
    return {"member_count": len(names), "checksum_count": len(expected)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    import json
    print(json.dumps(build(args.output), indent=2))


if __name__ == "__main__":
    main()
