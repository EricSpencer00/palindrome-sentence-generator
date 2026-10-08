"""Build the deterministic v1 public supplementary evidence and source archive."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "paper" / "v1-reproducibility.zip"

FILES = (
    "data/known_palindromes.json",
    "data/lexicon.txt",
    "experiments/__init__.py",
    "experiments/DIRECTIONAL-SEARCH-RESOURCE-SENSITIVITY-20260925.md",
    "experiments/LOCAL-LLM-READABILITY-AUDIT-20260926.md",
    "experiments/LOCAL-LLM-SINGLE-SENTENCE-SENSITIVITY-20260926.md",
    "experiments/LOCAL-LLM-READABILITY-AUDIT-RESULTS-20260926.md",
    "experiments/AWS-QWEN3-READABILITY-AUDIT-RESULTS-20260926.md",
    "experiments/READABILITY-LENGTH-CONSTRUCTION-RESULTS-20260926.md",
    "experiments/analyze_readability_length_construction_20260926.py",
    "experiments/aws_qwen3_readability_audit_20260926.py",
    "experiments/summarize_aws_qwen3_readability_audit_20260926.py",
    "experiments/pricing_snapshots/aws-bedrock-qwen3-32b-us-east-2-20260926.json",
    "experiments/DIRECTIONAL-SEARCH-BENCHMARK-20260925.md",
    "experiments/analyze_directional_search_resource_sensitivity_20260925.py",
    "experiments/analyze_paired_directional_search_benchmark_20260925.py",
    "experiments/audit_programmatic_readability.py",
    "experiments/infer_deepseek_static_verifier_20260925.py",
    "experiments/local_llm_readability_audit_20260926.py",
    "experiments/local_llm_single_sentence_brown_sensitivity_20260926.py",
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
    "paper/build_v1_supplement.py",
    "paper/audit_diagnostic_overlap.py",
    "paper/build_diagnostic_figures.py",
    "paper/figure-requirements.txt",
    "paper/fig/diagnostic-landscape.pdf",
    "paper/fig/fragment-concentration.pdf",
    "paper/fig/diagnostic-data.json",
    "paper/BASELINE-PROVENANCE-AUDIT-V1.md",
    "paper/audits/historical-search-baselines-v1.md",
    "paper/audits/historical-construction-evidence-v1.md",
    "paper/audits/historical-metric-controls-v1.md",
    "paper/naacl2027.tex",
    "paper/naacl2027-v1.tex",
    "paper/readability-requirements.txt",
    "paper/refs.bib",
    "paper/human_eval_v4_packet/sealed_key.json",
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
    "runs/local-llm-readability-audit-20260926.json",
    "runs/local-llm-single-sentence-brown-sensitivity-20260926.json",
    "runs/aws-qwen3-32b-readability-audit-20260926.json",
    "runs/readability-length-stratified-20260925.json",
    "runs/readability-length-stratified-sentence-aware-20260925.json",
    "tests/test_initial_state_search.py",
    "tools/polaris/payload/vocab30k.txt",
)

FRAGMENT_DIR = "runs/fragment-penalty-20261002/"
FRAGMENT_MANIFEST = FRAGMENT_DIR + "manifest.json"
FRAGMENT_RESULT = FRAGMENT_DIR + "results.json"
FRAGMENT_QUALITY_MANIFEST = FRAGMENT_DIR + "quality-manifest.json"
FRAGMENT_QUALITY_RESULT = FRAGMENT_DIR + "quality-results.json"
FRAGMENT_ANALYSIS = FRAGMENT_DIR + "analysis.json"
FRAGMENT_JUDGE = "experiments/judge_fragment_penalty_20261002.py"
FRAGMENT_ANALYZER = "experiments/analyze_fragment_penalty_20261002.py"
FILES = tuple(sorted(set(FILES) | set(json.loads((ROOT / FRAGMENT_MANIFEST).read_text())["source_sha256"]) | {
    FRAGMENT_MANIFEST, FRAGMENT_RESULT, FRAGMENT_QUALITY_MANIFEST,
    FRAGMENT_QUALITY_RESULT, FRAGMENT_ANALYSIS, FRAGMENT_JUDGE, FRAGMENT_ANALYZER,
    "experiments/FRAGMENT-PENALTY-RESULTS-20261002.md",
    "tests/test_fragment_experiment_analysis.py",
    "paper/fig/fragment-penalty-tradeoff.pdf", "paper/fig/fragment-penalty-motifs.pdf",
}))

FILES = tuple(sorted(set(FILES) | {
    "experiments/length_cap_probe_20261008.py",
    "experiments/resume_length_cap_probe_20261008.py",
    "experiments/verify_length_cap_probe_20261008.py",
    "experiments/LENGTH-CAP-PROBE-RESULTS-20261008.md",
} | {p.relative_to(ROOT).as_posix() for p in (ROOT / "runs/length-cap-probe-20261008").glob("*") if p.is_file()}))

README = """# v1 supplementary evidence and reproduction

The current manuscript is *Methods of Finding Longer, Readable Palindromes*.
It reports a fresh, frozen 360-job AWS experiment: 20 seeds, three length bands,
two search implementations, and penalties 0, 4, and 16 for six previously
identified trigrams. Each arm returns 90/120 exact outputs. Distinct surfaces
increase from 69 to 78, but strict structural acceptance falls from 59/120 to
28/120 and replacement fragments appear in 46/78 distinct penalized texts.

The new protocol, results, quality inventory and responses, and analysis are in
`runs/fragment-penalty-20261002/`. The readable report is
`experiments/FRAGMENT-PENALTY-RESULTS-20261002.md`. Reproduce the new figures
and 20,000 paired seed-bootstrap replicates offline:

```sh
python3 experiments/analyze_fragment_penalty_20261002.py
```

Install the plotting versions in `paper/figure-requirements.txt`. This analysis
needs no AWS account, model weights, or Brown corpus. The figures describe
seed variability within this vocabulary and search family, not human quality.
All 148 distinct candidates receive a separate blinded Qwen3 model rating.
No human ratings or matched published-generator benchmark is provided.

For fresh search on an authorized Linux worker, use
`experiments/fragment-penalty-requirements.txt`. In a separate copied extraction,
move `runs/fragment-penalty-20261002/results.json` to an archival filename,
then run `python3 experiments/fragment_penalty_20261002.py run --workers 32`.
The runner refuses to overwrite results and verifies its source/vocabulary
hashes. Keep original evidence intact. A new hosted audit additionally needs
an explicitly configured private AWS budget and account binding; do not resume
the archived quality result. Exact prompts, model ID, settings, and requests'
hashes are recorded, but the provider does not expose a random seed.

A separate 16-job development length-cap probe is retained in
`runs/length-cap-probe-20261008/`. Its two 804-letter outputs fail the strict
screen and are not readability successes. The cap32 arm cannot reach 800
letters; its zero yield is not a method-performance baseline. Verify offline
with `python3 experiments/verify_length_cap_probe_20261008.py`. The interrupted
checkpoint and unknown uncheckpointed outcomes are preserved.

The older 240-job resource collection and its 52 distinct outputs are retained
as development evidence. They supplied the six target fragments, not new test
outcomes. Their figures and score-sign audit remain reproducible:

```sh
python3 paper/build_diagnostic_figures.py
python3 paper/audit_diagnostic_overlap.py
```

The separate 52-item first-person packet concerns that historical collection,
not the 148 distinct outputs in the new intervention. The baseline-provenance
audits also concern historical claims; they are not reviews of this new experiment.

Author-specific machine names and repository commit IDs are omitted from the
exported run metadata. Source-file hashes, model revisions, and cross-file
reproduction hashes are retained and checked.

## Historical resource-comparison replay

Use Python 3.10+ and install the package versions recorded in the requirements
files. From the repository root, run:

```sh
python3 experiments/paired_directional_search_resource_sensitivity_20260925.py run \\
  --manifest runs/directional-search-resource-sensitivity-20260925/manifest.json \\
  --phase confirmatory --output /tmp/replayed-v1.json --workers 6
python3 experiments/analyze_directional_search_resource_sensitivity_20260925.py \\
  --manifest runs/directional-search-resource-sensitivity-20260925/manifest.json \\
  --results /tmp/replayed-v1.json \\
  --static-verifier-summary runs/directional-search-resource-sensitivity-20260925/static-verifier-summary-v4-aws.json \\
  --json-output /tmp/v1-summary.json --markdown-output /tmp/v1-summary.md
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
  --output /tmp/brown-v1.json
HF_HOME=/path/to/hf-cache OMP_NUM_THREADS=4 python3 -m experiments.score_gpt2_static_verifier_aws_20260925 \\
  --input runs/directional-search-resource-sensitivity-20260925/static-verifier-input-aws.json \\
  --output /tmp/gpt2-v1.json --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e \\
  --batch-size 8 --execution-host 'authorized CPU host'
python3 experiments/infer_deepseek_static_verifier_20260925.py \\
  --input runs/directional-search-resource-sensitivity-20260925/static-verifier-input-aws.json \\
  --output /tmp/deepseek-v1.json --budget-usd 0.20 --interval-seconds 0.75
python3 experiments/summarize_directional_static_verifier_20260925.py \\
  --input runs/directional-search-resource-sensitivity-20260925/static-verifier-input-aws.json \\
  --brown /tmp/brown-v1.json --gpt2 /tmp/gpt2-v1.json --deepseek /tmp/deepseek-v1.json \\
  --output /tmp/static-summary-v1.json
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

## Qwen 2.5 local-model audit

The archive also contains the frozen Qwen protocols, both runner scripts, the
two saved result JSON files, and the dated results report. These are
success-conditioned diagnostics on the 52 selected candidates. The first
run failed its prespecified Brown-control calibration gate; its candidate
scores are inconclusive. The follow-up is post-hoc, changes control units, and
does not rescore candidates. Neither run provides human ratings or generation
coverage.

The Brown manifest at
`runs/readability-length-stratified-20260925.json` is a privacy-preserving
projection: it retains held-out document IDs, sentence indices, token counts,
and sentence hashes required by the runners, but excludes Brown sentence
surfaces. The licensed Brown corpus itself is not included. Install the
specified NLTK version and Brown corpus locally; the runners reconstruct each
control and verify its text hash before scoring. The sealed candidate key is
rebound to the anonymized study inputs in this archive. The saved Qwen result
files are preserved as historical records and still identify their original
input hashes. Therefore a new run from this archive has a new input fingerprint;
do not try to resume either saved historical run. The exact original and
projected input hashes are listed in `supplement/QWEN-EXPORT-PROVENANCE.json`.

First validate the archive inputs without contacting a model:

```sh
python3 experiments/local_llm_readability_audit_20260926.py --prepare-only
python3 experiments/local_llm_single_sentence_brown_sensitivity_20260926.py --prepare-only
```

To create fresh result records, use the installed exact Qwen model digest in
the saved run metadata and an already available local Ollama endpoint. The
endpoint below is a local forwarded port; configure the SSH destination in
your environment without adding a machine name to the archive:

```sh
python3 experiments/local_llm_readability_audit_20260926.py \\
  --base-url http://127.0.0.1:11435 --output /tmp/qwen-audit-replay.json
python3 experiments/local_llm_single_sentence_brown_sensitivity_20260926.py \\
  --base-url http://127.0.0.1:11435 --output /tmp/qwen-brown-sensitivity-replay.json
```

These commands make new inference requests. The archived JSON files, rather
than a new inference run, are the exact records underlying the reported
results.

## AWS Qwen3 rubric audit

The archive contains the separate Bedrock Qwen3 32B run, source runner,
offline summarizer, pricing snapshot, and results report. The 244 saved
responses can be re-aggregated without model calls or Brown text:

```sh
python3 experiments/summarize_aws_qwen3_readability_audit_20260926.py \\
  --input runs/aws-qwen3-32b-readability-audit-20260926.json
```

This model-specific diagnostic rated all 52 candidates 0/3 on both dimensions;
the candidate and matched-shuffle joint-pass rates were both zero. The small
Brown screen passed 11/12 intact spans and 0/12 shuffles; 7 Brown spans
contained multiple sentences. It provides no human readability estimate or
improvement claim. The archived Brown manifest excludes sentence text; a
licensed local NLTK corpus is required to reconstruct controls from the
protocol. The public archive copy removes AWS request IDs and the credential
and profile identity block. Input-hash transformations are recorded in
`supplement/QWEN-EXPORT-PROVENANCE.json`; the archived historical result is
for offline reanalysis, not for `--resume` or fresh inference. A new Bedrock
run has no provider seed and can return different ratings.

## Archived descriptive cross-tab

`experiments/READABILITY-LENGTH-CONSTRUCTION-RESULTS-20260926.md` reports an
offline stratification of the frozen Brown, GPT-2, and DeepSeek diagnostics
for the 66 selected outputs, plus the 38/54-letter Diana-frame case. It is
descriptive model evidence. Its project-selected length groups establish no
length effect or generation improvement. The 38-letter reference has no
located primary human-rating records. Recreate the
report without network access or inference from the extracted archive root:

```sh
python3 experiments/analyze_readability_length_construction_20260926.py \\
  --output /tmp/readability-length-construction-rerun.md
```
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
QWEN_MAIN_PROTOCOL = "experiments/LOCAL-LLM-READABILITY-AUDIT-20260926.md"
QWEN_FOLLOWUP_PROTOCOL = "experiments/LOCAL-LLM-SINGLE-SENTENCE-SENSITIVITY-20260926.md"
QWEN_REPORT = "experiments/LOCAL-LLM-READABILITY-AUDIT-RESULTS-20260926.md"
QWEN_MAIN_RUNNER = "experiments/local_llm_readability_audit_20260926.py"
QWEN_FOLLOWUP_RUNNER = "experiments/local_llm_single_sentence_brown_sensitivity_20260926.py"
QWEN_MAIN_RESULT = "runs/local-llm-readability-audit-20260926.json"
QWEN_FOLLOWUP_RESULT = "runs/local-llm-single-sentence-brown-sensitivity-20260926.json"
AWS_QWEN_RESULT = "runs/aws-qwen3-32b-readability-audit-20260926.json"
QWEN_PACKET = "paper/human_eval_v4_packet/sealed_key.json"
QWEN_BROWN_SOURCE = "runs/readability-length-stratified-20260925.json"

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

    export_qwen_inputs(content, v4_manifest_sha, v4_result_sha, v4_input_sha)
    validate_internal_hashes(content)


def has_key(value: object, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(has_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(has_key(item, key) for item in value)
    return False


def project_brown_control_manifest(raw: bytes) -> dict:
    """Keep only Brown locations and hashes required to reconstruct controls."""
    source = json.loads(raw.decode("utf-8"))
    method = source.get("method", {})
    projected: dict[str, object] = {
        "export": {
            "purpose": "Qwen control reconstruction without source-text redistribution",
            "original_source_sha256": sha256(raw),
            "excluded": "All Brown sentence surfaces and unrelated scoring rows",
        },
        "method": {
            "heldout_file_count": method.get("heldout_file_count"),
            "heldout_fileid_manifest_sha256": method.get("heldout_fileid_manifest_sha256"),
        },
    }
    required = ("id", "brown_fileid", "sentence_indices_half_open", "surface_sha256")
    for group in ("matched_heldout_prose_controls", "length_stratified_heldout_controls"):
        rows = source.get(group)
        if not isinstance(rows, list):
            raise ValueError(f"Brown source manifest lacks {group}")
        copied = []
        for row in rows:
            missing = [key for key in required if key not in row]
            if missing:
                raise ValueError(f"Brown control is missing required metadata: {missing}")
            copied_row = {key: row[key] for key in required}
            if "source_word_count" in row:
                copied_row["source_word_count"] = row["source_word_count"]
            copied.append(copied_row)
        projected[group] = copied
    if len(projected["matched_heldout_prose_controls"]) != 11:
        raise ValueError("expected the frozen 11 matched Brown control locations")
    if len(projected["length_stratified_heldout_controls"]) != 108:
        raise ValueError("expected the frozen 108 length-stratified Brown controls")
    if has_key(projected, "surface"):
        raise ValueError("Brown projection unexpectedly contains source surfaces")
    return projected


def export_qwen_inputs(
    content: dict[str, bytes], v4_manifest_sha: str,
    v4_result_sha: str, v4_input_sha: str,
) -> None:
    """Rebind runnable Qwen inputs while preserving historical result records."""
    original_brown = (ROOT / QWEN_BROWN_SOURCE).read_bytes()
    original_packet = (ROOT / QWEN_PACKET).read_bytes()
    original_static = (ROOT / V4_INPUT).read_bytes()
    original_main_result = (ROOT / QWEN_MAIN_RESULT).read_bytes()
    original_followup_result = (ROOT / QWEN_FOLLOWUP_RESULT).read_bytes()
    original_aws_result = (ROOT / AWS_QWEN_RESULT).read_bytes()

    # The Qwen runner reads this same repo-relative path. Its corpus prose is
    # reconstructed from the user's installed NLTK Brown corpus and hash-checked.
    brown_projection = project_brown_control_manifest(original_brown)
    store_json(content, QWEN_BROWN_SOURCE, brown_projection)
    brown_projection_sha = sha256(content[QWEN_BROWN_SOURCE])

    # Candidate strings are unchanged, but their source packet must point to
    # the anonymized study inputs packed beside it.
    packet = parse_json(content, QWEN_PACKET)
    packet["input_manifest_sha256"] = v4_manifest_sha
    packet["search_result_sha256"] = v4_result_sha
    packet["source_input_sha256"] = v4_input_sha
    for candidate in packet.get("candidates", []):
        for source in candidate.get("source_rows", []):
            if "search_manifest_sha256" in source:
                source["search_manifest_sha256"] = v4_manifest_sha
            if "search_result_sha256" in source:
                source["search_result_sha256"] = v4_result_sha
    store_json(content, QWEN_PACKET, packet)

    # These are historical outputs. Their original source hashes and fingerprints
    # remain untouched, so an archive rerun gets a distinct, honest fingerprint.
    if content[QWEN_MAIN_RESULT] != original_main_result:
        raise ValueError("Qwen main result was unexpectedly rewritten during export")
    if content[QWEN_FOLLOWUP_RESULT] != original_followup_result:
        raise ValueError("Qwen follow-up result was unexpectedly rewritten during export")

    # Keep the hosted result and analysis but remove account/profile and
    # request IDs from the public archive copy.
    aws_result = parse_json(content, AWS_QWEN_RESULT)
    aws_result.pop("runtime_environment", None)
    aws_result.pop("runtime_environment_checks", None)
    for response in aws_result.get("responses", []):
        response.pop("aws_request_id", None)
        raw = response.get("raw_response", {})
        metadata = raw.get("response_metadata", {}) if isinstance(raw, dict) else {}
        if isinstance(metadata, dict):
            metadata.pop("request_id", None)
    store_json(content, AWS_QWEN_RESULT, aws_result)
    aws_archive_result_sha = sha256(content[AWS_QWEN_RESULT])

    qwen3_report = content["experiments/AWS-QWEN3-READABILITY-AUDIT-RESULTS-20260926.md"].decode("utf-8")
    qwen3_report = qwen3_report.replace(sha256(original_aws_result), aws_archive_result_sha)
    qwen3_report += (
        "\n\nFor the public archive copy, run the offline summarizer without `--verify-sources`; "
        "the archived source packet and Brown manifest are projected copies. Their exact "
        "original-to-archive hashes are recorded in `supplement/QWEN-EXPORT-PROVENANCE.json`.\n"
    )
    content["experiments/AWS-QWEN3-READABILITY-AUDIT-RESULTS-20260926.md"] = qwen3_report.encode("utf-8")

    content["supplement/QWEN-EXPORT-PROVENANCE.json"] = json_bytes({
        "schema_version": 1,
        "historical_results_preserved_byte_for_byte": True,
        "historical_result_sha256": {
            QWEN_MAIN_RESULT: sha256(original_main_result),
            QWEN_FOLLOWUP_RESULT: sha256(original_followup_result),
        },
        "historical_input_sha256": {
            "candidate_packet": sha256(original_packet),
            "static_verifier_input": sha256(original_static),
            "full_brown_control_manifest": sha256(original_brown),
        },
        "archive_input_sha256": {
            "candidate_packet": sha256(content[QWEN_PACKET]),
            "static_verifier_input": v4_input_sha,
            "brown_control_manifest_projection": brown_projection_sha,
        },
        "brown_projection": {
            "repo_relative_path": QWEN_BROWN_SOURCE,
            "retained": ["Brown file IDs", "sentence indices", "word counts", "surface SHA-256"],
            "omitted": "Brown sentence text and unrelated scoring rows",
            "reconstruction": "Requires a local licensed NLTK Brown corpus; runner verifies each reconstructed surface hash.",
        },
        "rerun_note": (
            "A fresh archive run uses projected inputs and therefore receives a new run fingerprint. "
            "Do not resume the preserved historical result files."
        ),
        "aws_qwen3_result": {
            "repo_relative_path": AWS_QWEN_RESULT,
            "original_result_sha256": sha256(original_aws_result),
            "archive_result_sha256": aws_archive_result_sha,
            "redacted_only_in_archive_copy": [
                "runtime_environment",
                "runtime_environment_checks",
                "responses[].aws_request_id",
                "responses[].raw_response.response_metadata.request_id",
            ],
            "original_source_input_sha256": {
                key: parse_json({AWS_QWEN_RESULT: original_aws_result}, AWS_QWEN_RESULT)
                    .get("source_artifacts", {}).get(key)
                for key in (
                    "candidate_packet_sha256",
                    "static_verifier_input_sha256",
                    "heldout_brown_control_manifest_sha256",
                )
            },
            "archive_source_input_sha256": {
                QWEN_PACKET: sha256(content[QWEN_PACKET]),
                V4_INPUT: sha256(content[V4_INPUT]),
                QWEN_BROWN_SOURCE: brown_projection_sha,
            },
            "historical_result_reaggregatable_offline": True,
        },
    })


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

    validate_qwen_archive(content, hashes)


def validate_qwen_archive(content: dict[str, bytes], hashes: dict[str, str]) -> None:
    """Check projected rerun inputs against immutable historical Qwen records."""
    provenance = parse_json(content, "supplement/QWEN-EXPORT-PROVENANCE.json")
    packet = parse_json(content, QWEN_PACKET)
    brown_projection = parse_json(content, QWEN_BROWN_SOURCE)
    main_result = parse_json(content, QWEN_MAIN_RESULT)
    followup_result = parse_json(content, QWEN_FOLLOWUP_RESULT)
    aws_result = parse_json(content, AWS_QWEN_RESULT)
    historical_results = provenance.get("historical_result_sha256", {})
    historical_inputs = provenance.get("historical_input_sha256", {})
    archive_inputs = provenance.get("archive_input_sha256", {})

    for result_path, result_hash in historical_results.items():
        require_hash(result_hash, hashes.get(result_path, ""), result_path)
    require_hash(
        historical_inputs.get("candidate_packet"),
        main_result.get("source_artifacts", {}).get("candidate_packet_sha256"),
        QWEN_MAIN_RESULT,
    )
    require_hash(
        historical_inputs.get("static_verifier_input"),
        main_result.get("source_artifacts", {}).get("static_verifier_input_sha256"),
        QWEN_MAIN_RESULT,
    )
    require_hash(
        historical_inputs.get("full_brown_control_manifest"),
        main_result.get("source_artifacts", {}).get("heldout_brown_control_manifest_sha256"),
        QWEN_MAIN_RESULT,
    )
    require_hash(
        brown_projection.get("export", {}).get("original_source_sha256"),
        historical_inputs.get("full_brown_control_manifest"),
        QWEN_BROWN_SOURCE,
    )
    require_hash(
        followup_result.get("method", {}).get("brown_source", {}).get("length_control_manifest_sha256"),
        historical_inputs.get("full_brown_control_manifest"),
        QWEN_FOLLOWUP_RESULT,
    )
    require_hash(archive_inputs.get("candidate_packet"), hashes[QWEN_PACKET], QWEN_PACKET)
    require_hash(archive_inputs.get("static_verifier_input"), hashes[V4_INPUT], V4_INPUT)
    require_hash(archive_inputs.get("brown_control_manifest_projection"), hashes[QWEN_BROWN_SOURCE], QWEN_BROWN_SOURCE)
    require_hash(packet.get("input_manifest_sha256"), hashes[V4_MANIFEST], QWEN_PACKET)
    require_hash(packet.get("search_result_sha256"), hashes[V4_RESULT], QWEN_PACKET)
    require_hash(packet.get("source_input_sha256"), hashes[V4_INPUT], QWEN_PACKET)

    require_hash(
        main_result.get("source_artifacts", {}).get("scorer_script_sha256"),
        hashes[QWEN_MAIN_RUNNER],
        QWEN_MAIN_RUNNER,
    )
    require_hash(
        followup_result.get("source_artifacts", {}).get("scorer_script_sha256"),
        hashes[QWEN_FOLLOWUP_RUNNER],
        QWEN_FOLLOWUP_RUNNER,
    )
    require_hash(
        followup_result.get("method", {}).get("first_run_provenance_hashes", {}).get("first_run_result_sha256"),
        historical_results.get(QWEN_MAIN_RESULT),
        QWEN_FOLLOWUP_RESULT,
    )
    require_hash(
        followup_result.get("method", {}).get("first_run_provenance_hashes", {}).get("first_run_scorer_sha256"),
        hashes[QWEN_MAIN_RUNNER],
        QWEN_FOLLOWUP_RESULT,
    )
    main_digest = main_result.get("model_facts", {}).get("model_digest")
    require_hash(
        followup_result.get("model_facts", {}).get("model_digest"),
        main_digest,
        QWEN_FOLLOWUP_RESULT,
    )
    if has_key(brown_projection, "surface") or has_key(brown_projection, "text"):
        raise ValueError("Brown archive projection contains source text")
    if provenance.get("historical_results_preserved_byte_for_byte") is not True:
        raise ValueError("Qwen historical result preservation is not attested")

    aws_export = provenance.get("aws_qwen3_result", {})
    require_hash(aws_export.get("archive_result_sha256"), hashes[AWS_QWEN_RESULT], AWS_QWEN_RESULT)
    if aws_export.get("historical_result_reaggregatable_offline") is not True:
        raise ValueError("AWS Qwen historical analysis is not attested as reaggregatable")
    original_sources = aws_export.get("original_source_input_sha256", {})
    for key in (
        "candidate_packet_sha256",
        "static_verifier_input_sha256",
        "heldout_brown_control_manifest_sha256",
    ):
        require_hash(
            original_sources.get(key),
            aws_result.get("source_artifacts", {}).get(key),
            AWS_QWEN_RESULT,
        )
    archive_sources = aws_export.get("archive_source_input_sha256", {})
    for name in (QWEN_PACKET, V4_INPUT, QWEN_BROWN_SOURCE):
        require_hash(archive_sources.get(name), hashes[name], AWS_QWEN_RESULT)
    if aws_result.get("status") != "complete" or len(aws_result.get("presentations", [])) != 244:
        raise ValueError("AWS Qwen archived run is incomplete")
    if aws_result.get("runtime_environment") or aws_result.get("runtime_environment_checks"):
        raise ValueError("AWS Qwen archive retained credential/profile identity metadata")
    for response in aws_result.get("responses", []):
        if "aws_request_id" in response:
            raise ValueError("AWS Qwen archive retained an AWS request ID")
        metadata = response.get("raw_response", {}).get("response_metadata", {})
        if isinstance(metadata, dict) and "request_id" in metadata:
            raise ValueError("AWS Qwen archive retained a provider request ID")
    sys.path.insert(0, str(ROOT / "experiments"))
    from local_llm_readability_audit_20260926 import compute_analysis
    if compute_analysis(aws_result["presentations"], aws_result["responses"]) != aws_result.get("analysis"):
        raise ValueError("AWS Qwen archived analysis does not reproduce from saved responses")


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
        if name != "paper/build_v1_supplement.py":
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
    export_fragment_experiment(content)
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
        validate_fragment_experiment(payloads)
    return {"member_count": len(names), "checksum_count": len(expected)}


def export_fragment_experiment(content: dict[str, bytes]) -> None:
    """Anonymous derivative, preserving every experimental outcome and rating."""
    original_hashes = {name: sha256(content[name]) for name in (
        FRAGMENT_MANIFEST, FRAGMENT_RESULT, FRAGMENT_QUALITY_MANIFEST,
        FRAGMENT_QUALITY_RESULT, FRAGMENT_ANALYSIS, FRAGMENT_JUDGE,
    )}
    # Make the exported runner portable without changing the raw frozen source.
    content[FRAGMENT_JUDGE] = re.sub(
        rb'^AWS = "/Users/[^"\n]+/aws"$', b'AWS = os.environ.get("AWS_CLI", "aws")',
        content[FRAGMENT_JUDGE], flags=re.MULTILINE)
    report = "experiments/FRAGMENT-PENALTY-RESULTS-20261002.md"
    content[report] = re.sub(rb'committed as `[0-9a-f]{8,40}`', b'frozen', content[report])
    manifest = parse_json(content, FRAGMENT_MANIFEST)
    manifest["source_sha256"] = {path: sha256(content[path]) for path in manifest["source_sha256"]}
    store_json(content, FRAGMENT_MANIFEST, manifest)
    result = parse_json(content, FRAGMENT_RESULT)
    result["manifest_sha256"] = sha256(content[FRAGMENT_MANIFEST])
    store_json(content, FRAGMENT_RESULT, result)
    quality_manifest = parse_json(content, FRAGMENT_QUALITY_MANIFEST)
    for key, path in (("search_manifest_sha256", FRAGMENT_MANIFEST),
                      ("search_results_sha256", FRAGMENT_RESULT),
                      ("prior_audit_sha256", AWS_QWEN_RESULT),
                      ("script_sha256", FRAGMENT_JUDGE)):
        quality_manifest[key] = sha256(content[path])
    store_json(content, FRAGMENT_QUALITY_MANIFEST, quality_manifest)
    quality = parse_json(content, FRAGMENT_QUALITY_RESULT)
    quality["manifest_sha256"] = sha256(content[FRAGMENT_QUALITY_MANIFEST])
    for response in quality["responses"]:
        response.get("response", {}).pop("ResponseMetadata", None)
    store_json(content, FRAGMENT_QUALITY_RESULT, quality)
    analysis = parse_json(content, FRAGMENT_ANALYSIS)
    analysis["manifest_sha256"] = sha256(content[FRAGMENT_MANIFEST])
    analysis["search_results_sha256"] = sha256(content[FRAGMENT_RESULT])
    analysis["quality"]["source_sha256"] = sha256(content[FRAGMENT_QUALITY_RESULT])
    store_json(content, FRAGMENT_ANALYSIS, analysis)
    provenance = parse_json(content, "supplement/QWEN-EXPORT-PROVENANCE.json")
    provenance["fragment_experiment"] = {
        "original_sha256": original_hashes,
        "export_sha256": {name: sha256(content[name]) for name in original_hashes},
        "changes": "Portable AWS executable lookup; rebind hashes to anonymized inputs; omit response transport metadata if present. Search outcomes, candidate texts, model output, usage and ratings are unchanged. Account and infrastructure records and Brown corpus text are not included.",
        "fresh_inference": "Saved records are for offline reanalysis. A new run requires a separate authorized AWS budget and fresh result paths; do not resume the archived audit.",
    }
    store_json(content, "supplement/QWEN-EXPORT-PROVENANCE.json", provenance)
    validate_fragment_experiment(content)


def validate_fragment_experiment(content: dict[str, bytes]) -> None:
    manifest = parse_json(content, FRAGMENT_MANIFEST)
    result = parse_json(content, FRAGMENT_RESULT)
    qm = parse_json(content, FRAGMENT_QUALITY_MANIFEST)
    q = parse_json(content, FRAGMENT_QUALITY_RESULT)
    analysis = parse_json(content, FRAGMENT_ANALYSIS)
    for path, expected in manifest["source_sha256"].items():
        require_hash(expected, sha256(content[path]), path)
    for obj, key, path in ((result,"manifest_sha256",FRAGMENT_MANIFEST),
                           (qm,"search_manifest_sha256",FRAGMENT_MANIFEST),
                           (qm,"search_results_sha256",FRAGMENT_RESULT),
                           (qm,"script_sha256",FRAGMENT_JUDGE),
                           (q,"manifest_sha256",FRAGMENT_QUALITY_MANIFEST),
                           (analysis,"search_results_sha256",FRAGMENT_RESULT),
                           (analysis,"manifest_sha256",FRAGMENT_MANIFEST),
                           (analysis["quality"],"source_sha256",FRAGMENT_QUALITY_RESULT)):
        require_hash(obj[key], sha256(content[path]), path)
    if result['status'] != 'complete' or len(result['results']) != 360:
        raise ValueError('incomplete fragment experiment')
    if q['status'] != 'complete' or len(q['responses']) != 184:
        raise ValueError('incomplete fragment quality audit')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    import json
    print(json.dumps(build(args.output), indent=2))


if __name__ == "__main__":
    main()
