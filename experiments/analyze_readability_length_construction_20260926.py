#!/usr/bin/env python3
"""Generate a sourced, offline length-by-construction readability analysis."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "experiments" / "READABILITY-LENGTH-CONSTRUCTION-RESULTS-20260926.md"
READABILITY_RUN = "runs/readability-length-stratified-sentence-aware-20260925.json"
V4_MANIFEST = "runs/directional-search-resource-sensitivity-20260925/manifest.json"
V4_RESULTS = "runs/directional-search-resource-sensitivity-20260925/confirmatory-v4-resource-sensitivity.json"
V4_SCORES = "runs/directional-search-resource-sensitivity-20260925/static-verifier-summary-v4-aws.json"
SOURCES = (READABILITY_RUN, V4_MANIFEST, V4_RESULTS, V4_SCORES)

METHOD_LABELS = {
    "center_out_project": "center-out",
    "outside_in_norvig_hoey_adaptation": "outside-in adaptation",
}
BAND_LABELS = {1: "30--49", 2: "50--79", 3: "80--119"}


def read_json(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def sha256_file(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def packaged_hashes() -> dict[str, str]:
    """Read the hashes of the exact public archive members when available."""
    manifest_path = ROOT / "supplement" / "MANIFEST.sha256"
    if manifest_path.is_file():
        manifest_text = manifest_path.read_text(encoding="utf-8")
    else:
        archive_path = ROOT / "paper" / "v1-reproducibility.zip"
        if not archive_path.is_file():
            raise FileNotFoundError(
                "run this from the repository root or an extracted reproducibility archive"
            )
        with ZipFile(archive_path) as archive:
            manifest_text = archive.read("supplement/MANIFEST.sha256").decode("utf-8")

    hashes: dict[str, str] = {}
    for line in manifest_text.splitlines():
        digest, relative = line.split("  ", maxsplit=1)
        hashes[relative] = digest
    return hashes


def signed(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}"


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot summarize an empty group")
    return statistics.mean(values)


def build_report() -> str:
    audit = read_json(READABILITY_RUN)
    manifest = read_json(V4_MANIFEST)
    raw_results = read_json(V4_RESULTS)
    verifier = read_json(V4_SCORES)
    scores = verifier["scores"]

    if audit.get("status") != "programmatic_diagnostic_not_human_readability_result":
        raise ValueError("the historical readability diagnostic status changed")
    if raw_results.get("status") != "complete":
        raise ValueError("the v4 result run is not marked complete")
    if len(audit.get("candidates", [])) != 11 or len(scores) != 66:
        raise ValueError("expected the frozen 11-text and 66-output audits")
    if len(raw_results.get("results", [])) != 240 or raw_results.get("task_count") != 240:
        raise ValueError("expected all 240 frozen v4 construction tasks")
    if len({row["id"] for row in scores}) != 66:
        raise ValueError("v4 static-verifier output IDs are not unique")

    controls = {
        row["matched_candidate_id"]: row
        for row in audit["matched_heldout_prose_controls"]
    }
    case_rows = {row["id"]: row for row in audit["candidates"]}
    expected_case_ids = {"38-control", "54-np-candidate"}
    if not expected_case_ids <= case_rows.keys() or not expected_case_ids <= controls.keys():
        raise ValueError("the frozen Diana frame pair or its prose controls are missing")

    band_bounds: dict[int, set[tuple[int, int]]] = {}
    for row in raw_results["results"]:
        if row.get("phase") == "confirmatory" and row.get("resource_configuration") == "escalated_v4":
            band_bounds.setdefault(row["band_id"], set()).add((row["min_letters"], row["max_letters"]))
    if any(len(bounds) != 1 for bounds in band_bounds.values()):
        raise ValueError("length-band definitions vary across frozen tasks")

    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in scores:
        grouped.setdefault((row["band_id"], row["method"]), []).append(row)
    expected_methods = set(METHOD_LABELS)
    for band in BAND_LABELS:
        for method in expected_methods:
            if len(grouped.get((band, method), [])) != 11:
                raise ValueError(f"expected 11 successful outputs for band {band}, method {method}")

    construction_groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in raw_results["results"]:
        key = (row["resource_configuration"], row["band_id"], row["method"])
        construction_groups.setdefault(key, []).append(row)
    construction_lines: list[str] = []
    for band in sorted(BAND_LABELS):
        for method in ("center_out_project", "outside_in_norvig_hoey_adaptation"):
            cell_results = {
                config: construction_groups.get((config, band, method), [])
                for config in ("original_v3", "escalated_v4")
            }
            for config, rows in cell_results.items():
                if len(rows) != 20:
                    raise ValueError(f"expected 20 tasks for {config}, band {band}, method {method}")
            escalated_successes = [
                row for row in cell_results["escalated_v4"]
                if row["status"] == "selected_basic_valid_output"
            ]
            actual_lengths = [row["letters"] for row in escalated_successes]
            if len(actual_lengths) != 11 or any(length <= 0 for length in actual_lengths):
                raise ValueError(f"expected 11 positive-length escalated outputs for band {band}, method {method}")
            original_successes = [
                row for row in cell_results["original_v3"]
                if row["status"] == "selected_basic_valid_output"
            ]
            construction_lines.append(
                f"| {BAND_LABELS[band]} | {METHOD_LABELS[method]} | "
                f"{len(original_successes)}/20 | {len(escalated_successes)}/20 "
                f"({100 * len(escalated_successes) / 20:.0f}%) | "
                f"{min(actual_lengths)}/{statistics.median(actual_lengths):g}/{max(actual_lengths)} |"
            )

    raw_selected = {
        (row["seed"], row["band_id"], row["resource_configuration"], row["method"]): row
        for row in raw_results["results"]
        if row["status"] == "selected_basic_valid_output"
    }
    scored_keys = {
        (row["seed"], row["band_id"], row["resource_configuration"], row["method"]): row
        for row in scores
    }
    if raw_selected.keys() != scored_keys.keys():
        raise ValueError("the 66 scored outputs do not match the 66 selected basic-valid outputs")
    for key, scored in scored_keys.items():
        raw = raw_selected[key]
        if raw["text_sha256"] != scored["text_sha256"] or raw["letters"] != scored["letters"]:
            raise ValueError(f"static-verifier row does not match its raw output: {key}")
    original_selected_count = sum(
        row["status"] == "selected_basic_valid_output"
        for row in raw_results["results"]
        if row["resource_configuration"] == "original_v3"
    )
    escalated_selected_count = sum(
        row["status"] == "selected_basic_valid_output"
        for row in raw_results["results"]
        if row["resource_configuration"] == "escalated_v4"
    )
    if original_selected_count + escalated_selected_count != len(raw_selected):
        raise ValueError("raw task outcome totals do not match the selected-output records")

    band_summaries: dict[tuple[int, str], dict[str, Any]] = {}
    band_lines: list[str] = []
    for band in sorted(BAND_LABELS):
        for method in ("center_out_project", "outside_in_norvig_hoey_adaptation"):
            rows = grouped[(band, method)]
            lengths = [row["letters"] for row in rows]
            brown = [row["brown_order_gain_nats_per_transition"] for row in rows]
            gpt2 = [row["gpt2_candidate_minus_shuffle"] for row in rows]
            deepseek = Counter(row["deepseek_candidate_vs_shuffle"] for row in rows)
            summary = {
                "rows": rows,
                "actual_length_range": (min(lengths), statistics.median(lengths), max(lengths)),
                "brown_mean": mean(brown),
                "brown_positive": sum(value > 0 for value in brown),
                "gpt2_mean": mean(gpt2),
                "gpt2_positive": sum(value > 0 for value in gpt2),
                "deepseek": deepseek,
            }
            band_summaries[(band, method)] = summary
            low, median, high = summary["actual_length_range"]
            deepseek_text = "/".join(str(deepseek.get(k, 0)) for k in ("candidate", "shuffle", "tie"))
            band_lines.append(
                f"| {BAND_LABELS[band]} | {METHOD_LABELS[method]} | 11 | "
                f"{low}/{median:g}/{high} | "
                f"{signed(summary['brown_mean'])} ({summary['brown_positive']}/11) | "
                f"{signed(summary['gpt2_mean'])} ({summary['gpt2_positive']}/11) | {deepseek_text} |"
            )

    pair_lines: list[str] = []
    for band in sorted(BAND_LABELS):
        outside = {row["seed"]: row for row in grouped[(band, "outside_in_norvig_hoey_adaptation")]}
        center = {row["seed"]: row for row in grouped[(band, "center_out_project")]}
        paired_seeds = sorted(outside.keys() & center.keys())
        if len(paired_seeds) != 11:
            raise ValueError(f"expected 11 same-seed construction pairs in band {band}")
        brown_diffs = [
            outside[seed]["brown_order_gain_nats_per_transition"]
            - center[seed]["brown_order_gain_nats_per_transition"]
            for seed in paired_seeds
        ]
        gpt2_diffs = [
            outside[seed]["gpt2_candidate_minus_shuffle"]
            - center[seed]["gpt2_candidate_minus_shuffle"]
            for seed in paired_seeds
        ]
        pair_lines.append(
            f"| {BAND_LABELS[band]} | 11 | {signed(mean(brown_diffs))} "
            f"({sum(value > 0 for value in brown_diffs)}/11 positive pairs) | "
            f"{signed(mean(gpt2_diffs))} ({sum(value > 0 for value in gpt2_diffs)}/11 positive pairs) |"
        )

    diana_lines: list[str] = []
    for case_id in ("38-control", "54-np-candidate"):
        candidate = case_rows[case_id]
        matched = controls[case_id]
        own_gain = candidate["brown_order_gain_vs_own_shuffle"]
        prose_gain = matched["brown_order_gain_vs_own_shuffle"]
        scorer_tokens = candidate["scorer_tokens"]
        tokens = Counter(re.findall(r"[A-Za-z]+", candidate["surface"].lower()))
        candidate_minus_control = own_gain - prose_gain
        diana_lines.append(
            f"| {case_id} | {candidate['letters']} | {scorer_tokens} | {candidate['sentence_count']} | "
            f"{len(tokens)} | {signed(own_gain, 4)} | {signed(prose_gain, 4)} | "
            f"{signed(candidate_minus_control, 4)} | {candidate['mechanism']} |"
        )

    short_case = case_rows["38-control"]
    long_case = case_rows["54-np-candidate"]
    case_length_delta = long_case["letters"] - short_case["letters"]
    case_word_delta = long_case["scorer_tokens"] - short_case["scorer_tokens"]
    case_sentence_delta = long_case["sentence_count"] - short_case["sentence_count"]
    diana_control_gap = (
        controls["54-np-candidate"]["brown_order_gain_vs_own_shuffle"]
        - case_rows["54-np-candidate"]["brown_order_gain_vs_own_shuffle"]
    )
    packaged = packaged_hashes()
    report_sources = (*SOURCES, "experiments/analyze_readability_length_construction_20260926.py")
    missing_packaged = set(report_sources) - packaged.keys()
    if missing_packaged:
        raise ValueError(f"source files missing from archive manifest: {sorted(missing_packaged)}")
    source_hash_lines = [
        f"| `{path}` | `{sha256_file(path)}` | `{packaged[path]}` |" for path in report_sources
    ]

    return f"""# Readability, length, and construction results

**Date:** 2026-09-26  
**Status:** offline descriptive analysis of frozen results; no model calls or AWS spend.

This report combines the sourced **Diana frame** 38/54-letter construction
case with frozen v4 language diagnostics, grouped by target length and search
construction strategy.

## Diana frame: more structure, but no human readability result

The 54-letter candidate preserves the inherited “aide / rips / nine / memos;
some / men / inspire / Diana” frame and adds noun-phrase material. It is a
bounded live-residual grammar construction; the 38-letter comparator is an
inherited reference whose authorship is unknown. Both are exact letter-level
palindromes. The 54-letter text is {case_length_delta} normalized letters,
{case_word_delta} scorer tokens, and {case_sentence_delta} sentence longer.
Scorer tokenization splits hyphenated compounds into alphabetic tokens. The
exact texts are:

- “{short_case['surface']}”
- “{long_case['surface']}”

| Text | Letters | Scorer tokens | Sentences | Unique scorer tokens | Candidate Brown gain | Matched Brown prose gain | Candidate minus prose | Construction provenance |
|---|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(diana_lines)}

Brown gain is mean log probability minus the mean for 32 same-word shuffles,
in nats per word transition. Both candidates beat their own shuffled baseline.
The prose controls match scorer-token count (9 and 13 tokens) and each has one
sentence; the 54-letter construction has two. The 38-letter reference has
greater Brown order gain than its matched prose control. The 54-letter
construction has {abs(diana_control_gap):.4f} fewer nats per transition than
its matched prose control. This is one inherited reference and one extension,
so it cannot show that added length or the construction caused a readability
change. These texts were not among the separate 52 candidates in the Qwen3
audit; its 0/3 ratings do not apply to them. Neither has project human ratings.

## V4 diagnostics by length and construction strategy

The 240-task run contains 20 seeds per resource × length × strategy cell.
The original-v3 resource setting selected {original_selected_count}/120
basic-valid outputs; escalated-v4 selected {escalated_selected_count}/120
({100 * escalated_selected_count / 120:.0f}%). Every escalated-v4
length × strategy cell selected 11/20. These are construction yields under
the study's basic validity checks, not grammaticality or readability ratings.

| Target letters | Strategy | Original-v3 selected valid | Escalated-v4 selected valid | Escalated actual letters min/median/max |
|---|---|---:|---:|---:|
{chr(10).join(construction_lines)}

The 66 static-verifier records were joined to raw construction outputs by
seed, length band, resource setting, and strategy. The analyzer checks that
text hashes and letter counts match. For the readability diagnostics below,
each cell contains 11 successful outputs from the 20-seed frozen v4 task set.
“Actual letters” gives minimum/median/maximum normalized letter counts within
the target band. Brown and GPT-2 columns give candidate-minus-own-shuffle
diagnostics; a positive count means the candidate exceeded its own shuffle
mean. DeepSeek columns count candidate/shuffle/tie choices from one pairwise
judgment per output.

Brown is the add-0.1 word-bigram diagnostic trained on 80% of Brown corpus
documents, with candidates held out. It regenerates 32 item-keyed word-order
shuffles. GPT-2 uses the fixed `openai-community/gpt2` revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e` and scores each candidate against
32 archived same-word shuffles. DeepSeek V3.2 made one counterbalanced
candidate-versus-shuffle judgment per output (Bedrock `us-east-2`, temperature
0, top-p 1); a new request may differ because the provider supplies no seed.
These are different automatic diagnostics and are not combined into a single
readability score. The frozen v4 manifest and scorer code identify the exact
protocol and inputs.

| Target letters | Strategy | n | Actual letters min/median/max | Brown gain, mean (positive) | GPT-2 diff, mean (positive) | DeepSeek candidate/shuffle/tie |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(band_lines)}

Paired within-seed contrasts compare outside-in minus center-out for the 11
seeds in each band where both methods produced an output. They are descriptive
subgroup summaries; no band-wise significance test was prespecified.

| Target letters | Paired seeds | Brown difference, mean (positive pairs) | GPT-2 difference, mean (positive pairs) |
|---|---:|---:|---:|
{chr(10).join(pair_lines)}

In the shortest band, all 11 outside-in outputs stop at the 49-letter ceiling.
The strategy comparison is not stable across diagnostics: Brown favors
outside-in on average in the middle and longest bands, GPT-2 does not show the
same pattern, and DeepSeek ties all 66 candidate/shuffle comparisons. Brown
positive counts for center-out fall from 10/11 in the shortest band to 2/11 in
the longest; outside-in counts are 10/11, 7/11, and 8/11. These are selected,
success-conditioned model diagnostics, not reader ratings or causal effects
of length or construction strategy.

## Reproduction and source hashes

From the repository root, regenerate this report without network access or
inference:

```sh
python3 experiments/analyze_readability_length_construction_20260926.py \\
  --output /tmp/readability-length-construction-rerun.md
```

The analyzer validates the frozen sample sizes, length bands, same-seed
pairs, and Diana candidate/control IDs before writing the report. Run it from
the repository root or the extracted reproducibility archive. The current
checkout hash is shown beside the hash of the corresponding public archive
member; some archived JSON files are privacy-projected, as documented in the
supplement manifest. These hashes identify the exact inputs used in each
location; reanalysis from the archive reproduces the reported numeric results.

| Source | Current checkout SHA-256 | Public archive member SHA-256 |
|---|---|---|
{chr(10).join(source_hash_lines)}

The v4 outputs and Brown/GPT-2/DeepSeek diagnostics are from the same frozen
66-output audit used in the manuscript. The Diana pair is a separate
retrospective case series. No score here is a human readability scale.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    report = build_report()
    output.write_text(report, encoding="utf-8")
    print(f"wrote {output}")
    print(f"sha256 {hashlib.sha256(report.encode('utf-8')).hexdigest()}")
    print("model_calls 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
