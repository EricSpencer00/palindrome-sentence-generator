"""Validate and summarize the frozen v3 directional-search experiment."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.paired_directional_search_benchmark_20260925 import (  # noqa: E402
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    EXPERIMENT,
    METHODS,
    sha256_file,
    verify_frozen_manifest,
)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(manifest_path: Path, result_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    verify_frozen_manifest(manifest)
    result = json.loads(result_path.read_text())
    manifest_hash = sha256_file(manifest_path)
    if result.get("experiment") != EXPERIMENT or result.get("phase") != "confirmatory":
        raise ValueError("the supplied result is not the v3 confirmatory run")
    if result.get("protocol_manifest_sha256") != manifest_hash:
        raise ValueError("result was not produced from the supplied frozen manifest")

    expected_jobs = {
        (row["pair_id"], row["method"]): row
        for row in manifest["jobs"] if row["phase"] == "confirmatory"
    }
    rows = result["results"]
    observed_keys = [(row["pair_id"], row["method"]) for row in rows]
    if len(observed_keys) != len(set(observed_keys)) or set(observed_keys) != set(expected_jobs):
        raise ValueError("result rows do not match the frozen confirmatory jobs exactly")

    for row in rows:
        expected = expected_jobs[(row["pair_id"], row["method"])]
        for key in ("seed", "band_id", "min_letters", "max_letters", "execution_order_within_pair"):
            if row.get(key) != expected[key]:
                raise ValueError(f"result differs from frozen job on {key}: {row['pair_id']}")
        if row["text"]:
            tape = "".join(ch.lower() for ch in row["text"] if ch.isascii() and ch.isalpha())
            if tape != tape[::-1] or hashlib.sha256(row["text"].encode()).hexdigest() != row["text_sha256"]:
                raise ValueError(f"selected output exactness/hash check failed: {row['pair_id']}")
            if not all(row["basic_structural_checks"].values()):
                raise ValueError(f"selected output failed a basic admission check: {row['pair_id']}")
        elif row["text_sha256"] is not None:
            raise ValueError(f"empty result carries a text hash: {row['pair_id']}")

    by_method = defaultdict(list)
    by_band_method = defaultdict(list)
    by_pair = defaultdict(dict)
    for row in rows:
        by_method[row["method"]].append(row)
        by_band_method[(row["band_id"], row["method"])].append(row)
        by_pair[row["pair_id"]][row["method"]] = row

    method_summary = {}
    for method in METHODS:
        group = by_method[method]
        successes = sum(bool(row["text"]) for row in group)
        closure_tasks = sum(row["eligible_exact_closure_candidates"] > 0 for row in group)
        selected_hashes = [row["text_sha256"] for row in group if row["text_sha256"]]
        method_summary[method] = {
            "requested_tasks": len(group),
            "basic_valid_outputs": successes,
            "eligible_native_closure_tasks": closure_tasks,
            "eligible_native_closure_candidates": sum(
                row["eligible_exact_closure_candidates"] for row in group
            ),
            "basic_valid_closure_candidates": sum(
                row["basic_valid_closure_candidates"] for row in group
            ),
            "deadline_reached": sum(row["status"].startswith("deadline_") for row in group),
            "selected_outputs_passing_strict_gate": sum(
                row["strict_admitted_selected_output"] is True for row in group
            ),
            "selected_output_denominator_for_strict_gate": successes,
            "unique_selected_text_hashes": len(set(selected_hashes)),
            "duplicate_selected_output_count": len(selected_hashes) - len(set(selected_hashes)),
            "median_search_seconds": statistics.median(row["elapsed_seconds"] for row in group),
        }

    bands = {}
    for (band_id, method), group in sorted(by_band_method.items()):
        bands[f"band-{band_id}:{method}"] = {
            "requested_tasks": len(group),
            "basic_valid_outputs": sum(bool(row["text"]) for row in group),
            "eligible_native_closure_tasks": sum(
                row["eligible_exact_closure_candidates"] > 0 for row in group
            ),
            "eligible_native_closure_candidates": sum(
                row["eligible_exact_closure_candidates"] for row in group
            ),
            "median_search_seconds": statistics.median(row["elapsed_seconds"] for row in group),
        }

    pair_differences = []
    seed_differences: dict[int, list[float]] = defaultdict(list)
    for pair_id, pair in by_pair.items():
        if set(pair) != set(METHODS):
            raise ValueError(f"incomplete paired task: {pair_id}")
        by_name = {name: pair[name] for name in METHODS}
        outside = by_name[METHODS[0]]
        center = by_name[METHODS[1]]
        difference = int(bool(outside["text"])) - int(bool(center["text"]))
        pair_differences.append(difference)
        seed_differences[outside["seed"]].append(float(difference))

    seed_ids = sorted(seed_differences)
    rng = random.Random(BOOTSTRAP_SEED)
    bootstrap_differences = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = [rng.choice(seed_ids) for _ in seed_ids]
        cluster_values = [
            statistics.fmean(seed_differences[seed]) for seed in sampled
        ]
        bootstrap_differences.append(statistics.fmean(cluster_values))
    observed_difference = statistics.fmean(pair_differences)
    bootstrap_interval = [percentile(bootstrap_differences, 0.025),
                          percentile(bootstrap_differences, 0.975)]

    return {
        "experiment": EXPERIMENT,
        "protocol_version": manifest["protocol_version"],
        "phase": result["phase"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": manifest_hash,
        "result_sha256": sha256_file(result_path),
        "task_count": len(rows),
        "paired_task_count": len(by_pair),
        "length_bands": manifest["design"]["length_bands_normalized_ascii_letters_inclusive"],
        "length_stratified": bands,
        "by_method": method_summary,
        "paired_basic_valid_coverage_difference_outside_minus_center": observed_difference,
        "paired_seed_cluster_bootstrap": {
            "resamples": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "percentile_95_interval": bootstrap_interval,
            "interpretation": (
                "degenerate because no arm produced an eligible closure; this is not evidence of equivalence, zero underlying success probability, or language-quality parity"
                if not any(row["text"] for row in rows)
                else "descriptive seed-cluster bootstrap interval"
            ),
        },
        "language_model_diagnostics": {
            "brown": "not run: no selected outputs",
            "gpt2": "not run: no selected outputs",
            "deepseek_v3_2": "not run: no jointly successful output pairs; AWS calls and spend for this experiment are zero",
        },
        "interpretation_limit": "This run estimates output coverage for the two configured algorithms under the frozen bounds. With no closures, it provides no evidence about candidate language quality or which method reads better.",
        "runtime": result.get("runtime"),
        "elapsed_wall_seconds": result["elapsed_wall_seconds"],
        "workers": result["workers"],
    }


def as_markdown(summary: dict) -> str:
    lines = [
        "# Directional search benchmark v3: results",
        "",
        "## Scope",
        "",
        "This was a preregistered operational comparison of the empty-center project search and an outside-in residual-search adaptation. The algorithms used their native closure rules. It was not a causal test of search direction. The v1 and v2 empty-residual pilots informed the protocol and are excluded from the confirmatory denominator.",
        "",
        f"The frozen manifest SHA-256 is `{summary['manifest_sha256']}`; the confirmatory result SHA-256 is `{summary['result_sha256']}`. The run contains {summary['paired_task_count']} paired tasks and {summary['task_count']} requested method runs.",
        "",
        "## Confirmatory results",
        "",
        "| Configured algorithm | Tasks | Eligible closures | Basic-valid outputs | Strict-pass outputs / selected | Median seconds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, values in summary["by_method"].items():
        lines.append(
            f"| `{method}` | {values['requested_tasks']} | {values['eligible_native_closure_tasks']} tasks / {values['eligible_native_closure_candidates']} candidates | {values['basic_valid_outputs']} | {values['selected_outputs_passing_strict_gate']} / {values['selected_output_denominator_for_strict_gate']} | {values['median_search_seconds']:.4f} |"
        )
    lines.extend([
        "",
        "| Length band (normalized letters) | Outside-in: valid / tasks | Center-out: valid / tasks |",
        "|---|---:|---:|",
    ])
    by_band_method = summary["length_stratified"]
    for band_id, band_range in enumerate(summary["length_bands"], 1):
        label = f"{band_range[0]}–{band_range[1]}"
        outside = by_band_method[f"band-{band_id}:outside_in_norvig_hoey_adaptation"]
        center = by_band_method[f"band-{band_id}:center_out_project"]
        lines.append(
            f"| {label} | {outside['basic_valid_outputs']} / {outside['requested_tasks']} | {center['basic_valid_outputs']} / {center['requested_tasks']} |"
        )
    lines.extend([
        "",
        f"Both arms reached zero eligible closures and produced 0/{summary['paired_task_count']} basic-valid outputs. There were no timeouts, no outputs to deduplicate, and no selected outputs to evaluate under the strict gate. The paired output-coverage difference (outside-in minus center-out) was {summary['paired_basic_valid_coverage_difference_outside_minus_center']:.3f}. The seed-cluster bootstrap interval is `{summary['paired_seed_cluster_bootstrap']['percentile_95_interval']}`; because every observed outcome was zero, this interval is degenerate and is not evidence of equivalence or zero underlying success probability.",
        "",
        "Brown, GPT-2, and DeepSeek V3.2 scoring was not run on this benchmark: there were no selected outputs, and therefore no jointly successful pairs to send to Bedrock. This experiment used no AWS inference calls. These results say nothing about the language quality of outputs from either algorithm.",
        "",
        "## Reproduction and artifacts",
        "",
        "The exact vocabulary, jobs, settings, source hashes, citations, and excluded-pilot hashes are in `manifest.json`. The NUC run artifact preserves one row for each requested method task, including failures and search time. Run validation and summarization with:",
        "",
        "```sh",
        "python3 experiments/paired_directional_search_benchmark_20260925.py run --manifest runs/directional-search-benchmark-20260925/manifest.json --phase confirmatory --output /tmp/replayed-confirmatory.json --workers 6",
        "python3 experiments/analyze_paired_directional_search_benchmark_20260925.py --manifest runs/directional-search-benchmark-20260925/manifest.json --results runs/directional-search-benchmark-20260925/confirmatory-v3-native-closure.json --json-output /tmp/summary.json --markdown-output /tmp/summary.md",
        "```",
        "",
        "The search command requires the pinned `wordfreq==3.1.1` dependency. Use the source, Python version, and host recorded in the result artifact; the experiment source hashes are checked before jobs start.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.manifest, args.results)
    args.json_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    args.markdown_output.write_text(as_markdown(summary))
    print(json.dumps({
        "json_output": str(args.json_output),
        "markdown_output": str(args.markdown_output),
        "task_count": summary["task_count"],
        "basic_valid_outputs": sum(
            value["basic_valid_outputs"] for value in summary["by_method"].values()
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
