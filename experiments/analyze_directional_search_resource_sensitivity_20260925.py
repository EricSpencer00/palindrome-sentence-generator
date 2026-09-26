"""Validate and summarize the frozen v4 resource-sensitivity study."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.paired_directional_search_resource_sensitivity_20260925 import (  # noqa: E402
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    EXPERIMENT,
    METHODS,
    RESOURCE_CONFIGURATIONS,
    sha256_file,
    verify_frozen_manifest,
)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    at = (len(ordered) - 1) * probability
    low = int(at)
    high = min(low + 1, len(ordered) - 1)
    fraction = at - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def summarize(manifest_path: Path, result_path: Path,
              static_verifier_path: Path | None = None) -> dict:
    manifest = json.loads(manifest_path.read_text())
    verify_frozen_manifest(manifest)
    result = json.loads(result_path.read_text())
    manifest_hash = sha256_file(manifest_path)
    if result.get("experiment") != EXPERIMENT or result.get("phase") != "confirmatory":
        raise ValueError("the supplied result is not the v4 confirmatory run")
    if result.get("protocol_manifest_sha256") != manifest_hash:
        raise ValueError("result was not produced from the supplied frozen manifest")

    expected_jobs = {
        (job["pair_id"], job["resource_configuration"], job["method"]): job
        for job in manifest["jobs"] if job["phase"] == "confirmatory"
    }
    rows = result["results"]
    observed_keys = [
        (row["pair_id"], row["resource_configuration"], row["method"])
        for row in rows
    ]
    if len(observed_keys) != len(set(observed_keys)) or set(observed_keys) != set(expected_jobs):
        raise ValueError("result rows do not exactly match the frozen confirmatory jobs")

    by_condition = defaultdict(list)
    by_band_condition = defaultdict(list)
    by_matched_task = defaultdict(dict)
    for row in rows:
        key = (row["pair_id"], row["resource_configuration"], row["method"])
        expected = expected_jobs[key]
        for field in (
            "seed", "band_id", "min_letters", "max_letters",
            "execution_order_within_seed_band_block",
        ):
            if row.get(field) != expected[field]:
                raise ValueError(f"result differs from frozen job on {field}: {key}")
        if row["text"]:
            tape = "".join(ch.lower() for ch in row["text"] if ch.isascii() and ch.isalpha())
            if tape != tape[::-1] or hashlib.sha256(row["text"].encode()).hexdigest() != row["text_sha256"]:
                raise ValueError(f"selected output exactness/hash mismatch: {key}")
            if not all(row["basic_structural_checks"].values()):
                raise ValueError(f"selected output fails a basic gate: {key}")
        by_condition[(row["resource_configuration"], row["method"])].append(row)
        by_band_condition[(row["band_id"], row["resource_configuration"], row["method"])].append(row)
        by_matched_task[row["pair_id"]][(row["resource_configuration"], row["method"])] = row

    condition_summary = {}
    for configuration in RESOURCE_CONFIGURATIONS:
        for method in METHODS:
            group = by_condition[(configuration, method)]
            output_rows = [row for row in group if row["text"]]
            hashes = [row["text_sha256"] for row in output_rows]
            condition_summary[f"{configuration}:{method}"] = {
                "requested_tasks": len(group),
                "basic_valid_outputs": len(output_rows),
                "eligible_native_closure_tasks": sum(
                    row["eligible_exact_closure_candidates"] > 0 for row in group
                ),
                "eligible_native_closure_candidates": sum(
                    row["eligible_exact_closure_candidates"] for row in group
                ),
                "basic_valid_closure_candidates": sum(
                    row["basic_valid_closure_candidates"] for row in group
                ),
                "deadline_reached": sum(row["status"].startswith("deadline_") for row in group),
                "strict_pass_outputs": sum(
                    row["strict_admitted_selected_output"] is True for row in output_rows
                ),
                "strict_pass_denominator": len(output_rows),
                "unique_output_hashes": len(set(hashes)),
                "duplicate_output_count": len(hashes) - len(set(hashes)),
                "median_search_seconds": statistics.median(row["elapsed_seconds"] for row in group),
            }

    by_configuration = {}
    for configuration in RESOURCE_CONFIGURATIONS:
        group = [row for row in rows if row["resource_configuration"] == configuration]
        by_configuration[configuration] = {
            "requested_tasks": len(group),
            "basic_valid_outputs": sum(bool(row["text"]) for row in group),
            "coverage": sum(bool(row["text"]) for row in group) / len(group),
            "eligible_native_closure_tasks": sum(
                row["eligible_exact_closure_candidates"] > 0 for row in group
            ),
            "deadline_reached": sum(row["status"].startswith("deadline_") for row in group),
            "median_search_seconds": statistics.median(row["elapsed_seconds"] for row in group),
        }

    by_band = {}
    for (band_id, configuration, method), group in sorted(by_band_condition.items()):
        by_band[f"band-{band_id}:{configuration}:{method}"] = {
            "requested_tasks": len(group),
            "basic_valid_outputs": sum(bool(row["text"]) for row in group),
            "eligible_native_closure_tasks": sum(
                row["eligible_exact_closure_candidates"] > 0 for row in group
            ),
        }

    escalated_outputs = [
        row for row in rows
        if row["resource_configuration"] == "escalated_v4" and row["text"]
    ]
    output_surface_audit = {
        "selected_output_count": len(escalated_outputs),
        "strict_pass_count": sum(
            row["strict_admitted_selected_output"] is True
            for row in escalated_outputs
        ),
        "unique_texts_by_method": {
            method: len({
                row["text_sha256"] for row in escalated_outputs
                if row["method"] == method
            })
            for method in METHODS
        },
        "outputs_containing_exact_phrase_one_poem_association": sum(
            "one poem association" in row["text"].casefold()
            for row in escalated_outputs
        ),
    }

    resource_differences_by_seed = defaultdict(list)
    resource_differences_by_method = defaultdict(list)
    for task_id, conditions in by_matched_task.items():
        if len(conditions) != len(RESOURCE_CONFIGURATIONS) * len(METHODS):
            raise ValueError(f"incomplete four-condition block: {task_id}")
        original = conditions["original_v3", METHODS[0]]
        for method in METHODS:
            baseline = conditions["original_v3", method]
            escalated = conditions["escalated_v4", method]
            difference = int(bool(escalated["text"])) - int(bool(baseline["text"]))
            resource_differences_by_seed[baseline["seed"]].append(float(difference))
            resource_differences_by_method[method].append(difference)
        if original["pair_id"] != task_id:
            raise AssertionError("internal task-key mismatch")

    observed_difference = statistics.fmean(
        value for seed_values in resource_differences_by_seed.values()
        for value in seed_values
    )
    seed_ids = sorted(resource_differences_by_seed)
    rng = random.Random(BOOTSTRAP_SEED)
    bootstrap = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_seeds = [rng.choice(seed_ids) for _ in seed_ids]
        bootstrap.append(statistics.fmean(
            statistics.fmean(resource_differences_by_seed[seed])
            for seed in sampled_seeds
        ))
    interval = [percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)]
    any_success = any(row["text"] for row in rows)
    static_verifier = None
    if static_verifier_path is not None:
        static_verifier = json.loads(static_verifier_path.read_text())
        expected_static_count = by_configuration["escalated_v4"]["basic_valid_outputs"]
        if static_verifier.get("status") != "automatic_diagnostics_not_human_readability_results":
            raise ValueError("static verifier summary has an unexpected status")
        if static_verifier.get("inputs", {}).get("candidate_count") != expected_static_count:
            raise ValueError("static verifier candidate count does not match escalated outputs")
        if static_verifier.get("inputs", {}).get("deepseek_usage", {}).get("comparisons") != expected_static_count:
            raise ValueError("DeepSeek comparison count does not match frozen selected outputs")
        expected_candidates = {}
        for row in escalated_outputs:
            candidate_id = (
                f"candidate:seed-{row['seed']}:band-{row['band_id']}:"
                f"escalated_v4:{row['method']}"
            )
            expected_candidates[candidate_id] = row
        observed_scores = {
            row["id"]: row for row in static_verifier.get("scores", [])
        }
        if set(observed_scores) != set(expected_candidates):
            raise ValueError("static verifier score IDs do not match v4 selected outputs")
        for candidate_id, expected in expected_candidates.items():
            observed = observed_scores[candidate_id]
            for field in ("text_sha256", "seed", "band_id", "resource_configuration", "method"):
                if observed.get(field) != expected[field]:
                    raise ValueError(
                        f"static verifier {field} does not match search output: {candidate_id}"
                    )

    return {
        "experiment": EXPERIMENT,
        "protocol_version": 4,
        "status": "confirmatory_results_validated",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": manifest_hash,
        "result_sha256": sha256_file(result_path),
        "task_count": len(rows),
        "seed_band_block_count": len(by_matched_task),
        "resource_configurations": manifest["resource_configurations"],
        "length_bands": manifest["design"]["length_bands_normalized_ascii_letters_inclusive"],
        "by_configuration": by_configuration,
        "by_configuration_and_method": condition_summary,
        "length_stratified": by_band,
        "output_surface_audit": output_surface_audit,
        "primary_escalated_minus_original_coverage_difference": observed_difference,
        "primary_seed_cluster_bootstrap": {
            "cluster_count": len(seed_ids),
            "resamples": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "percentile_95_interval": interval,
            "interpretation": (
                "degenerate because there are no observed successes; not evidence of equivalence or zero underlying success probability"
                if not any_success else "descriptive paired seed-cluster bootstrap interval"
            ),
        },
        "secondary_algorithm_coverage_differences_escalated_minus_original": {
            method: statistics.fmean(values)
            for method, values in resource_differences_by_method.items()
        },
        "static_verifier_status": (
            f"completed for {static_verifier['inputs']['candidate_count']} escalated_v4 outputs; no original_v3 outputs existed"
            if static_verifier is not None else
            "pending: run only on frozen selected outputs"
            if any_success else "not run: no selected outputs; DeepSeek AWS spend is zero"
        ),
        "static_verifier": ({
            "summary_sha256": sha256_file(static_verifier_path),
            "input_manifest_sha256": static_verifier["input_manifest_sha256"],
            "inputs": static_verifier["inputs"],
            "by_resource_configuration_and_method": static_verifier["by_resource_configuration_and_method"],
            "joint_success_method_diagnostics": static_verifier["joint_success_method_diagnostics"],
            "joint_success_seed_cluster_bootstrap": static_verifier["joint_success_seed_cluster_bootstrap"],
        } if static_verifier is not None else None),
        "resource_history": manifest["design"]["previous_protocol_results"],
        "runtime": result.get("runtime"),
        "elapsed_wall_seconds": result["elapsed_wall_seconds"],
        "workers": result["workers"],
    }


def as_markdown(summary: dict) -> str:
    lines = [
        "# Resource-sensitivity study v4: results",
        "",
        "## Why this follow-up exists",
        "",
        "The preregistered v3 comparison produced no eligible closures: 0/90 tasks in each arm. Toy-vocabulary controls then verified that both search APIs can reach and admit a known target. v4 was frozen on fresh seeds to test whether a larger resource bundle changes coverage. It is an adaptive follow-up to v3, not an independent confirmation of a pre-v3 hypothesis.",
        "",
        f"The v4 manifest hash is `{summary['manifest_sha256']}` and the confirmatory result hash is `{summary['result_sha256']}`. The frozen manifest retains hashes for v1–v3 results. This confirmatory phase contains {summary['task_count']} method/configuration runs across {summary['seed_band_block_count']} seed-by-length blocks.",
        "",
        "## Primary result",
        "",
        "| Resource configuration | Tasks | Basic-valid outputs | Coverage | Eligible closure tasks | Deadline reached | Median seconds / task |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for configuration, values in summary["by_configuration"].items():
        lines.append(
            f"| `{configuration}` | {values['requested_tasks']} | {values['basic_valid_outputs']} | {values['coverage']:.3f} | {values['eligible_native_closure_tasks']} | {values['deadline_reached']} | {values['median_search_seconds']:.3f} |"
        )
    interval = summary["primary_seed_cluster_bootstrap"]["percentile_95_interval"]
    interval_text = f"[{interval[0]:.3f}, {interval[1]:.3f}]"
    lines.extend([
        "",
        f"The paired coverage difference (escalated minus original) was {summary['primary_escalated_minus_original_coverage_difference']:.3f}, with a descriptive paired seed-cluster bootstrap 95% interval of `{interval_text}`.",
        "",
        "## Method-level coverage",
        "",
        "| Resource configuration | Algorithm | Tasks | Basic-valid | Eligible closure tasks | Strict-pass / selected | Unique outputs | Median seconds |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for key, values in summary["by_configuration_and_method"].items():
        configuration, method = key.split(":", 1)
        lines.append(
            f"| `{configuration}` | `{method}` | {values['requested_tasks']} | {values['basic_valid_outputs']} | {values['eligible_native_closure_tasks']} | {values['strict_pass_outputs']} / {values['strict_pass_denominator']} | {values['unique_output_hashes']} | {values['median_search_seconds']:.3f} |"
        )
    lines.extend([
        "",
        f"The resource settings change beam width, candidate limit, children per parent, and cooperative time budget together. The estimate therefore compares two resource bundles, not the isolated effect of runtime. The two algorithms also retain different native closure rules, so their contrast is descriptive rather than a causal effect of growth direction.",
        f"Among the 66 selected escalated outputs, {summary['output_surface_audit']['strict_pass_count']} passed every additional mechanical screen; this is a selected-output pass rate, not strict search coverage. The outside-in and center-out methods produced {summary['output_surface_audit']['unique_texts_by_method'][METHODS[0]]} and {summary['output_surface_audit']['unique_texts_by_method'][METHODS[1]]} unique texts, respectively. The exact phrase `one poem association` occurs in {summary['output_surface_audit']['outputs_containing_exact_phrase_one_poem_association']} of 66 outputs, illustrating that exactness and lexical screens do not ensure coherent prose.",
        "",
        "## Language diagnostics",
        "",
        f"{summary['static_verifier_status']}. Brown order gain, pinned GPT-2 token likelihood, and one counterbalanced DeepSeek V3.2 comparison against a deterministic same-word shuffle are only computed for selected basic-valid outputs. No model diagnostic is a human rating.",
    ])
    static = summary.get("static_verifier")
    if static:
        outside = static["by_resource_configuration_and_method"]["escalated_v4:outside_in_norvig_hoey_adaptation"]
        center = static["by_resource_configuration_and_method"]["escalated_v4:center_out_project"]
        paired = static["joint_success_method_diagnostics"]["escalated_v4"]
        paired_bootstrap = static["joint_success_seed_cluster_bootstrap"]["escalated_v4"]
        paired_brown_ci = paired_bootstrap["brown_outside_minus_center_95_percentile_interval"]
        paired_gpt2_ci = paired_bootstrap["gpt2_outside_minus_center_95_percentile_interval"]
        lines.extend([
            "",
            "| Diagnostic, computed only among selected valid outputs | Outside-in (n=33) | Center-out (n=33) |",
            "|---|---:|---:|",
            f"| Brown order gain vs 32 same-word shuffles, mean (positive count) | {outside['brown_mean_order_gain_nats_per_transition']:.3f} ({outside['brown_positive_order_gain_count']}/33) | {center['brown_mean_order_gain_nats_per_transition']:.3f} ({center['brown_positive_order_gain_count']}/33) |",
            f"| GPT-2 candidate minus 32-shuffle mean token log probability (positive count) | {outside['gpt2_mean_candidate_minus_shuffle_logprob']:.3f} ({outside['gpt2_candidate_above_shuffle_count']}/33) | {center['gpt2_mean_candidate_minus_shuffle_logprob']:.3f} ({center['gpt2_candidate_above_shuffle_count']}/33) |",
            f"| DeepSeek V3.2 choice: candidate / shuffle / tie | {outside['deepseek_candidate_preferred']} / {outside['deepseek_shuffle_preferred']} / {outside['deepseek_ties']} | {center['deepseek_candidate_preferred']} / {center['deepseek_shuffle_preferred']} / {center['deepseek_ties']} |",
            "",
            f"For the 33 seed-by-band tasks where both methods returned an output, the descriptive paired outside-in-minus-center-out differences were {paired['brown_mean_outside_minus_center']:.3f} Brown nats/transition (seed-cluster 95% interval `[{paired_brown_ci[0]:.3f}, {paired_brown_ci[1]:.3f}]`) and {paired['gpt2_mean_outside_minus_center']:.3f} GPT-2 nats/token (interval `[{paired_gpt2_ci[0]:.3f}, {paired_gpt2_ci[1]:.3f}]`). These are post-run, conditional model diagnostics, not human judgments or population-level method effects. DeepSeek tied all 66 separate candidate-versus-shuffle pairs; it did not directly compare the two algorithms.",
            f"The DeepSeek run used {static['inputs']['deepseek_usage']['input_tokens']:,} input and {static['inputs']['deepseek_usage']['output_tokens']:,} output tokens, with an estimated charge of USD {static['inputs']['deepseek_estimated_cost_usd']:.5f}; the preflight upper bound was USD {static['inputs']['deepseek_method']['preflight_cost_ceiling_usd']:.5f}.",
            f"The static verifier summary SHA-256 is `{static['summary_sha256']}`; its frozen input SHA-256 is `{static['input_manifest_sha256']}`.",
        ])
    lines.extend([
        "",
        "## Reproduction",
        "",
        "The manifest freezes the vocabulary, 240 jobs, resource settings, arm order, source hashes, model methods, and v1–v3 history. The raw run retains all requested rows and failures. The saved verifier input contains every selected text and deterministic same-word shuffle; model scores can be rerun against it on an authorized CPU host. Reproduce and validate with:",
        "",
        "```sh",
        "python3 experiments/paired_directional_search_resource_sensitivity_20260925.py run --manifest runs/directional-search-resource-sensitivity-20260925/manifest.json --phase confirmatory --output /tmp/replayed-v4.json --workers 6",
        "NLTK_DATA=/path/to/nltk_data python3 -m experiments.score_directional_search_brown_20260925 --input runs/directional-search-resource-sensitivity-20260925/static-verifier-input-aws.json --output /tmp/brown-v4.json",
        "HF_HOME=/path/to/hf-cache OMP_NUM_THREADS=4 python3 -m experiments.score_gpt2_static_verifier_aws_20260925 --input runs/directional-search-resource-sensitivity-20260925/static-verifier-input-aws.json --output /tmp/gpt2-v4.json --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e --batch-size 8 --execution-host 'authorized CPU host'",
        "python3 experiments/infer_deepseek_static_verifier_20260925.py --input runs/directional-search-resource-sensitivity-20260925/static-verifier-input-aws.json --output /tmp/deepseek-v4.json --budget-usd 0.20 --interval-seconds 0.75",
        "python3 experiments/summarize_directional_static_verifier_20260925.py --input runs/directional-search-resource-sensitivity-20260925/static-verifier-input-aws.json --brown /tmp/brown-v4.json --gpt2 /tmp/gpt2-v4.json --deepseek /tmp/deepseek-v4.json --output /tmp/static-verifier-summary-v4.json",
        "python3 experiments/analyze_directional_search_resource_sensitivity_20260925.py --manifest runs/directional-search-resource-sensitivity-20260925/manifest.json --results runs/directional-search-resource-sensitivity-20260925/confirmatory-v4-resource-sensitivity.json --static-verifier-summary /tmp/static-verifier-summary-v4.json --json-output /tmp/v4-summary.json --markdown-output /tmp/v4-summary.md",
        "```",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--static-verifier-summary", type=Path)
    args = parser.parse_args()
    summary = summarize(args.manifest, args.results, args.static_verifier_summary)
    args.json_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    args.markdown_output.write_text(as_markdown(summary))
    print(json.dumps({
        "json_output": str(args.json_output),
        "markdown_output": str(args.markdown_output),
        "task_count": summary["task_count"],
        "outputs": sum(v["basic_valid_outputs"] for v in summary["by_configuration"].values()),
    }, indent=2))


if __name__ == "__main__":
    main()
