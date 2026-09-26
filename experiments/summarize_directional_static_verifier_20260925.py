"""Join and summarize v4 Brown, GPT-2, and DeepSeek diagnostics."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import statistics

PAIRED_BOOTSTRAP_REPLICATES = 10_000
PAIRED_BOOTSTRAP_SEED = 20260925


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    at = (len(ordered) - 1) * probability
    low = int(at)
    high = min(low + 1, len(ordered) - 1)
    fraction = at - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def summarize(input_path: Path, brown_path: Path, gpt2_path: Path,
              deepseek_path: Path, output_path: Path) -> dict:
    input_data = json.loads(input_path.read_text())
    brown = json.loads(brown_path.read_text())
    gpt2 = json.loads(gpt2_path.read_text())
    deepseek = json.loads(deepseek_path.read_text())
    input_hash = sha256_file(input_path)
    for name, result in (("Brown", brown), ("GPT-2", gpt2), ("DeepSeek", deepseek)):
        observed_hash = result.get("input_manifest_sha256")
        if observed_hash != input_hash:
            raise ValueError(f"{name} result input hash does not match the frozen verifier input")

    candidate_items = [item for item in input_data["items"] if item["kind"] == "candidate"]
    candidates = {item["id"]: item for item in candidate_items}
    brown_rows = {row["id"]: row for row in brown["scores"]}
    gpt2_rows = {row["id"]: row for row in gpt2["scores"]}
    deepseek_rows = {row["candidate_id"]: row for row in deepseek["comparisons"]}
    if set(brown_rows) != set(candidates):
        raise ValueError("Brown candidate score rows do not match frozen candidates")
    if not set(candidates).issubset(gpt2_rows):
        raise ValueError("GPT-2 is missing candidate rows")

    records = []
    for item_id, item in candidates.items():
        metadata = item["source"]
        shuffle_ids = [f"shuffle:candidate:{item['pair_id']}:{n:02d}" for n in range(32)]
        missing_shuffles = [shuffle_id for shuffle_id in shuffle_ids if shuffle_id not in gpt2_rows]
        if missing_shuffles:
            raise ValueError(f"GPT-2 is missing same-word shuffles for {item_id}")
        candidate_logprob = gpt2_rows[item_id]["mean_token_logprob"]
        shuffle_logprobs = [gpt2_rows[shuffle_id]["mean_token_logprob"] for shuffle_id in shuffle_ids]
        deepseek_row = deepseek_rows.get(item_id)
        deepseek_choice = None
        if deepseek_row:
            choice = deepseek_row.get("parsed", {}).get("choice")
            if choice == "TIE":
                deepseek_choice = "tie"
            elif choice in {"A", "B"}:
                deepseek_choice = (
                    "candidate" if choice == deepseek_row["candidate_side"] else "shuffle"
                )
            else:
                deepseek_choice = "unparsed"
        records.append({
            "id": item_id,
            "pair_id": item["pair_id"],
            "seed": metadata["seed"],
            "band_id": metadata["band_id"],
            "resource_configuration": metadata["resource_configuration"],
            "method": metadata["method"],
            "letters": len("".join(ch for ch in item["text"] if ch.isascii() and ch.isalpha())),
            "brown_order_gain_nats_per_transition": brown_rows[item_id]["brown_order_gain_vs_own_shuffle"],
            "gpt2_candidate_mean_token_logprob": candidate_logprob,
            "gpt2_same_word_shuffle_mean_token_logprob": statistics.fmean(shuffle_logprobs),
            "gpt2_candidate_minus_shuffle": candidate_logprob - statistics.fmean(shuffle_logprobs),
            "gpt2_shuffle_count": len(shuffle_logprobs),
            "deepseek_candidate_vs_shuffle": deepseek_choice,
            "deepseek_raw_response": deepseek_row.get("raw_visible_response") if deepseek_row else None,
            "text_sha256": item["text_sha256"],
        })

    def summarize_group(group: list[dict]) -> dict:
        brown_gains = [row["brown_order_gain_nats_per_transition"] for row in group]
        gpt2_deltas = [row["gpt2_candidate_minus_shuffle"] for row in group]
        deepseek_counts = Counter(row["deepseek_candidate_vs_shuffle"] for row in group)
        return {
            "candidate_count": len(group),
            "brown_mean_order_gain_nats_per_transition": mean(brown_gains),
            "brown_median_order_gain_nats_per_transition": statistics.median(brown_gains) if group else None,
            "brown_positive_order_gain_count": sum(value > 0 for value in brown_gains),
            "gpt2_mean_candidate_minus_shuffle_logprob": mean(gpt2_deltas),
            "gpt2_median_candidate_minus_shuffle_logprob": statistics.median(gpt2_deltas) if group else None,
            "gpt2_candidate_above_shuffle_count": sum(value > 0 for value in gpt2_deltas),
            "deepseek_candidate_preferred": deepseek_counts["candidate"],
            "deepseek_shuffle_preferred": deepseek_counts["shuffle"],
            "deepseek_ties": deepseek_counts["tie"],
            "deepseek_unparsed_or_missing": deepseek_counts["unparsed"] + deepseek_counts[None],
        }

    by_condition = {}
    for condition in sorted({(r["resource_configuration"], r["method"]) for r in records}):
        group = [r for r in records if (r["resource_configuration"], r["method"]) == condition]
        by_condition["%s:%s" % condition] = summarize_group(group)

    paired = defaultdict(list)
    for row in records:
        # Search pair ids include the method name; seed × length band is the
        # shared experimental task key for the two methods.
        paired[(row["seed"], row["band_id"], row["resource_configuration"])].append(row)
    method_differences = {}
    joint_by_configuration = {}
    for configuration in sorted({row["resource_configuration"] for row in records}):
        joint = []
        for (seed, band_id, config), group in paired.items():
            if config != configuration or len(group) != 2:
                continue
            by_method = {row["method"]: row for row in group}
            if set(by_method) != {"outside_in_norvig_hoey_adaptation", "center_out_project"}:
                continue
            outside = by_method["outside_in_norvig_hoey_adaptation"]
            center = by_method["center_out_project"]
            joint.append({
                "seed": seed,
                "band_id": band_id,
                "brown_outside_minus_center": outside["brown_order_gain_nats_per_transition"] - center["brown_order_gain_nats_per_transition"],
                "gpt2_outside_minus_center": outside["gpt2_candidate_mean_token_logprob"] - center["gpt2_candidate_mean_token_logprob"],
                "deepseek_choice_difference_candidate_vs_shuffle": (
                    int(outside["deepseek_candidate_vs_shuffle"] == "candidate")
                    - int(center["deepseek_candidate_vs_shuffle"] == "candidate")
                ),
            })
        joint_by_configuration[configuration] = joint
        method_differences[configuration] = {
            "joint_method_success_tasks": len(joint),
            "brown_mean_outside_minus_center": mean([row["brown_outside_minus_center"] for row in joint]),
            "gpt2_mean_outside_minus_center": mean([row["gpt2_outside_minus_center"] for row in joint]),
            "deepseek_mean_difference_in_separate_candidate_vs_shuffle_choices": mean([
                row["deepseek_choice_difference_candidate_vs_shuffle"] for row in joint
            ]),
            "deepseek_interpretation": "not a direct DeepSeek comparison between the two algorithm outputs",
        }

    paired_cluster_bootstrap = {}
    for configuration, joint in joint_by_configuration.items():
        by_seed: dict[int, list[dict]] = defaultdict(list)
        for row in joint:
            by_seed[row["seed"]].append(row)
        seed_ids = sorted(by_seed)
        bootstrap = {metric: [] for metric in ("brown", "gpt2", "deepseek")}
        if seed_ids:
            rng = random.Random(PAIRED_BOOTSTRAP_SEED)
            for _ in range(PAIRED_BOOTSTRAP_REPLICATES):
                sampled_seeds = [rng.choice(seed_ids) for _ in seed_ids]
                for metric, field in (
                    ("brown", "brown_outside_minus_center"),
                    ("gpt2", "gpt2_outside_minus_center"),
                    ("deepseek", "deepseek_choice_difference_candidate_vs_shuffle"),
                ):
                    values = [row[field] for seed in sampled_seeds for row in by_seed[seed]]
                    bootstrap[metric].append(statistics.fmean(values))
        paired_cluster_bootstrap[configuration] = {
            "joint_success_method_tasks": len(joint),
            "seed_cluster_count": len(seed_ids),
            "resamples": PAIRED_BOOTSTRAP_REPLICATES,
            "seed": PAIRED_BOOTSTRAP_SEED,
            "brown_outside_minus_center_95_percentile_interval": (
                [percentile(bootstrap["brown"], 0.025), percentile(bootstrap["brown"], 0.975)]
                if bootstrap["brown"] else None
            ),
            "gpt2_outside_minus_center_95_percentile_interval": (
                [percentile(bootstrap["gpt2"], 0.025), percentile(bootstrap["gpt2"], 0.975)]
                if bootstrap["gpt2"] else None
            ),
            "deepseek_separate_candidate_vs_shuffle_choice_difference_95_percentile_interval": (
                [percentile(bootstrap["deepseek"], 0.025), percentile(bootstrap["deepseek"], 0.975)]
                if bootstrap["deepseek"] else None
            ),
            "interpretation": "Descriptive seed-cluster bootstrap conditional on both methods yielding a basic-valid output for the same seed-by-band task; it is not a causal or population-level language-quality estimate.",
        }

    result = {
        "experiment": "directional-search-static-verifier-summary-20260925",
        "status": "automatic_diagnostics_not_human_readability_results",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_manifest_sha256": input_hash,
        "inputs": {
            "candidate_count": len(records),
            "brown_result_sha256": sha256_file(brown_path),
            "gpt2_result_sha256": sha256_file(gpt2_path),
            "deepseek_result_sha256": sha256_file(deepseek_path),
            "deepseek_model": deepseek.get("model"),
            "deepseek_method": deepseek.get("method"),
            "deepseek_usage": deepseek.get("usage"),
            "deepseek_estimated_cost_usd": deepseek.get("estimated_cost_usd"),
            "gpt2_model": gpt2.get("model"),
            "brown_method": brown.get("method"),
        },
        "by_resource_configuration_and_method": by_condition,
        "joint_success_method_diagnostics": method_differences,
        "joint_success_seed_cluster_bootstrap": paired_cluster_bootstrap,
        "scores": records,
        "interpretation": "Brown and GPT-2 quantify local order or model fit; DeepSeek supplies one prompted model preference against a same-word shuffle. None is a human rating or validates readability, meaning, or coherence.",
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--brown", type=Path, required=True)
    parser.add_argument("--gpt2", type=Path, required=True)
    parser.add_argument("--deepseek", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.input, args.brown, args.gpt2, args.deepseek, args.output)
    print(json.dumps({
        "output": str(args.output),
        "candidate_count": report["inputs"]["candidate_count"],
        "deepseek_cost_usd": report["inputs"]["deepseek_estimated_cost_usd"],
        "conditions": report["by_resource_configuration_and_method"],
    }, indent=2))


if __name__ == "__main__":
    main()
