"""Paired, fixed-work comparison of terminal and incremental POS filtering.

Each trial explores the same vocabulary under the same popped-state budget.
The seed defines a stable rank for every vocabulary item; it does not drive a
stateful random-number generator.  Consequently, whenever both arms reach the
same partial state, they expand its siblings in exactly the same order.

The experiment measures search work rather than stopping on elapsed time or a
target number of accepted outputs.  Every arm runs in a fresh process so its
maximum resident-set size is independently measurable.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import random
import resource
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.exhaustive import enumerate_palindromes
from llm_palindrome.pairs import acceptable_pair, split_at_mirror
from llm_palindrome.centerout import COState, _expand
from llm_palindrome.search import WordTries, unit_letters
from llm_palindrome.sentence_plan import SentencePlan
from paper.evidence_paths import evidence_path
from paper.provenance import environment, snapshot


ARMS = ("terminal", "incremental")
COUNT_FIELDS = (
    "expansion_calls",
    "candidate_expansions",
    "states_generated",
    "states_pushed",
    "states_popped",
    "state_pruned",
    "closed_states",
    "eligible_closures",
    "accepted",
)


def _rss_mib() -> float:
    """Return this process's maximum resident-set size in MiB."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux and the BSDs exposed in our environments
    # report KiB. Record the platform in provenance so this conversion is
    # independently checkable.
    divisor = 1024 * 1024 if platform.system() == "Darwin" else 1024
    return value / divisor


def _load_inputs(config: dict):
    payload = json.loads(gzip.decompress(
        evidence_path("inputs/brown.json.gz").read_bytes()))
    raw = evidence_path("inputs/vocab30k.txt").read_text().split()[:config["vocab"]]
    words = [word for word in raw if word in payload["table"]]
    tries = WordTries(words)
    plan = SentencePlan(payload["table"], payload["shapes"],
                        config["min_words"], config["max_units"] // 2)
    return words, tries, plan


def run_trial(task: tuple[int, str, dict, dict | None]) -> dict:
    seed, arm, config, opening = task
    words, tries, plan = _load_inputs(config)
    rss_before = _rss_mib()
    stats: dict[str, int | str] = {}
    accepted: set[tuple[str, str]] = set()
    started_cpu = time.process_time()
    started_wall = time.perf_counter()

    for units in enumerate_palindromes(
            tries,
            min_letters=config["min_letters"],
            max_letters=config["max_letters"],
            max_overhang=config["max_overhang"],
            max_units=config["max_units"],
            node_budget=config["node_budget"],
            shard=opening["index"] if opening else 0,
            shards=len(words) if opening else 1,
            order_seed=seed,
            traversal=config.get("traversal", "dfs"),
            reverse_order=config.get("reverse_order", False),
            allow_state=plan.state_possible if arm == "incremental" else None,
            stats=stats):
        split = split_at_mirror(units)
        if split is None:
            continue
        left, right = split
        if not acceptable_pair(left, right, min_words=config["min_words"]):
            continue
        if not (plan.complete(left) and plan.complete(right)):
            continue
        accepted.add((" ".join(left), " ".join(right)))

    cpu_seconds = time.process_time() - started_cpu
    wall_seconds = time.perf_counter() - started_wall
    if stats["states_generated"] != stats["states_pushed"] + stats["state_pruned"]:
        raise AssertionError("generated states must be pushed or rejected by the POS gate")
    if arm == "terminal" and stats["state_pruned"]:
        raise AssertionError("the terminal arm must not prune partial states")
    if len(accepted) > stats["yielded"]:
        raise AssertionError("accepted outputs cannot exceed eligible closures")
    ordered = sorted(accepted)
    digest = hashlib.sha256(
        "".join(f"{left}\0{right}\n" for left, right in ordered).encode("utf-8")
    ).hexdigest()
    result = {
        "seed": seed,
        "arm": arm,
        "opening": opening,
        "vocabulary": len(words),
        "ordering": (
            "SHA-256(seed, NUL, vocabulary item); "
            f"{'descending' if config.get('reverse_order') else 'ascending'} rank; "
            f"{config.get('traversal', 'dfs').upper()} frontier"
        ),
        "stop_reason": stats.get("stop_reason", "unrecorded"),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "rss_before_search_mib": rss_before,
        "peak_rss_mib": _rss_mib(),
        "peak_frontier": stats.get("peak_frontier", 0),
        "accepted_sha256": digest,
        "accepted_sample": [
            {"left": left, "right": right} for left, right in ordered[:5]
        ],
        "accepted_pairs": [
            {"left": left, "right": right} for left, right in ordered
        ],
    }
    result.update({field: stats.get(
        "yielded" if field == "eligible_closures" else field, 0)
        for field in COUNT_FIELDS})
    result["accepted"] = len(accepted)
    return result


def _rate(rows: list[dict], numerator: str, denominator: str,
          scale: float = 1.0) -> float:
    return scale * sum(row[numerator] for row in rows) / sum(
        row[denominator] for row in rows)


def _paired_rate_ratio(rows: list[dict], numerator: str, denominator: str,
                       indices: list[int]) -> float | None:
    by_seed = {(row["seed"], row["arm"]): row for row in rows}
    seeds = sorted({row["seed"] for row in rows})
    sampled = [seeds[index] for index in indices]

    def rate(arm):
        selected = [by_seed[(seed, arm)] for seed in sampled]
        den = sum(row[denominator] for row in selected)
        return (sum(row[numerator] for row in selected) / den) if den else None

    terminal, incremental = rate("terminal"), rate("incremental")
    if terminal in (None, 0) or incremental is None:
        return None
    return incremental / terminal


def bootstrap_ratio_ci(rows: list[dict], numerator: str, denominator: str,
                       *, replicates: int = 10000,
                       seed: int = 20260911) -> list[float | None]:
    """Paired percentile bootstrap CI for an incremental/terminal rate ratio."""
    n = len({row["seed"] for row in rows})
    rng = random.Random(seed)
    values = []
    for _ in range(replicates):
        ratio = _paired_rate_ratio(
            rows, numerator, denominator, [rng.randrange(n) for _ in range(n)])
        if ratio is not None:
            values.append(ratio)
    if not values:
        return [None, None]
    values.sort()
    lo = values[int(0.025 * (len(values) - 1))]
    hi = values[int(0.975 * (len(values) - 1))]
    return [lo, hi]


def summarize(rows: list[dict], bootstrap_replicates: int = 10000) -> dict:
    by_arm = {arm: [row for row in rows if row["arm"] == arm] for arm in ARMS}
    seeds = sorted({row["seed"] for row in rows})
    if any({row["seed"] for row in by_arm[arm]} != set(seeds) for arm in ARMS):
        raise ValueError("Every seed must have one result for each arm")
    vocabularies = {row["vocabulary"] for row in rows if "vocabulary" in row}
    if len(vocabularies) > 1:
        raise ValueError("All trials must use the same vocabulary")
    openings = {row["seed"]: row.get("opening") for row in rows}
    for seed in seeds:
        paired_openings = {json.dumps(row.get("opening"), sort_keys=True)
                           for row in rows if row["seed"] == seed}
        if len(paired_openings) != 1:
            raise ValueError("Paired arms must use the same opening")

    arms = {}
    for arm, selected in by_arm.items():
        totals = {field: sum(row[field] for row in selected)
                  for field in COUNT_FIELDS}
        arms[arm] = {
            "trials": len(selected),
            "stop_reasons": dict(Counter(row["stop_reason"] for row in selected)),
            "totals": totals,
            "mean_per_trial": {field: statistics.fmean(row[field] for row in selected)
                               for field in COUNT_FIELDS},
            "cpu_seconds_total": sum(row["cpu_seconds"] for row in selected),
            "cpu_seconds_mean": statistics.fmean(row["cpu_seconds"] for row in selected),
            "wall_seconds_total": sum(row["wall_seconds"] for row in selected),
            "peak_rss_mib_mean": statistics.fmean(row["peak_rss_mib"] for row in selected),
            "peak_rss_mib_max": max(row["peak_rss_mib"] for row in selected),
            "peak_frontier_mean": statistics.fmean(row["peak_frontier"] for row in selected),
            "accepted_per_million_generated": _rate(
                selected, "accepted", "states_generated", 1_000_000),
            "accepted_per_million_popped": _rate(
                selected, "accepted", "states_popped", 1_000_000),
            "accepted_per_cpu_second": _rate(
                selected, "accepted", "cpu_seconds"),
            "accepted_median": statistics.median(row["accepted"] for row in selected),
            "accepted_range": [min(row["accepted"] for row in selected),
                               max(row["accepted"] for row in selected)],
            "trials_with_accepted_output": sum(row["accepted"] > 0
                                                for row in selected),
        }

    ratios = {}
    for label, numerator, denominator in (
        ("accepted_per_generated", "accepted", "states_generated"),
        ("accepted_per_popped", "accepted", "states_popped"),
        ("accepted_per_cpu_second", "accepted", "cpu_seconds"),
    ):
        terminal_rate = _rate(by_arm["terminal"], numerator, denominator)
        incremental_rate = _rate(by_arm["incremental"], numerator, denominator)
        ratios[label] = {
            "incremental_over_terminal": (
                incremental_rate / terminal_rate if terminal_rate else None),
            "paired_bootstrap_95pct_ci": bootstrap_ratio_ci(
                rows, numerator, denominator, replicates=bootstrap_replicates),
        }

    terminal_generated = arms["terminal"]["totals"]["states_generated"]
    incremental = arms["incremental"]
    ratios["incremental_gate_rejection_rate"] = (
        incremental["totals"]["state_pruned"]
        / incremental["totals"]["states_generated"])
    ratios["generated_states_incremental_over_terminal"] = (
        incremental["totals"]["states_generated"] / terminal_generated)
    ratios["cpu_seconds_incremental_over_terminal"] = (
        incremental["cpu_seconds_total"] / arms["terminal"]["cpu_seconds_total"])
    ratios["mean_peak_rss_incremental_over_terminal"] = (
        incremental["peak_rss_mib_mean"] / arms["terminal"]["peak_rss_mib_mean"])
    by_seed = {(row["seed"], row["arm"]): row for row in rows}
    comparisons = [
        (by_seed[(seed, "incremental")]["accepted"]
         > by_seed[(seed, "terminal")]["accepted"])
        - (by_seed[(seed, "incremental")]["accepted"]
           < by_seed[(seed, "terminal")]["accepted"])
        for seed in seeds
    ]
    ratios["accepted_pairwise"] = {
        "incremental_wins": comparisons.count(1),
        "ties": comparisons.count(0),
        "terminal_wins": comparisons.count(-1),
    }

    return {
        "experimental_unit": ("paired fixed opening subtree"
                              if any(openings.values())
                              else "paired deterministic ordering seed"),
        "seeds": seeds,
        "openings": [openings[seed] for seed in seeds if openings[seed] is not None],
        "vocabulary": next(iter(vocabularies)) if vocabularies else None,
        "arms": arms,
        "ratios": ratios,
        "inference": {
            "method": "paired percentile bootstrap over trials",
            "replicates": bootstrap_replicates,
            "caution": (
                "Trials vary fixed opening subtrees and deterministic traversal order "
                "over one vocabulary; they are not independent corpus or system "
                "replications. The node budget is a maximum for both arms; confidence "
                "intervals characterize sensitivity to these selected searches."
            ),
        },
    }


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def _fmt_ci(bounds: list[float | None]) -> str:
    if any(value is None for value in bounds):
        return "not estimable"
    return f"[{bounds[0]:.3f}, {bounds[1]:.3f}]"


def _fmt_ratio(value: float | None) -> str:
    return "not estimable" if value is None else f"{value:.3f}x"


def results_markdown(config: dict, summary: dict) -> str:
    terminal = summary["arms"]["terminal"]
    incremental = summary["arms"]["incremental"]
    ratio = summary["ratios"]

    def means(field):
        return (terminal["mean_per_trial"][field],
                incremental["mean_per_trial"][field])

    generated = means("states_generated")
    pushed = means("states_pushed")
    popped = means("states_popped")
    closures = means("eligible_closures")
    accepted = means("accepted")
    ci_generated = ratio["accepted_per_generated"]["paired_bootstrap_95pct_ci"]
    ci_popped = ratio["accepted_per_popped"]["paired_bootstrap_95pct_ci"]
    ci_cpu = ratio["accepted_per_cpu_second"]["paired_bootstrap_95pct_ci"]
    gate_pct = 100 * ratio["incremental_gate_rejection_rate"]
    seed_count = len(summary["seeds"])
    if summary["openings"]:
        unit = (
            f"{seed_count} paired fixed opening subtrees, selected without POS tags "
            f"or outcome data; ordering seeds {summary['seeds'][0]}--{summary['seeds'][-1]}."
        )
        stopping = (
            f"At most {config['node_budget']:,} popped states per arm and opening; "
            "a fully pruned or exhausted subtree stops earlier."
        )
    else:
        unit = (f"{seed_count} paired ordering seeds: "
                f"{summary['seeds'][0]}--{summary['seeds'][-1]}.")
        stopping = (f"{config['node_budget']:,} popped states per arm and seed; "
                    "no wall-time or accepted-output stop.")

    return f"""# Controlled POS-pruning experiment

## Design

- {unit}
- {stopping}
- Full frozen vocabulary intersection ({summary['vocabulary']:,} admitted of {config['vocab']:,} requested entries), Brown tag table, and the manuscript's {config['min_letters']}--{config['max_letters']}-letter, {config['max_units']}-word, {config['max_overhang']}-overhang limits.
- Stable SHA-256 word ranks make expansion order a pure function of seed and word. Shared states receive identical sibling order in both arms.
- Arms use the same completed-candidate checks. Only the incremental arm applies POS feasibility before pushing a state.
- CPU time is process time. Peak RSS is measured in a fresh process per arm. Trials ran concurrently, so summed wall time is not an elapsed experiment duration.

## Results

Values below are means per paired trial unless marked as rates.

| Metric | Terminal POS | Incremental POS |
|---|---:|---:|
| States generated | {_fmt(generated[0], 1)} | {_fmt(generated[1], 1)} |
| States pushed | {_fmt(pushed[0], 1)} | {_fmt(pushed[1], 1)} |
| States popped | {_fmt(popped[0], 1)} | {_fmt(popped[1], 1)} |
| Eligible terminal closures | {_fmt(closures[0], 1)} | {_fmt(closures[1], 1)} |
| Distinct accepted outputs | {_fmt(accepted[0], 2)} | {_fmt(accepted[1], 2)} |
| CPU seconds | {_fmt(terminal['cpu_seconds_mean'], 3)} | {_fmt(incremental['cpu_seconds_mean'], 3)} |
| Accepted / million generated | {_fmt(terminal['accepted_per_million_generated'], 2)} | {_fmt(incremental['accepted_per_million_generated'], 2)} |
| Accepted / million popped | {_fmt(terminal['accepted_per_million_popped'], 2)} | {_fmt(incremental['accepted_per_million_popped'], 2)} |
| Accepted / CPU second | {_fmt(terminal['accepted_per_cpu_second'], 3)} | {_fmt(incremental['accepted_per_cpu_second'], 3)} |
| Peak RSS, MiB | {_fmt(terminal['peak_rss_mib_mean'], 2)} | {_fmt(incremental['peak_rss_mib_mean'], 2)} |
| Peak frontier states | {_fmt(terminal['peak_frontier_mean'], 1)} | {_fmt(incremental['peak_frontier_mean'], 1)} |

The incremental gate rejected {gate_pct:.2f}% of otherwise generated states before insertion. Across all trials, the incremental/terminal rate ratios were:

| Rate ratio | Estimate | Paired bootstrap 95% interval |
|---|---:|---:|
| Accepted / generated state | {_fmt_ratio(ratio['accepted_per_generated']['incremental_over_terminal'])} | {_fmt_ci(ci_generated)} |
| Accepted / popped state | {_fmt_ratio(ratio['accepted_per_popped']['incremental_over_terminal'])} | {_fmt_ci(ci_popped)} |
| Accepted / CPU second | {_fmt_ratio(ratio['accepted_per_cpu_second']['incremental_over_terminal'])} | {_fmt_ci(ci_cpu)} |

Incremental filtering returned more accepted outputs in {ratio['accepted_pairwise']['incremental_wins']} paired trials, tied in {ratio['accepted_pairwise']['ties']}, and returned fewer in {ratio['accepted_pairwise']['terminal_wins']}. The terminal and incremental arms produced at least one accepted output in {terminal['trials_with_accepted_output']} and {incremental['trials_with_accepted_output']} of {seed_count} trials, respectively.

## Interpretation limits

The paired budget removes the earlier wall-time, output-cap, arm-order, and traversal-dependent-randomness confounds. It does not make every bounded walk exhaustive. The intervals measure variation across the fixed opening subtrees and their deterministic orderings over one frozen vocabulary; they are not corpus-level or hardware-population intervals. POS-shape admission is structural and is not evidence of grammaticality, meaning, or readability.
"""


def _tasks(seeds: list[int], config: dict,
           openings: list[dict] | None = None) -> list[tuple[int, str, dict, dict | None]]:
    if openings is not None and len(openings) != len(seeds):
        raise ValueError("Openings and seeds must have equal length")
    tasks = []
    for index, seed in enumerate(seeds):
        arms = ARMS if seed % 2 == 0 else tuple(reversed(ARMS))
        opening = openings[index] if openings is not None else None
        tasks.extend((seed, arm, config, opening) for arm in arms)
    return tasks


def select_openings(config: dict, count: int, min_branches: int,
                    selection_seed: int) -> list[dict]:
    """Select nontrivial roots without consulting tags or search outcomes."""
    words, tries, _ = _load_inputs(config)
    root = COState(sort_key=0.0, left=(), right=(), overhang="", owner="R",
                   center_len=0)
    candidates = []
    for index, (_, word, overhang, owner) in enumerate(
            _expand(root, tries, limit=10 ** 6)):
        if len(overhang) > config["max_overhang"]:
            continue
        if len(unit_letters(word)) > config["max_letters"]:
            continue
        state = COState(sort_key=0.0, left=(word,), right=(),
                        overhang=overhang, owner=owner, center_len=0)
        branches = len(_expand(state, tries, limit=10 ** 6))
        if branches >= min_branches:
            candidates.append({"index": index, "word": word,
                               "immediate_branches": branches})
    if len(candidates) < count:
        raise ValueError(
            f"Only {len(candidates)} openings have at least {min_branches} branches")
    prefix = str(selection_seed).encode("ascii") + b"\0"
    candidates.sort(key=lambda row: (
        hashlib.sha256(prefix + row["word"].encode("utf-8")).digest(),
        row["index"],
    ))
    return candidates[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (
        ("vocab", 30000),
        ("min-letters", 20),
        ("max-letters", 44),
        ("min-words", 3),
        ("max-units", 18),
        ("max-overhang", 16),
        ("node-budget", 100000),
        ("trials", 30),
        ("base-seed", 0),
        ("bootstrap-replicates", 10000),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--trial-mode", choices=("seeds", "openings"),
                        default="openings")
    parser.add_argument("--traversal", choices=("dfs", "bfs"), default="dfs")
    parser.add_argument("--reverse-order", action="store_true",
                        help="reverse the stable sibling order before frontier insertion")
    parser.add_argument("--opening-min-branches", type=int, default=20)
    parser.add_argument("--opening-selection-seed", type=int, default=20260911)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if min(args.vocab, args.min_words, args.max_units, args.max_overhang,
           args.node_budget, args.trials, args.workers,
           args.bootstrap_replicates, args.opening_min_branches) <= 0:
        parser.error("counts, limits, workers, and bootstrap replicates must be positive")
    if args.min_letters > args.max_letters:
        parser.error("min-letters cannot exceed max-letters")
    args.out_dir.mkdir(parents=True, exist_ok=False)

    config = {key: value for key, value in vars(args).items()
              if key not in {"out_dir", "workers", "trials", "base_seed",
                             "bootstrap_replicates", "trial_mode",
                             "opening_min_branches", "opening_selection_seed"}}
    seeds = list(range(args.base_seed, args.base_seed + args.trials))
    openings = (select_openings(config, args.trials, args.opening_min_branches,
                                args.opening_selection_seed)
                if args.trial_mode == "openings" else None)
    source_files = [
        ROOT / "experiments/controlled_pos_pruning.py",
        ROOT / "paper/evidence_paths.py",
        ROOT / "paper/provenance.py",
    ] + [ROOT / "llm_palindrome" / (name + ".py") for name in (
        "__init__", "centerout", "exhaustive", "pairs", "search",
        "sentence_plan", "syntax",
    )]
    inputs = {name: hashlib.sha256(evidence_path(name).read_bytes()).hexdigest()
              for name in ("inputs/brown.json.gz", "inputs/vocab30k.txt")}
    provenance = {
        "scope": "Controlled fixed-popped-state POS-pruning experiment.",
        "configuration": config,
        "seeds": seeds,
        "trial_mode": args.trial_mode,
        "openings": openings,
        "opening_selection": ({
            "minimum_immediate_structural_branches": args.opening_min_branches,
            "selection_seed": args.opening_selection_seed,
            "method": "lowest SHA-256(seed, NUL, opening word) ranks",
            "uses_pos_tags_or_search_outcomes": False,
        } if openings is not None else None),
        "workers": args.workers,
        "task_order": "balanced terminal-first/incremental-first by seed parity",
        "inputs_sha256": inputs,
        "source": snapshot(ROOT, source_files),
        "multiprocessing_start_method": "spawn",
        "fresh_process_per_arm": True,
        "environment": environment(),
    }
    (args.out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n")

    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=context,
            max_tasks_per_child=1) as pool:
        rows = list(pool.map(run_trial, _tasks(seeds, config, openings)))
    rows.sort(key=lambda row: (row["seed"], row["arm"]))
    with (args.out_dir / "trials.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = summarize(rows, bootstrap_replicates=args.bootstrap_replicates)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    report = results_markdown(config, summary)
    (args.out_dir / "RESULTS.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
