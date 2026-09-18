"""Run the recorded structural configuration using the included current source.

This is a new timed run, not a reconstruction of missing historical provenance.
Only the standard library is needed. Defaults match the recorded configuration.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.exhaustive import enumerate_palindromes
from llm_palindrome.pairs import acceptable_pair, split_at_mirror
from llm_palindrome.search import WordTries
from llm_palindrome.sentence_plan import SentencePlan
from paper.evidence_paths import evidence_path
from paper.provenance import environment, snapshot


def run_arm(name, tries, plan, args, rank):
    stats, rows, seen = {}, [], set()
    started = time.time()
    for units in enumerate_palindromes(
            tries, min_letters=args.min_letters, max_letters=args.max_letters,
            max_overhang=args.max_overhang, max_units=args.max_units,
            shard=rank, shards=args.shards, node_budget=args.node_budget,
            deadline=started + args.seconds_per_arm, shuffle_seed=rank,
            allow_state=plan.state_possible if name == "planned" else None,
            stats=stats):
        split = split_at_mirror(units)
        if split is None:
            continue
        left, right = split
        if not acceptable_pair(left, right, min_words=args.min_words):
            continue
        if not (plan.complete(left) and plan.complete(right)):
            continue
        key = (" ".join(left), " ".join(right))
        if key in seen:
            continue
        seen.add(key)
        rows.append({"left": key[0], "right": key[1]})
        if len(rows) >= args.max_hits:
            stats["stop_reason"] = "max_hits"
            break
    return {"arm": name, "rank": rank, "seed": rank,
            "nodes": stats.get("nodes", 0), "closures": stats.get("yielded", 0),
            "state_pruned": stats.get("state_pruned", 0), "hits": len(rows),
            "seconds": round(time.time() - started, 3),
            "stop_reason": stats.get("stop_reason", "unrecorded"), "rows": rows}


def worker(task):
    rank, args = task
    payload = json.loads(gzip.decompress(evidence_path("inputs/brown.json.gz").read_bytes()))
    raw = evidence_path("inputs/vocab30k.txt").read_text().split()[:args.vocab]
    words = [word for word in raw if word in payload["table"]]
    tries = WordTries(words)
    plan = SentencePlan(payload["table"], payload["shapes"], args.min_words, args.max_units // 2)
    arms = [run_arm(name, tries, plan, args, rank) for name in ("terminal", "planned")]
    result = {"rank": rank, "shards": args.shards, "vocab": len(words),
              "environment": environment(), "arms": arms}
    (args.out_dir / f"summary_r{rank:04d}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (("vocab", 30000), ("min-letters", 20), ("max-letters", 44),
                          ("min-words", 3), ("max-units", 18), ("max-overhang", 16),
                          ("node-budget", 20000000), ("max-hits", 10000), ("shards", 32)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--seconds-per-arm", type=float, default=600)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if min(args.shards, args.node_budget, args.max_hits, args.seconds_per_arm) <= 0:
        parser.error("shards, budgets, and hit limits must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    source_files = [ROOT / "paper/rerun_search.py", ROOT / "paper/evidence_paths.py",
                    ROOT / "paper/provenance.py"]
    source_files += [ROOT / "llm_palindrome" / (name + ".py") for name in
                     ("__init__", "exhaustive", "pairs", "search", "sentence_plan", "syntax", "centerout", "lexicon")]
    config = {key: value for key, value in vars(args).items() if key != "out_dir"}
    inputs = {name: hashlib.sha256(evidence_path(name).read_bytes()).hexdigest()
              for name in ("inputs/brown.json.gz", "inputs/vocab30k.txt")}
    (args.out_dir / "provenance.json").write_text(json.dumps({
        "scope": "New run of the recorded configuration with current source.",
        "configuration": config, "arm_order": ["terminal", "planned"],
        "inputs_sha256": inputs, "source": snapshot(ROOT, source_files),
        "multiprocessing_start_method": "spawn",
    }, indent=2) + "\n")
    with ProcessPoolExecutor(max_workers=args.shards, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(worker, [(rank, args) for rank in range(args.shards)]))
    aggregate = {}
    for name in ("terminal", "planned"):
        arms = [arm for result in results for arm in result["arms"] if arm["arm"] == name]
        pairs = sorted({(row["left"], row["right"]) for arm in arms for row in arm["rows"]})
        aggregate[name] = {key: sum(arm[key] for arm in arms)
                           for key in ("nodes", "closures", "state_pruned", "seconds")}
        aggregate[name].update(hits=len(pairs), pairs=[{"left": left, "right": right} for left, right in pairs])
    (args.out_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    print(json.dumps({name: {key: value for key, value in row.items() if key != "pairs"}
                      for name, row in aggregate.items()}, indent=2))


if __name__ == "__main__":
    main()
