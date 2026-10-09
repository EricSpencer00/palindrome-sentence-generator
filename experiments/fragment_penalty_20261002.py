"""Frozen, paired fragment-penalty experiment. Substantive search runs on AWS.

The intervention changes only the search scorer. It charges a fixed amount for
each occurrence of six trigrams selected from the *previous* 52-text collection.
An exact potential difference handles temporary joins that disappear during
outside-in growth. No judge, test-set statistics, or new output tunes the scorer.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import itertools
import json
import multiprocessing
from pathlib import Path
import platform
import random
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "runs/fragment-penalty-20261002"
OLD = ROOT / "runs/directional-search-resource-sensitivity-20260925/manifest.json"
DATA = ROOT / "paper/fig/diagnostic-data.json"
STRENGTHS = (0, 4, 16)
SEEDS = tuple(range(7100, 7120))
METHODS = ("outside_in_norvig_hoey_adaptation", "center_out_project")
BANDS = ((30, 49), (50, 79), (80, 119))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def tokens(text):
    return tuple(re.findall(r"[a-z]+", text.lower()))


def trigrams(words):
    return [tuple(words[i:i+3]) for i in range(len(words)-2)]


class FragmentScorer:
    """Zipf objective minus lambda times the current concatenation's potential."""
    def __init__(self, base, fragments, strength):
        self.base = base
        self.fragments = frozenset(tuple(x) for x in fragments)
        self.strength = strength
        self.calls = 0

    def potential(self, seq):
        return sum(x in self.fragments for x in trigrams(seq))

    def change(self, left, right, placement, growth):
        seq = left + right
        index = (0 if growth == "prepend" else len(left)-1) if placement == "L" else (
            len(left) if growth == "prepend" else len(seq)-1)
        # Only windows touching the inserted word or the old gap change.
        after = seq[max(0, index-2):index+3]
        before = seq[max(0, index-2):index] + seq[index+1:index+3]
        return self.potential(after) - self.potential(before)

    def word_delta(self, left, right, placement, word, growth):
        self.calls += 1
        base = self.base.word_delta(left, right, placement, word, growth)
        if self.strength == 0:
            return base
        return base - self.strength * self.change(left, right, placement, growth)


def freeze():
    old = json.loads(OLD.read_text())
    data = json.loads(DATA.read_text())
    fragments = [phrase.split() for phrase, _ in data["statistics"]["frequent_trigrams"]]
    sources = ["experiments/fragment_penalty_20261002.py",
               "experiments/paired_directional_search_resource_sensitivity_20260925.py",
               "tests/test_fragment_penalty_20261002.py",
               "paper/fig/diagnostic-data.json",
               "runs/directional-search-resource-sensitivity-20260925/manifest.json"]
    sources += [p.relative_to(ROOT).as_posix() for p in (ROOT / "llm_palindrome").glob("*.py")]
    sources += ["data/lexicon.txt", "data/known_palindromes.json",
                "experiments/fragment-penalty-requirements.txt"]
    jobs = []
    for seed, band, method in itertools.product(SEEDS, range(3), METHODS):
        order = list(STRENGTHS)
        random.Random(f"20261002:{seed}:{band}:{method}").shuffle(order)
        for position, strength in enumerate(order):
            jobs.append({"seed": seed, "band_id": band+1,
                         "min_letters": BANDS[band][0], "max_letters": BANDS[band][1],
                         "method": method, "penalty": strength,
                         "resource_configuration": "escalated_v4",
                         "block": f"{seed}:{band+1}:{method}", "order": position+1,
                         "id": f"s{seed}-b{band+1}-{'out' if method == METHODS[0] else 'center'}-p{strength}"})
    manifest = {
        "experiment": "fragment-penalty-20261002", "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_new_search", "seeds": SEEDS, "strengths": STRENGTHS,
        "fragments": fragments, "fragment_source": "six most prevalent trigrams in the prior 52 distinct texts; counts and alphabetical tie-break frozen in diagnostic-data.json",
        "vocabulary": old["vocabulary"], "settings": old["resource_configurations"]["escalated_v4"],
        "source_sha256": {s: sha(ROOT/s) for s in sorted(sources)}, "jobs": jobs,
        "design": {
            "sample": "20 fresh seeds, 3 length bands, 2 native algorithms, 3 penalty strengths = 360 jobs; 120 jobs per strength",
            "allocation": "each seed-band-method block executes all 3 strengths sequentially in seeded random order on the same worker",
            "intervention": "add -lambda*(F(child)-F(parent)) to existing Zipf delta; F counts occurrences of six frozen trigrams, including overlapping occurrences and current join windows; telescopes to -lambda*F(final)",
            "selection": "maximize the resulting accumulated score per normalized letter; identical baseline admission gates; model ratings never select outputs",
            "budget": "same 180-second cooperative wall budget, beam 256, candidate 800, per-parent 32, 400 steps; actual elapsed time and scorer calls retained; fixed wall time is not equal expansion count",
            "primary_endpoints": ["exact basic-valid yield / all 120 requested jobs per arm", "maximum prevalence of ANY trigram among distinct outputs (not just targeted fragments); report count and denominator", "mean pairwise Jaccard distance between distinct outputs' trigram sets"],
            "secondary_endpoints": ["distinct surfaces / successful jobs", "fraction of distinct outputs with any targeted trigram", "strict mechanical admission / all jobs", "blinded Qwen3 32B grammaticality and meaning, using the existing 0-3 rubric and prose/shuffle control gate; model evidence only"],
            "uncertainty": "20,000 paired seed-cluster percentile bootstrap replicates (seed 20261002), keeping methods, bands and arms together; conditional diversity recomputed after deduplicating within each resample; intervals describe search-seed variability, not semantic-task or human-population uncertainty",
            "comparisons": "both fixed strengths versus baseline; report all arms, methods and bands, successes and failures; no winner selection or confirmatory p-values",
            "falsification": "penalty can reduce yield, replace old fragments with new repeated fragments, or fail to improve model-rated language quality; all outcomes retained",
            "adaptive_scope": "intervention motivated by previously inspected archive; fresh held-out seed block fixed before running; not a registered trial or a human readability study",
        },
    }
    path = OUT / "manifest.json"
    if path.exists():
        raise FileExistsError("refusing to overwrite the frozen protocol")
    write(path, manifest)
    print(json.dumps({"manifest": str(path), "sha256": sha(path), "jobs": len(jobs), "fragments": fragments}))


_TRIES = _BASE = _MANIFEST = None


def init_worker(manifest):
    global _TRIES, _BASE, _MANIFEST
    from llm_palindrome.search import WordTries
    from llm_palindrome.generate import ZipfScorer
    _MANIFEST = manifest
    _TRIES = WordTries(manifest["vocabulary"]["words"])
    _BASE = ZipfScorer()


def run_block(jobs):
    from experiments.paired_directional_search_resource_sensitivity_20260925 import run_job
    rows = []
    for job in jobs:
        scorer = FragmentScorer(_BASE, _MANIFEST["fragments"], job["penalty"])
        started = time.monotonic()
        try:
            row = run_job(job, _TRIES, scorer)
        except Exception as exc:
            row = {**job, "text": None, "status": "execution_error",
                   "error": f"{type(exc).__name__}: {exc}", "elapsed_seconds": time.monotonic()-started}
        row["scorer_calls"] = scorer.calls
        row["target_fragment_occurrences"] = scorer.potential(tokens(row["text"])) if row.get("text") else None
        rows.append(row)
    return rows


def run(workers):
    import importlib.metadata
    from experiments.paired_directional_search_resource_sensitivity_20260925 import RESOURCE_CONFIGURATIONS
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert platform.system() == "Linux", "substantive search is restricted to the AWS Linux worker"
    assert importlib.metadata.version("wordfreq") == "3.1.1"
    assert manifest["settings"] == RESOURCE_CONFIGURATIONS["escalated_v4"]
    for path, expected in manifest["source_sha256"].items():
        assert sha(ROOT/path) == expected, f"frozen source changed: {path}"
    vocab = manifest["vocabulary"]
    assert hashlib.sha256(("\n".join(vocab["words"])+"\n").encode()).hexdigest() == vocab["filtered_word_sha256"]
    output = OUT / "results.json"
    if output.exists():
        raise FileExistsError("refusing to replace experiment results")
    blocks = {}
    for job in manifest["jobs"]:
        blocks.setdefault(job["block"], []).append(job)
    block_list = list(blocks.values())
    random.Random(20261002).shuffle(block_list)
    result = {"manifest_sha256": sha(manifest_path), "status": "running", "workers": workers,
              "started_utc": datetime.now(timezone.utc).isoformat(), "runtime": {"python": platform.python_version(), "platform": platform.platform(), "wordfreq": importlib.metadata.version("wordfreq")}, "results": []}
    write(output, result)
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork"),
                             initializer=init_worker, initargs=(manifest,)) as pool:
        futures = [pool.submit(run_block, block) for block in block_list]
        for future in as_completed(futures):
            result["results"].extend(future.result())
            result["results"].sort(key=lambda r: r["id"])
            result["elapsed_seconds"] = time.monotonic()-start
            write(output, result)
            print(json.dumps({"completed": len(result["results"]), "of": len(manifest["jobs"]), "seconds": round(result["elapsed_seconds"])}), flush=True)
    result["status"] = "complete"
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    assert {r["id"] for r in result["results"]} == {r["id"] for r in manifest["jobs"]}
    write(output, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "run"])
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    freeze() if args.action == "freeze" else run(args.workers)
