"""Small paired development probe for low fragment-penalty strengths.

This is an exploratory feasibility and structural-failure-diagnostics probe,
not a grammar or readability evaluation. It reuses the October 2
FragmentScorer and its frozen target trigrams, with a 256-word cap and a
60-second cooperative budget per job.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
import multiprocessing
from pathlib import Path
import platform
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "runs/fragment-penalty-tuning-20261008"
PRIOR = ROOT / "runs/fragment-penalty-20261002/manifest.json"
SEEDS = (9100, 9101)
BANDS = ((120, 159), (800, 999))
METHODS = ("outside_in", "center_out")
PENALTIES = (0, 1, 2, 4)
SETTINGS = {
    "beam_width": 256,
    "candidate_limit": 800,
    "per_parent": 32,
    "max_steps": 400,
    "diversity": 0.4,
    "max_overhang": 24,
    "max_words": 256,
    "cooperative_budget_seconds": 60,
}
BASIC_CHECKS = (
    "supported_ascii_letters", "nonempty", "exact_letter_palindrome",
    "length_band", "word_form", "lexicon_words", "ordinary_short_words",
    "not_forbidden_catalogue_control", "not_catalogue_family_derivative",
    "not_forbidden_catalogue_endpoint_scaffold", "absent_from_local_catalogue",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def freeze(output_path: Path | None = None) -> None:
    from experiments.fragment_penalty_20261002 import tokens

    prior = json.loads(PRIOR.read_text())
    source_paths = [
        "experiments/tune_fragment_penalty_20261008.py",
        "experiments/fragment_penalty_20261002.py",
        "experiments/paired_directional_search_resource_sensitivity_20260925.py",
        "llm_palindrome/search.py", "llm_palindrome/centerout.py",
        "llm_palindrome/generate.py", "llm_palindrome/admission.py",
        "llm_palindrome/lexicon.py", "llm_palindrome/shortwords.py",
        "llm_palindrome/scoring.py", "data/lexicon.txt",
        "data/known_palindromes.json", "paper/fig/diagnostic-data.json",
        "tools/polaris/payload/vocab30k.txt",
        "runs/fragment-penalty-20261002/manifest.json",
    ]
    source_paths.extend(
        path.relative_to(ROOT).as_posix()
        for path in sorted((ROOT / "llm_palindrome").glob("*.py"))
    )
    source_paths = sorted(set(source_paths))
    jobs = []
    for seed, band, method in itertools.product(SEEDS, BANDS, METHODS):
        order = list(PENALTIES)
        random.Random(f"20261008-tune:{seed}:{band[0]}:{method}").shuffle(order)
        block = f"seed-{seed}:band-{band[0]}:{method}"
        for position, penalty in enumerate(order, 1):
            jobs.append({
                "id": f"s{seed}-n{band[0]}-{method}-p{penalty}",
                "block": block, "seed": seed, "min_letters": band[0],
                "max_letters": band[1], "method": method,
                "penalty": penalty, "execution_order": position,
            })
    setup_attempts = []
    for attempt in (1, 2):
        archive = OUT / f"attempt-{attempt}/runs/fragment-penalty-tuning-20261008"
        manifest_file = archive / ("manifest.json" if attempt == 1 else "manifest-v2.json")
        result_file = archive / ("results.json" if attempt == 1 else "results-v2.json")
        if manifest_file.is_file() and result_file.is_file():
            initial_result = json.loads(result_file.read_text())
            setup_attempts.append({
                "attempt": attempt,
                "manifest_sha256": digest(manifest_file),
                "results_sha256": digest(result_file),
                "rows": len(initial_result.get("results", [])),
                "execution_error_rows": sum(
                    row.get("status") == "execution_error"
                    for row in initial_result.get("results", [])
                ),
                "errors": sorted({
                    row.get("error", "") for row in initial_result.get("results", [])
                    if row.get("error")
                }),
                "classification": "setup failure before any search expansions; excluded from the search-job denominator",
            })
    third_setup = OUT / "setup-attempt-3.json"
    if third_setup.is_file():
        setup_attempts.append(json.loads(third_setup.read_text()))
    manifest = {
        "experiment": "fragment-penalty-tuning-20261008",
        "protocol_revision": len(setup_attempts) + 1,
        "superseded_setup_attempts": setup_attempts,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_search",
        "objective_order": [
            "maximize basic-valid exact-output feasibility per penalty",
            "describe mechanical strict-gate residuals among selected outputs",
            "retain all failures and per-closure failure counters",
        ],
        "interpretation": (
            "Small adaptive development probe for parameter tuning. It does not "
            "establish readability, human preference, or generalization. No "
            "human ratings or language-model selection are included."
        ),
        "heldout_human": "reserved as a separate follow-up; this probe does not consume or claim held-out human evidence",
        "seeds": list(SEEDS), "bands": [list(x) for x in BANDS],
        "methods": list(METHODS), "penalties": list(PENALTIES),
        "settings": SETTINGS,
        "fragments": prior["fragments"],
        "fragment_source": "exact six-trigram list embedded in the frozen October 2 manifest",
        "vocabulary": prior["vocabulary"],
        "prior_manifest_sha256": digest(PRIOR),
        "source_sha256": {p: digest(ROOT / p) for p in sorted(source_paths)},
        "jobs": jobs,
        "design": (
            "2 seeds x 2 length bands x 2 native search methods x 4 penalties = "
            "32 jobs. Each seed-band-method block runs every penalty once in a "
            "deterministically randomized order. Penalty 0 is the original "
            "Zipf score through the same FragmentScorer wrapper."
        ),
        "budget": "60-second cooperative deadline per job; at most 32 jobs; two NUC workers",
    }
    path = output_path or (OUT / "manifest.json")
    if path.exists():
        raise FileExistsError("refusing to overwrite frozen protocol")
    save(path, manifest)
    print(json.dumps({"manifest_sha256": digest(path), "jobs": len(jobs)}))


def run_job(job: dict, manifest: dict) -> dict:
    from experiments.fragment_penalty_20261002 import FragmentScorer, tokens
    from llm_palindrome.admission import mechanical_admission_checks, normalize_letters
    from llm_palindrome.centerout import centerout_search
    from llm_palindrome.generate import ZipfScorer
    from llm_palindrome.search import State, WordTries, beam_search

    counters: Counter[str] = Counter()
    failed_basic: Counter[str] = Counter()
    failed_strict: Counter[str] = Counter()
    tries = _TRIES
    if tries is None:
        raise RuntimeError("worker trie was not initialized")
    scorer = FragmentScorer(ZipfScorer(), manifest["fragments"], job["penalty"])
    start = time.monotonic()
    deadline = start + SETTINGS["cooperative_budget_seconds"]

    def render(words):
        value = " ".join(words).strip()
        return value[:1].upper() + value[1:] + "." if value else ""

    def allow_state(left, right):
        words = tuple(left) + tuple(right)
        if len(words) > SETTINGS["max_words"]:
            counters["word_cap_rejections"] += 1
            return False
        if sum(map(len, words)) > job["max_letters"]:
            counters["letter_cap_rejections"] += 1
            return False
        return True

    def allow_closed(left, right):
        counters["eligible_exact_closure_candidates"] += 1
        words = tuple(left) + tuple(right)
        checks = mechanical_admission_checks(
            render(words), min_letters=job["min_letters"],
            max_letters=job["max_letters"],
        )
        for name, passed in checks.items():
            if not passed:
                failed_strict[name] += 1
                if name in BASIC_CHECKS:
                    failed_basic[name] += 1
        basic = all(checks.get(name, False) for name in BASIC_CHECKS)
        counters["basic_valid_closure_candidates"] += int(basic)
        if not basic:
            counters["rejected_closure_candidates"] += 1
        return basic

    common = {
        "min_letters": job["min_letters"], "seed": job["seed"],
        "allow_state": allow_state, "allow_closed": allow_closed,
        "deadline": deadline, "beam_width": SETTINGS["beam_width"],
        "max_steps": SETTINGS["max_steps"],
        "candidate_limit": SETTINGS["candidate_limit"],
        "per_parent": SETTINGS["per_parent"],
        "diversity": SETTINGS["diversity"],
    }
    if job["method"] == "outside_in":
        words = beam_search(
            tries, scorer, max_word_uses=None,
            initial_state=State(0.0, (), (), "", "L", 0.0), **common,
        )
    else:
        words = centerout_search(
            tries, scorer, center="", max_overhang=SETTINGS["max_overhang"],
            **common,
        )
    elapsed = time.monotonic() - start
    text = render(words)
    checks = mechanical_admission_checks(
        text, min_letters=job["min_letters"], max_letters=job["max_letters"],
    ) if text else {}
    letters = "".join(c.lower() for c in text if c.isascii() and c.isalpha())
    exact = bool(letters and letters == letters[::-1])
    basic = bool(text and all(checks.get(name, False) for name in BASIC_CHECKS))
    if text and (not exact or not basic):
        raise AssertionError("returned output failed independent exact/basic check")
    timed_out = time.monotonic() >= deadline
    return {
        **job,
        "status": (
            "deadline_best_so_far_output" if timed_out and basic else
            "deadline_no_valid_output" if timed_out else
            "selected_basic_valid_output" if basic else "no_basic_valid_output"
        ),
        "elapsed_seconds": elapsed,
        "deadline_reached": timed_out,
        "text": text if basic else None,
        "letters": len(letters), "word_count": len(words),
        "independently_exact": exact if basic else None,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest() if basic else None,
        "target_fragment_occurrences": scorer.potential(tokens(text)) if basic else None,
        "strict_admitted_selected_output": bool(basic and all(checks.values())) if basic else None,
        "strict_failed_checks_selected_output": sorted(k for k, v in checks.items() if not v) if basic else [],
        "checks": checks if basic else {},
        "counters": dict(sorted(counters.items())),
        "basic_failure_counters": dict(sorted(failed_basic.items())),
        "strict_failure_counters": dict(sorted(failed_strict.items())),
        "scorer_calls": scorer.calls,
        "readability_evidence": "not evaluated",
    }


_TRIES = None
_CURRENT_MANIFEST = None


def init_worker(manifest: dict) -> None:
    global _TRIES, _CURRENT_MANIFEST
    from llm_palindrome.search import WordTries
    _TRIES = WordTries(manifest["vocabulary"]["words"])
    _CURRENT_MANIFEST = manifest


def run(workers: int, manifest_path: Path | None = None,
        output_path: Path | None = None) -> None:
    if platform.system() != "Linux":
        raise RuntimeError("substantive search is restricted to the NUC Linux worker")
    if importlib.metadata.version("wordfreq") != "3.1.1":
        raise RuntimeError("expected pinned wordfreq 3.1.1")
    manifest_path = manifest_path or (OUT / "manifest.json")
    manifest = json.loads(manifest_path.read_text())
    for path, expected in manifest["source_sha256"].items():
        if digest(ROOT / path) != expected:
            raise ValueError(f"frozen source changed: {path}")
    prior = json.loads(PRIOR.read_text())
    if digest(PRIOR) != manifest["prior_manifest_sha256"]:
        raise ValueError("prior frozen manifest changed")
    words = manifest["vocabulary"]["words"]
    if hashlib.sha256(("\n".join(words) + "\n").encode()).hexdigest() != manifest["vocabulary"]["filtered_word_sha256"]:
        raise ValueError("frozen vocabulary hash mismatch")
    output = output_path or (OUT / "results.json")
    if output.exists():
        raise FileExistsError("refusing to overwrite results")
    blocks: dict[str, list[dict]] = {}
    for job in manifest["jobs"]:
        blocks.setdefault(job["block"], []).append(job)
    for block in blocks.values():
        if len(block) != len(PENALTIES) or set(j["penalty"] for j in block) != set(PENALTIES):
            raise AssertionError("invalid paired block")
    result = {
        "status": "running", "manifest_sha256": digest(manifest_path),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "workers": workers, "runtime": {
            "host": platform.node(), "python": platform.python_version(),
            "wordfreq": importlib.metadata.version("wordfreq"),
        },
        "results": [],
    }
    save(output, result)
    started = time.monotonic()
    block_list = list(blocks.values())
    random.Random(20261008).shuffle(block_list)
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("fork"),
        initializer=init_worker, initargs=(manifest,),
    ) as pool:
        future_blocks = {pool.submit(_run_block, block): block for block in block_list}
        for future in as_completed(future_blocks):
            try:
                rows = future.result()
            except Exception as exc:
                block = future_blocks[future]
                rows = [
                    {**job, "status": "execution_error", "text": None,
                     "error": f"{type(exc).__name__}: {exc}"}
                    for job in block
                ]
            result["results"].extend(rows)
            result["results"].sort(key=lambda row: row["id"])
            result["elapsed_wall_seconds"] = time.monotonic() - started
            save(output, result)
            print(json.dumps({"completed": len(result["results"]), "of": 32}), flush=True)
    if {r["id"] for r in result["results"]} != {j["id"] for j in manifest["jobs"]}:
        raise AssertionError("job/result ID mismatch")
    result["status"] = "complete" if all(r["status"] != "execution_error" for r in result["results"]) else "completed_with_errors"
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    result["elapsed_wall_seconds"] = time.monotonic() - started
    save(output, result)
    summarize(manifest, result, manifest_path, output)


def _run_block(jobs: list[dict]) -> list[dict]:
    if _TRIES is None:
        raise RuntimeError("worker was not initialized")
    # Scorer state is isolated for each job; all four conditions stay together.
    return [run_job(job, _CURRENT_MANIFEST) for job in jobs]


def summarize(manifest: dict, result: dict, manifest_path: Path,
              results_path: Path) -> None:
    by_penalty = {}
    for penalty in PENALTIES:
        rows = [r for r in result["results"] if r["penalty"] == penalty]
        by_penalty[str(penalty)] = {
            "requested": len(rows),
            "basic_valid_exact_outputs": sum(bool(r.get("text")) and r.get("independently_exact") is True for r in rows),
            "strict_admitted_selected_outputs": sum(r.get("strict_admitted_selected_output") is True for r in rows),
            "deadline_jobs": sum(bool(r.get("deadline_reached")) for r in rows),
            "execution_errors": sum(r["status"] == "execution_error" for r in rows),
            "status_counts": dict(sorted(Counter(r["status"] for r in rows).items())),
            "strict_failed_check_counts_on_selected_outputs": dict(sorted(
                (name, sum(name in r.get("strict_failed_checks_selected_output", []) for r in rows))
                for name in {n for r in rows for n in r.get("strict_failed_checks_selected_output", [])}
            )),
        }
    summary = {
        "experiment": manifest["experiment"],
        "manifest_sha256": digest(manifest_path),
        "results_sha256": digest(results_path),
        "status": result["status"],
        "objective_order": manifest["objective_order"],
        "aggregate_by_penalty": by_penalty,
        "caveat": manifest["interpretation"],
    }
    summary_path = results_path.with_name(results_path.name.replace("results", "summary", 1))
    save(summary_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.manifest)
    else:
        run(args.workers, args.manifest, args.output)
