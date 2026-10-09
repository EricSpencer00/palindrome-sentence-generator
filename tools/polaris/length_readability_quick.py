"""Bounded CPU search diagnostic; language quality requires separate evaluation.

Run only on an authorized compute node, for example:
python tools/polaris/length_readability_quick.py --out runs/quick-length-20261009
Resources are initialized once; at most 12 forked jobs run at a time. Each job
has a cooperative search budget and an independent parent-enforced deadline.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import platform
import random
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from llm_palindrome.bigram import BigramModel
from llm_palindrome.centerout import centerout_search
from llm_palindrome.overhang import DebtIndex, OverhangAware
from llm_palindrome.phrases import build_inventory
from llm_palindrome.scoring import unit_words
from llm_palindrome.search import State, WordTries, beam_search, unit_letters
from llm_palindrome.semantic import RankOrderScorer

BANDS = ((30, 79), (80, 159), (160, 319), (320, 639), (640, 999), (1000, 1499))
SEEDS = (100900, 100901, 100902)
CONFIG = {
    "vocab_size": 30000, "phrase_count": 2000, "beam_width": 48,
    "candidate_limit": 200, "per_parent": 8, "max_steps": 1500,
    "max_words": 1024, "max_overhang": 24, "diversity": 0.4,
    "search_seconds": 20.0, "hard_seconds": 25.0,
    "order_weight": 1.0, "length_weight": 0.2, "reuse_weight": -2.0,
    "debt_weight": 2.0, "dead_penalty": 30.0, "strict_max_words": 128,
}


def normalized(text):
    if any(c.isalpha() and not c.isascii() for c in text):
        raise ValueError("unsupported non-ASCII letters")
    return "".join(c.lower() for c in text if c.isascii() and c.isalpha())


def surface(units):
    value = " ".join(unit_words(units))
    return value[:1].upper() + value[1:] + "." if value else ""


def unit_counts(units):
    return {"letters": sum(len(unit_letters(u)) for u in units),
            "words": sum(len(u.split()) for u in units), "units": len(units),
            "phrase_units": sum(len(u.split()) > 1 for u in units)}


def within_caps(units, max_letters, max_words):
    counts = unit_counts(units)
    return counts["letters"] <= max_letters and counts["words"] <= max_words


def make_jobs():
    jobs = []
    rng = random.Random(20261009)
    for low, high in BANDS:
        for seed in SEEDS:
            block = [{"id": f"{direction}-{inventory}-n{low}-s{seed}",
                      "direction": direction, "inventory": inventory,
                      "min_letters": low, "max_letters": high, "seed": seed}
                     for direction in ("outside_in", "center_out")
                     for inventory in ("words", "words_phrases")]
            rng.shuffle(block)
            jobs.extend(block)
    return jobs


class CountedRankScorer(RankOrderScorer):
    """Keep rank/order features, counting repeated constituent words correctly."""
    calls = 0

    def word_delta(self, left, right, placement, word, growth):
        self.calls += 1
        score = super().word_delta(left, right, placement, word, growth)
        inner = word.split()
        old_uses = sum((left + right).count(w) for w in inner) - len(inner)
        after, inserted = Counter(unit_words(left + right)), Counter(inner)
        uses = sum(n * (after[w] - n) + n * (n - 1) / 2 for w, n in inserted.items())
        return score + self.reuse_weight * (uses - old_uses)


def load_resources(vocab_path, bigram_path, config):
    raw = vocab_path.read_text(encoding="utf-8").split()[:config["vocab_size"]]
    words = list(dict.fromkeys(w.lower() for w in raw if w.isascii() and w.isalpha()))
    if not words:
        raise ValueError("empty frozen vocabulary")
    bigrams = BigramModel.from_file(str(bigram_path), vocab=words)
    phrases = build_inventory(str(bigram_path), vocab=words, top_n=config["phrase_count"])
    inventories = {"words": words, "words_phrases": words + phrases}
    return {"words": words, "bigrams": bigrams, "phrases": phrases,
            "tries": {name: WordTries(units) for name, units in inventories.items()}}


def exact_record(units):
    text = surface(units)
    tape = normalized(text)
    return {"text": text, **unit_counts(units),
            "independent_exact": bool(tape) and tape == tape[::-1],
            "normalized_sha256": hashlib.sha256(tape.encode("ascii")).hexdigest(),
            "source": "fresh_search_diagnostic", "readability_evidence": "not_evaluated"}


def run_job(job, resources, config, on_progress=None):
    started = time.monotonic()
    deadline = started + config["search_seconds"]
    counters = Counter()
    longest = None
    tries = resources["tries"][job["inventory"]]
    base = CountedRankScorer(resources["words"], resources["bigrams"],
                            order_weight=config["order_weight"],
                            length_weight=config["length_weight"], reuse_weight=config["reuse_weight"])
    scorer = OverhangAware(base, DebtIndex(tries, limit=96),
                          debt_weight=config["debt_weight"], dead_penalty=config["dead_penalty"])

    def allow_state(left, right):
        counts = unit_counts(left + right)
        counters["state_gate_calls"] += 1
        if counts["letters"] > job["max_letters"]:
            counters["letter_cap_rejections"] += 1
            return False
        if counts["words"] > config["max_words"]:
            counters["word_cap_rejections"] += 1
            return False
        return True

    def allow_closed(left, right):
        nonlocal longest
        counts = unit_counts(left + right)
        if not job["min_letters"] <= counts["letters"] <= job["max_letters"]:
            return False
        if counts["words"] > config["max_words"]:
            return False
        counters["in_band_closures"] += 1
        if longest is None or counts["letters"] > longest["letters"]:
            candidate = exact_record(left + right)
            if not candidate["independent_exact"]:
                raise AssertionError("closure failed independent exactness")
            longest = candidate
            if on_progress:
                on_progress({"longest_exact": longest, "counters": dict(counters),
                             "scorer_calls": base.calls,
                             "elapsed_seconds": time.monotonic() - started})
        return True

    common = {key: config[key] for key in ("beam_width", "max_steps", "candidate_limit", "per_parent", "diversity")}
    common.update(min_letters=job["min_letters"], seed=job["seed"], deadline=deadline,
                  allow_state=allow_state, allow_closed=allow_closed)
    if job["direction"] == "outside_in":
        if config["max_overhang"] != 24:
            raise ValueError("outside-in search currently fixes overhang at 24")
        units = beam_search(tries, scorer, initial_state=State(0.0, (), (), "", "L", 0.0), **common)
    elif job["direction"] == "center_out":
        units = centerout_search(tries, scorer, center="", max_overhang=config["max_overhang"], **common)
    else:
        raise ValueError("unknown direction")
    search_elapsed = time.monotonic() - started
    timed_out = time.monotonic() >= deadline
    selected = exact_record(units) if units else None
    if selected and (not selected["independent_exact"] or not within_caps(units, job["max_letters"], config["max_words"])
                     or selected["letters"] < job["min_letters"]):
        raise AssertionError("returned output failed independent length/exactness check")
    result = {**job, "status": ("deadline_with_output" if timed_out else "completed_with_output") if selected
              else ("deadline_no_output" if timed_out else "completed_no_output"),
              "selected": selected, "longest_exact": longest, "deadline_reached": timed_out,
              "search_elapsed_seconds": search_elapsed, "counters": dict(counters),
              "scorer_calls": base.calls, "strict_status": "no_output",
              "strict_checks": None, "strict_admitted": None,
              "cap_observability": "word/letter rejections counted; native step exhaustion and overhang pruning not instrumented"}
    # Publish construction before optional stricter checks, which can be costly.
    if on_progress:
        on_progress({**result, "strict_status": "pending"})
    if selected:
        if selected["words"] > config["strict_max_words"]:
            result["strict_status"] = "not_run_word_limit"
        else:
            try:
                from llm_palindrome.admission import mechanical_admission_checks
                checks = mechanical_admission_checks(selected["text"], min_letters=job["min_letters"], max_letters=job["max_letters"])
                result.update(strict_status="checked", strict_checks=checks, strict_admitted=all(checks.values()))
            except Exception as exc:
                result.update(strict_status="error", strict_error=f"{type(exc).__name__}: {exc}")
    result["elapsed_seconds"] = time.monotonic() - started
    return result


def failure_row(job, status, elapsed, error=None, progress=None):
    result = {**job, "status": status, "selected": None, "longest_exact": None,
              "elapsed_seconds": elapsed, "strict_status": "not_completed",
              "strict_checks": None, "strict_admitted": None,
              "deadline_reached": status == "hard_deadline"}
    if progress:
        result["partial_progress"] = progress
        result["longest_exact"] = progress.get("longest_exact")
    if error:
        result["error"] = error
    return result


def _child(connection, job, resources, config):
    started = time.monotonic()
    try:
        result = run_job(job, resources, config,
                         on_progress=lambda value: connection.send(("progress", value)))
    except Exception as exc:
        result = failure_row(job, "execution_error", time.monotonic() - started,
                             f"{type(exc).__name__}: {exc}")
    try:
        connection.send(("result", result))
    finally:
        connection.close()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_jobs(jobs, resources, config, workers, out, child_target=_child):
    """Supervise separate forked processes; a hung job cannot hold the pool."""
    if workers < 1:
        raise ValueError("workers must be positive")
    context = mp.get_context("fork")
    pending = list(jobs)
    active, results = {}, []
    out.mkdir(parents=True, exist_ok=True)

    def checkpoint():
        atomic_json(out / "progress.json", {"completed": len(results), "total": len(jobs),
                    "active": [entry["job"]["id"] for entry in active.values()], "results": results})

    try:
        while pending or active:
            while pending and len(active) < workers:
                job = pending.pop(0)
                receiver, sender = context.Pipe(duplex=False)
                process = context.Process(target=child_target, args=(sender, job, resources, config))
                started = time.monotonic()
                try:
                    process.start()
                except Exception as exc:
                    receiver.close()
                    sender.close()
                    results.append(failure_row(job, "execution_error", time.monotonic() - started,
                                               f"process_start: {type(exc).__name__}: {exc}"))
                    checkpoint()
                    continue
                sender.close()
                active[job["id"]] = dict(job=job, process=process, connection=receiver,
                                        started=started, progress=None)
            for job_id, entry in list(active.items()):
                result = None
                connection, process = entry["connection"], entry["process"]
                while connection.poll():
                    try:
                        kind, value = connection.recv()
                    except EOFError:
                        break
                    if kind == "progress":
                        entry["progress"] = value
                    elif kind == "result":
                        result = value
                elapsed = time.monotonic() - entry["started"]
                if result is None and elapsed >= config["hard_seconds"]:
                    result = failure_row(entry["job"], "hard_deadline", elapsed, progress=entry["progress"])
                elif result is None and not process.is_alive():
                    # The final send can occur between the first poll and the
                    # liveness check. Drain it before reporting a worker exit.
                    while connection.poll():
                        try:
                            kind, value = connection.recv()
                        except EOFError:
                            break
                        if kind == "progress":
                            entry["progress"] = value
                        elif kind == "result":
                            result = value
                    if result is None:
                        result = failure_row(entry["job"], "worker_exit", elapsed,
                                             f"exitcode={process.exitcode}", entry["progress"])
                if result is not None:
                    if result["status"] == "execution_error" and entry["progress"]:
                        result["partial_progress"] = entry["progress"]
                        result["longest_exact"] = entry["progress"].get("longest_exact")
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=1)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=1)
                    connection.close()
                    del active[job_id]
                    results.append(result)
                    atomic_json(out / f"{job_id}.json", result)
                    checkpoint()
                    print(json.dumps({"completed": len(results), "total": len(jobs),
                                      "job": job_id, "status": result["status"]}), flush=True)
            if active:
                time.sleep(0.02)
    finally:
        for entry in active.values():
            process = entry["process"]
            if process.is_alive():
                process.terminate()
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join(timeout=1)
            entry["connection"].close()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vocab-file", type=Path, default=ROOT / "tools/polaris/payload/vocab30k.txt")
    parser.add_argument("--bigram-file", type=Path, default=ROOT / "data/count_2w.txt")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if platform.system() != "Linux":
        raise SystemExit("Run the full grid on an authorized Linux compute node.")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("refusing to overwrite an existing run")
    if not 1 <= args.workers <= 12:
        raise ValueError("workers must be between 1 and 12")
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    paths = [Path(__file__), args.vocab_file, args.bigram_file]
    paths += [ROOT / "llm_palindrome" / name for name in (
        "search.py", "centerout.py", "semantic.py", "scoring.py", "overhang.py", "phrases.py",
        "bigram.py", "admission.py", "lexicon.py", "validator.py")]
    paths += [ROOT / "data/lexicon.txt", ROOT / "data/known_palindromes.json"]
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "host": socket.gethostname(),
                "python": platform.python_version(), "config": CONFIG, "workers": args.workers,
                "jobs": make_jobs(), "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                "interpretation": "Exploratory construction diagnostic, not a readability result or new generator.",
                "scorer": "RankOrderScorer with constituent-word repetition correction plus lexical debt lookahead; native closure selection retained",
                "phrase_source": "Top attested count_2w pairs whose constituents are in the frozen vocabulary"}
    atomic_json(args.out / "manifest.json", manifest)
    try:
        resources = load_resources(args.vocab_file, args.bigram_file, CONFIG)
        setup_seconds = time.monotonic() - started
        atomic_json(args.out / "resources.json", {"words": len(resources["words"]),
                    "phrases": resources["phrases"], "setup_seconds": setup_seconds})
        results = run_jobs(manifest["jobs"], resources, CONFIG, args.workers, args.out / "jobs")
    except Exception as exc:
        atomic_json(args.out / "run-error.json", {"error": f"{type(exc).__name__}: {exc}",
                    "elapsed_seconds": time.monotonic() - started})
        raise
    atomic_json(args.out / "results.json", {"status": "completed_diagnostic", "results": results,
                "setup_seconds": setup_seconds, "elapsed_seconds": time.monotonic() - started,
                "terminal_counts": dict(Counter(row["status"] for row in results)),
                "readability_evidence": "not_evaluated", "human_evidence": "not_evaluated"})


if __name__ == "__main__":
    main()
