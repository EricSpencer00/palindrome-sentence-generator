"""Offline local-word-order diagnostics; never a grammar or meaning score.

Score fresh search outcomes, selected historical exemplars, credited Reddit
palindromes, and ordinary prose using one fixed Norvig count_2w model. Preserve
all search failures and prose controls. The count table is not a held-out judge.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.bigram import BigramModel

DEFAULT_BANDS = ((30, 79), (80, 159), (160, 319), (320, 639), (640, 999), (1000, 1499))
METRICS = ("local_order_gain_nats_per_pair", "attested_pair_rate", "repeated_word_rate",
           "repeated_bigram_rate", "repeated_trigram_rate", "letters", "words")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def sentences(text: str) -> list[list[str]]:
    """Conservative .!?/newline boundaries; contractions split into ASCII words."""
    return [words for part in re.split(r"[.!?\n]+", text)
            if (words := re.findall(r"[a-z]+", part.lower()))]


def own_word_shuffles(parts: list[list[str]], count: int, seed: int = 20261009):
    """Preserve the whole word multiset and every sentence's word count."""
    key = json.dumps(parts, separators=(",", ":"))
    rng = random.Random(int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "big"))
    flat = [word for part in parts for word in part]
    for _ in range(count):
        mixed = list(flat)
        rng.shuffle(mixed)
        cursor, output = 0, []
        for part in parts:
            output.append(mixed[cursor:cursor + len(part)])
            cursor += len(part)
        yield output


def mean(values):
    present = [v for v in values if v is not None]
    return statistics.fmean(present) if present else None


def ngrams(parts: list[list[str]], n: int) -> list[tuple[str, ...]]:
    return [tuple(part[i:i + n]) for part in parts for i in range(len(part) - n + 1)]


def repeat_rate(values) -> float:
    return (len(values) - len(set(values))) / len(values) if values else 0.0


def score_parts(model: BigramModel, parts: list[list[str]]) -> float | None:
    return mean(model.forward(a, b) for a, b in ngrams(parts, 2))


def features(text: str, model: BigramModel, reddit_tapes: set[str],
             known_tapes: set[str], *, shuffles: int = 32) -> dict:
    if shuffles < 1:
        raise ValueError("at least one shuffle is required")
    parts = sentences(text)
    words = [word for part in parts for word in part]
    pairs = ngrams(parts, 2)
    tape = normalized(text)
    unsupported = sorted({c for c in text if c.isalpha() and not c.isascii()})
    observed = score_parts(model, parts)
    baseline = mean(score_parts(model, s) for s in own_word_shuffles(parts, shuffles))
    return {
        "letters": len(tape), "words": len(words), "sentence_word_counts": list(map(len, parts)),
        "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest(),
        "exact_palindrome": bool(tape) and not unsupported and tape == tape[::-1],
        "unsupported_letters": unsupported,
        "scored_pair_count": len(pairs), "shuffle_count": shuffles,
        "norvig_mean_logprob_nats_per_pair": observed,
        "own_shuffle_mean_logprob_nats_per_pair": baseline,
        "local_order_gain_nats_per_pair": observed - baseline if observed is not None and baseline is not None else None,
        "attested_pair_rate": mean(int(model.observed(a, b)) for a, b in pairs),
        "repeated_word_rate": repeat_rate(words),
        "repeated_bigram_rate": repeat_rate(pairs),
        "repeated_trigram_rate": repeat_rate(ngrams(parts, 3)),
        "literal_association_occurrences": words.count("association"),
        "literal_one_poem_association_occurrences": ngrams(parts, 3).count(("one", "poem", "association")),
        "in_reddit_exact_lookup": bool(tape) and tape in reddit_tapes,
        "in_known_catalogue_lookup": bool(tape) and tape in known_tapes,
        "absent_from_both_lookups": bool(tape) and tape not in reddit_tapes and tape not in known_tapes,
    }


def band_label(band) -> str:
    return f"{band[0]}-{band[1]}"


def text_band(text: str, bands) -> str:
    letters = len(normalized(text))
    return next((band_label(b) for b in bands if b[0] <= letters <= b[1]), "outside_search_bands")


def prose_rows(payload, key=None) -> list[dict]:
    """Use explicit source labels; never infer prose from non-palindromicity."""
    items = payload if isinstance(payload, list) else payload.get("items", payload.get("rows", []))
    keys = {r["id"]: r for r in key} if key is not None else {}
    if len(keys) != len(key or []):
        raise ValueError("duplicate prose control key ID")
    result = []
    for row in items:
        joined = {**keys.get(row.get("id"), {}), **row}
        tags = [joined.get("kind"), joined.get("source"), joined.get("type")]
        labels = " ".join(str(t) for t in tags).lower()
        if ("prose" in labels or "brown" in labels) and "shuffle" not in labels:
            text = joined.get("text", joined.get("surface"))
            if isinstance(text, str) and text.strip():
                result.append({**joined, "text": text})
    if not result:
        raise ValueError("no explicitly labelled ordinary prose texts in controls input")
    return result


def analyze(results: dict, current: dict, reddit: dict, controls: list[dict],
            known: list, model: BigramModel, *, shuffles: int = 32) -> dict:
    jobs = results.get("results", [])
    if len({r["id"] for r in jobs}) != len(jobs):
        raise ValueError("duplicate search job IDs")
    bands = sorted({(int(r["min_letters"]), int(r["max_letters"])) for r in jobs}) or list(DEFAULT_BANDS)
    if any(lo > hi for lo, hi in bands) or any(bands[i][1] >= bands[i + 1][0] for i in range(len(bands) - 1)):
        raise ValueError("invalid or overlapping search bands")
    reddit_tapes = {normalized(r["text"]) for r in reddit["exact"]}
    known_tapes = {normalized(r if isinstance(r, str) else r["text"]) for r in known}
    cache = {}
    def measure(text):
        if text not in cache:
            cache[text] = features(text, model, reddit_tapes, known_tapes, shuffles=shuffles)
        return cache[text]
    rows = []
    for job in jobs:
        selected = job.get("selected")
        text = selected.get("text") if isinstance(selected, dict) else None
        row = {"id": job["id"], "cohort": f"new/{job['direction']}/{job['inventory']}",
               "band": band_label((job["min_letters"], job["max_letters"])),
               "text": text, "search_outcome": job, "features": measure(text) if text else None}
        longest = job.get("longest_exact")
        if isinstance(longest, dict) and longest.get("text"):
            row["longest_exact_features"] = measure(longest["text"])
        rows.append(row)
    for i, source in enumerate(current.get("results", [])):
        text = source.get("surface", source.get("text"))
        if not isinstance(text, str) or not text.strip():
            continue
        rows.append({"id": source.get("id", f"current-{i}"), "cohort": "current_selected",
                     "band": text_band(text, bands), "text": text, "source_record": source,
                     "features": measure(text)})
    for i, source in enumerate(reddit["exact"]):
        text = source["text"]
        rows.append({"id": source.get("sha256", f"reddit-{i}"), "cohort": "reddit_credited",
                     "band": text_band(text, bands), "text": text, "sources": source.get("sources", []),
                     "source_record": source, "features": measure(text)})
    for i, source in enumerate(controls):
        text = source["text"]
        rows.append({"id": source.get("id", f"prose-{i}"), "cohort": "ordinary_prose_control",
                     "band": text_band(text, bands), "text": text, "source_record": source,
                     "features": measure(text)})
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["band"], row["cohort"])].append(row)
    cohorts = sorted({r["cohort"] for r in rows})
    summaries = []
    for band in [band_label(b) for b in bands] + ["outside_search_bands"]:
        for cohort in cohorts:
            group = grouped[(band, cohort)]
            output = [r for r in group if r["features"] is not None]
            exact = [r for r in output if r["features"]["exact_palindrome"]]
            eligible = output if cohort == "ordinary_prose_control" else exact
            summary = {"band": band, "cohort": cohort, "records": len(group),
                       "selected_outputs": len(output), "exact_outputs": len(exact),
                       "non_palindrome_outputs": len(output) - len(exact),
                       "no_selected_output": len(group) - len(output),
                       "metric_rows": len(eligible),
                       "metric_population": "all ordinary prose" if cohort == "ordinary_prose_control" else "exact selected outputs",
                       "distinct_normalized_texts": len({r["features"]["normalized_sha256"] for r in eligible})}
            if cohort.startswith("new/"):
                summary["selected_output_rate_per_requested_job"] = len(output) / len(group) if group else None
                summary["exact_output_rate_per_requested_job"] = len(exact) / len(group) if group else None
                summary["terminal_status_counts"] = dict(Counter(r["search_outcome"]["status"] for r in group))
                summary["jobs_with_any_exact_closure"] = sum(bool(r.get("longest_exact_features", {}).get("exact_palindrome")) for r in group)
            summary["means"] = {m: mean(r["features"][m] for r in eligible) for m in METRICS}
            summary["literal_association_output_count"] = sum(r["features"]["literal_association_occurrences"] > 0 for r in eligible)
            summary["absent_from_both_lookups_count"] = sum(r["features"]["absent_from_both_lookups"] for r in eligible)
            summaries.append(summary)
    return {
        "status": "completed_offline_diagnostic", "search_status": results.get("status"),
        "method": {"model": "llm_palindrome.bigram.BigramModel; fixed Norvig count_2w",
                   "shuffles": shuffles, "shuffle_seed": 20261009,
                   "shuffle": "same words globally shuffled; each .!?/newline sentence word count preserved",
                   "score": "mean observed forward log probability minus own-shuffle mean, within-sentence pairs only",
                   "novelty": "normalized exact lookup against credited Reddit sample and local known catalogue"},
        "limits": [
            "These features measure local word order and repetition, not grammar, meaning, or human readability.",
            "Norvig count_2w is independent of these measurements but is not claimed held-out from search or source texts.",
            "Current results are selected exemplars, and Reddit is a convenience sample; neither is a random baseline.",
            "Length bands align comparisons but do not exactly match lengths or remove selection bias.",
            "All requested search outcomes remain in denominators; longest closure is separate from the search-selected output.",
            "Ordinary prose is retained regardless of palindrome status; punctuation heuristics may split abbreviations.",
            "Absence from two finite lookups does not establish originality or ownership.",
        ], "bands": [list(b) for b in bands], "rows": rows, "summaries": summaries,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--current", type=Path, default=ROOT / "paper/week_results.json")
    ap.add_argument("--reddit", type=Path, default=ROOT / "data/reddit-palindromes/2026-10-09/audit/audit.json")
    ap.add_argument("--controls", type=Path, default=ROOT / "runs/argonne-length-readability-20261009/controls.json")
    ap.add_argument("--control-key", type=Path)
    ap.add_argument("--known", type=Path, default=ROOT / "data/known_palindromes.json")
    ap.add_argument("--bigram-file", "--bigrams", dest="bigrams", type=Path, default=ROOT / "data/count_2w.txt")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--shuffles", type=int, default=32)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    paths = {k: getattr(args, k) for k in ("results", "current", "reddit", "controls", "known", "bigrams")}
    if args.control_key:
        paths["control_key"] = args.control_key
    data = {k: json.loads(p.read_text()) for k, p in paths.items() if k != "bigrams"}
    report = analyze(data["results"], data["current"], data["reddit"],
                     prose_rows(data["controls"], data.get("control_key")), data["known"],
                     BigramModel.from_file(str(args.bigrams)), shuffles=args.shuffles)
    report["inputs"] = {k: {"path": str(p), "sha256": digest(p)} for k, p in paths.items()}
    if isinstance(data["controls"], dict):
        report["control_provenance"] = {k: v for k, v in data["controls"].items() if k not in ("items", "rows")}
    report["current_selection_note"] = data["current"].get("selection_note")
    report["script_sha256"] = digest(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": report["status"], "rows": len(report["rows"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
