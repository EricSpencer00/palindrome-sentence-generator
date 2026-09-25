"""Score the frozen exact examples with Brown word-order diagnostics.

These scores are a cheap local diagnostic, not a readability certificate. The
main comparison holds each passage's words fixed and measures how much its
observed order beats deterministic shuffles of those same words. Length-matched
intact Brown spans check that the diagnostic recognizes ordinary corpus order.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.audit_programmatic_readability import (  # noqa: E402
    BrownBigramModel,
    order_gain,
    tokens,
)


def ascii_tape(text: str) -> str:
    return re.sub(r"[^A-Za-z]", "", text).lower()


def outside_in_exact(text: str) -> bool:
    letters = [c.lower() for c in text if c.isascii() and c.isalpha()]
    left, right = 0, len(letters) - 1
    while left < right:
        if letters[left] != letters[right]:
            return False
        left += 1
        right -= 1
    return True


def verify_candidate(row: dict) -> dict:
    text = row.get("surface")
    if not isinstance(text, str) or not text:
        raise ValueError(f"{row.get('id', '<unknown>')} has no rendered surface")
    tape = ascii_tape(text)
    digest = hashlib.sha256(tape.encode("ascii")).hexdigest()
    checks = {
        "outside_in_ascii_scan": outside_in_exact(text),
        "normalized_reverse_equality": bool(tape) and tape == tape[::-1],
        "letter_count_matches_manifest": len(tape) == row.get("letters"),
        "normalized_sha256_matches_manifest": digest == row.get("normalized_sha256"),
        "manifest_independently_exact": row.get("independently_exact") is True,
    }
    if not all(checks.values()):
        raise ValueError(f"{row.get('id')} failed exactness audit: {checks}")
    manifest_word_count = len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)*", text))
    if manifest_word_count != row.get("metrics", {}).get("word_count"):
        raise ValueError(f"{row.get('id')} word count differs from frozen manifest")
    return {
        "checks": checks,
        "letters": len(tape),
        "manifest_word_count": manifest_word_count,
        "scorer_token_count": len(tokens(text)),
        "normalized_sha256": digest,
    }


def select_brown_window(sentences: Sequence[tuple[int, list[str]]], target_words: int,
                        occupied: Sequence[tuple[int, int]], control_id: str
                        ) -> tuple[int, int, int]:
    """Choose a closest non-overlapping half-open Brown sentence-index span."""
    if target_words < 1 or not sentences:
        raise ValueError("target_words and sentences must be nonempty")
    prefix = [0]
    for _, words in sentences:
        prefix.append(prefix[-1] + len(words))

    options: list[tuple[int, str, int, int, int]] = []
    for start in range(len(sentences)):
        goal = prefix[start] + target_words
        crossing = bisect_left(prefix, goal, lo=start + 1)
        for end in {max(start + 1, min(crossing - 1, len(sentences))),
                    max(start + 1, min(crossing, len(sentences)))}:
            first_sentence = sentences[start][0]
            after_last_sentence = sentences[end - 1][0] + 1
            if any(first_sentence < used_end and after_last_sentence > used_start
                   for used_start, used_end in occupied):
                continue
            count = prefix[end] - prefix[start]
            stable_tie = hashlib.sha256(
                f"{control_id}:{first_sentence}:{after_last_sentence}".encode()
            ).hexdigest()
            options.append((abs(count - target_words), stable_tie,
                            first_sentence, after_last_sentence, count))
    if not options:
        raise ValueError(f"could not find a non-overlapping Brown span for {control_id}")
    _, _, first, after_last, count = min(options)
    return first, after_last, count


def score_item(model: BrownBigramModel, item_id: str, text: str, *,
               source: str, seed: int, shuffles: int) -> dict:
    words = tokens(text)
    if not words:
        raise ValueError(f"{item_id} has no alphabetic words")
    observed, gain = order_gain(model, words, item_id, seed, shuffles)
    return {
        "id": item_id,
        "source": source,
        "word_count": len(words),
        "brown_bigram_logprob": observed,
        "brown_order_gain_vs_own_shuffle": gain,
        "mean_own_shuffle_logprob": observed - gain if gain is not None else None,
        "shuffles": shuffles,
    }


def run(manifest_path: Path, *, seed: int = 20260925, shuffles: int = 32) -> dict:
    if shuffles < 2:
        raise ValueError("shuffles must be at least 2")
    manifest = json.loads(manifest_path.read_text())
    candidates = manifest.get("results")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("week-results manifest has no selected result rows")
    exact_rows = []
    for candidate in candidates:
        exact_rows.append((candidate, verify_candidate(candidate)))

    from nltk.corpus import brown

    raw_sentences = brown.sents()
    clean_sentences = [
        (index, [word.lower() for word in sentence if word.isalpha()])
        for index, sentence in enumerate(raw_sentences)
    ]
    clean_sentences = [(index, words) for index, words in clean_sentences if words]
    model = BrownBigramModel.from_brown()

    candidate_rows = []
    prose_rows = []
    occupied: list[tuple[int, int]] = []
    for candidate, audit in exact_rows:
        item_id = candidate["id"]
        row = score_item(model, item_id, candidate["surface"],
                         source="exact_palindrome_candidate", seed=seed,
                         shuffles=shuffles)
        row.update({
            "surface": candidate["surface"],
            "letters": audit["letters"],
            "exactness": audit,
            "provenance": candidate.get("source", {}),
            "mechanism": candidate.get("mechanism"),
            "lineage": candidate.get("lineage"),
            "reader_status": candidate.get("reader_status"),
        })
        candidate_rows.append(row)

        first, after_last, control_words = select_brown_window(
            clean_sentences, row["word_count"], occupied,
            f"brown-control-{item_id}")
        occupied.append((first, after_last))
        control_surface = " ".join(
            " ".join(raw_sentences[index])
            for index in range(first, after_last)
        )
        control = score_item(
            model, f"brown-control-{item_id}", control_surface,
            source="intact_brown_prose_control", seed=seed, shuffles=shuffles)
        control.update({
            "matched_candidate_id": item_id,
            "candidate_word_count": row["word_count"],
            "source_corpus": "NLTK Brown corpus, contiguous complete-sentence span",
            "source_sentence_indices_half_open": [first, after_last],
            "source_word_count": control_words,
            "word_count_difference": control_words - row["word_count"],
            "surface_sha256": hashlib.sha256(control_surface.encode("utf-8")).hexdigest(),
        })
        prose_rows.append(control)

    candidate_gains = [row["brown_order_gain_vs_own_shuffle"] for row in candidate_rows]
    prose_gains = [row["brown_order_gain_vs_own_shuffle"] for row in prose_rows]
    prose_by_candidate = {row["matched_candidate_id"]: row for row in prose_rows}
    paired_gaps = [
        prose_by_candidate[row["id"]]["brown_order_gain_vs_own_shuffle"]
        - row["brown_order_gain_vs_own_shuffle"]
        for row in candidate_rows
    ]
    candidate_mean = statistics.fmean(candidate_gains)
    prose_mean = statistics.fmean(prose_gains)
    ranked = sorted(candidate_rows,
                    key=lambda row: row["brown_order_gain_vs_own_shuffle"],
                    reverse=True)
    return {
        "experiment": "week-results-brown-order-gain-local-20260925",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "diagnostic_not_human_readability_result",
        "manifest": {
            "path": str(manifest_path.relative_to(ROOT)),
            "snapshot_revision": manifest.get("snapshot"),
            "candidate_count": len(candidate_rows),
        },
        "method": {
            "scorer": "NLTK Brown word-bigram mean log probability",
            "primary_metric": "observed word-order score minus the mean over deterministic shuffles of the same word multiset",
            "shuffle_count_per_item": shuffles,
            "random_seed": seed,
            "control": "one contiguous intact Brown sentence span per candidate, nearest by word count, plus same-word shuffles",
            "corpus_sentences": len(raw_sentences),
            "nltk_version": __import__("nltk").__version__,
        },
        "summary": {
            "candidate_mean_order_gain": candidate_mean,
            "candidate_median_order_gain": statistics.median(candidate_gains),
            "intact_prose_control_mean_order_gain": prose_mean,
            "intact_prose_control_median_order_gain": statistics.median(prose_gains),
            "intact_prose_controls_with_positive_order_gain": sum(x > 0 for x in prose_gains),
            "intact_prose_control_count": len(prose_rows),
            "candidate_minus_prose_mean_order_gain": candidate_mean - prose_mean,
            "matched_controls_with_higher_order_gain": sum(gap > 0 for gap in paired_gaps),
            "mean_matched_prose_minus_candidate_order_gain": statistics.fmean(paired_gaps),
            "candidate_rank_by_local_order_gain": [row["id"] for row in ranked],
        },
        "limits": [
            "A word-bigram model measures local word order, not grammatical completeness, recoverable meaning, discourse coherence, or human readability.",
            "This diagnostic must not certify a candidate or select a paper example by itself.",
            "Brown controls validate only that the metric responds to ordinary corpus word order versus shuffled words; they are not human ratings.",
            "A blinded human comparison with intact prose and shuffled controls remains the readability test.",
        ],
        "candidates": candidate_rows,
        "intact_prose_controls": prose_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "paper/week_results.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs/readability-scorer-week-results-20260925.json")
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--shuffles", type=int, default=32)
    args = parser.parse_args()
    report = run(args.manifest, seed=args.seed, shuffles=args.shuffles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "candidates": report["manifest"]["candidate_count"],
        "candidate_mean_order_gain": report["summary"]["candidate_mean_order_gain"],
        "prose_control_mean_order_gain": report["summary"]["intact_prose_control_mean_order_gain"],
        "positive_prose_controls": report["summary"]["intact_prose_controls_with_positive_order_gain"],
        "top_ranked": report["summary"]["candidate_rank_by_local_order_gain"][:3],
    }, indent=2))


if __name__ == "__main__":
    main()
