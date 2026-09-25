"""Calibrate palindrome word-order scores against held-out Brown prose by length.

Brown documents are split before model construction: the diagnostic language
model sees only training documents, while all intact prose controls come from
disjoint held-out documents. Controls are complete contiguous sentence spans,
matched exactly where possible to scorer-token count, and compared with
deterministic shuffles of their own words. Nothing here certifies readability.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.audit_programmatic_readability import BrownBigramModel, order_gain, tokens  # noqa: E402
from experiments.score_week_results_readability import (  # noqa: E402
    select_brown_window,
    verify_candidate,
)


HOLDOUT_SEED = "brown-readability-file-holdout-20260925"
LENGTH_TARGETS = (16, 32, 64, 128, 192, 256, 512, 1024, 2048)
CONTROLS_PER_TARGET = 12
SHUFFLES = 32
SEED = 20260925


def manifest_label(manifest_path: Path) -> str:
    """Keep in-root paths reproducible and external paths free of host details."""
    resolved = manifest_path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.name


def split_fileids(fileids: list[str]) -> tuple[list[str], list[str]]:
    """Deterministically hold out roughly one fifth of Brown documents."""
    train, heldout = [], []
    for fileid in sorted(fileids):
        bucket = int(hashlib.sha256(
            f"{HOLDOUT_SEED}:{fileid}".encode()).hexdigest()[:8], 16) % 5
        (heldout if bucket == 0 else train).append(fileid)
    if not train or not heldout:
        raise ValueError("Brown document split produced an empty partition")
    return train, heldout


def model_from_training_documents(brown, fileids: list[str]) -> BrownBigramModel:
    unigrams: Counter = Counter()
    bigrams: Counter = Counter()
    for fileid in fileids:
        for sentence in brown.sents(fileid):
            words = tokens(" ".join(sentence))
            if not words:
                continue
            sequence = ["<s>"] + words + ["</s>"]
            unigrams.update(sequence[:-1])
            bigrams.update(zip(sequence, sequence[1:]))
    return BrownBigramModel(unigrams, bigrams, len(unigrams))


def holdout_documents(brown, fileids: list[str]) -> dict[str, tuple[list[list[str]], list[tuple[int, list[str]]]]]:
    """Return raw sentences and scorer-token sentences, keyed by Brown file."""
    out = {}
    for fileid in fileids:
        raw = brown.sents(fileid)
        clean = [(index, tokens(" ".join(sentence)))
                 for index, sentence in enumerate(raw)]
        clean = [(index, words) for index, words in clean if words]
        out[fileid] = (raw, clean)
    return out


def select_heldout_span(documents: dict, target: int,
                        occupied: dict[str, list[tuple[int, int]]],
                        span_id: str) -> tuple[str, int, int, int, str]:
    choices = []
    for fileid, (raw, clean) in sorted(documents.items()):
        try:
            first, after_last, count = select_brown_window(
                clean, target, occupied.get(fileid, []), f"{span_id}:{fileid}")
        except ValueError:
            continue
        tie = hashlib.sha256(
            f"{span_id}:{fileid}:{first}:{after_last}".encode()).hexdigest()
        choices.append((abs(count - target), tie, fileid,
                        first, after_last, count, raw))
    if not choices:
        raise ValueError(f"no unoccupied held-out Brown span for {span_id}")
    _, _, fileid, first, after_last, count, raw = min(choices)
    surface = " ".join(" ".join(sentence) for sentence in raw[first:after_last])
    return fileid, first, after_last, count, surface


def score(model: BrownBigramModel, item_id: str, text: str) -> dict:
    words = tokens(text)
    observed, gain = order_gain(model, words, item_id, SEED, SHUFFLES)
    return {
        "id": item_id,
        "scorer_tokens": len(words),
        "brown_bigram_logprob": observed,
        "brown_order_gain_vs_own_shuffle": gain,
        "mean_own_shuffle_logprob": observed - gain if gain is not None else None,
        "shuffle_count": SHUFFLES,
    }


def aggregate(rows: list[dict], target: int) -> dict:
    gains = [row["brown_order_gain_vs_own_shuffle"] for row in rows]
    lengths = [row["source_word_count"] for row in rows]
    return {
        "target_scorer_tokens": target,
        "control_count": len(rows),
        "actual_tokens_min": min(lengths),
        "actual_tokens_median": statistics.median(lengths),
        "actual_tokens_max": max(lengths),
        "mean_order_gain_nats_per_transition": statistics.fmean(gains),
        "median_order_gain_nats_per_transition": statistics.median(gains),
        "positive_order_gain_controls": sum(value > 0 for value in gains),
    }


def run(manifest_path: Path = ROOT / "paper/week_results.json") -> dict:
    from nltk.corpus import brown
    import nltk

    manifest = json.loads(manifest_path.read_text())
    candidates = manifest["results"]
    audits = {row["id"]: verify_candidate(row) for row in candidates}
    train_ids, heldout_ids = split_fileids(list(brown.fileids()))
    model = model_from_training_documents(brown, train_ids)
    holdout = holdout_documents(brown, heldout_ids)
    occupied: dict[str, list[tuple[int, int]]] = {}

    candidate_rows, matched_controls = [], []
    for candidate in candidates:
        candidate_id = candidate["id"]
        audit = audits[candidate_id]
        candidate_score = score(model, candidate_id, candidate["surface"])
        candidate_score.update({
            "surface": candidate["surface"],
            "letters": audit["letters"],
            "exactness": audit,
            "provenance": candidate.get("source", {}),
            "mechanism": candidate.get("mechanism"),
            "lineage": candidate.get("lineage"),
            "reader_status": candidate.get("reader_status"),
        })
        candidate_rows.append(candidate_score)

        span_id = f"matched-prose-{candidate_id}"
        fileid, first, after_last, count, surface = select_heldout_span(
            holdout, candidate_score["scorer_tokens"], occupied, span_id)
        occupied.setdefault(fileid, []).append((first, after_last))
        control = score(model, span_id, surface)
        control.update({
            "source": "heldout_intact_prose_control",
            "matched_candidate_id": candidate_id,
            "source_word_count": count,
            "word_count_difference": count - candidate_score["scorer_tokens"],
            "brown_fileid": fileid,
            "sentence_indices_half_open": [first, after_last],
            "surface_sha256": hashlib.sha256(surface.encode("utf-8")).hexdigest(),
        })
        matched_controls.append(control)

    length_controls = []
    for target in LENGTH_TARGETS:
        for replicate in range(CONTROLS_PER_TARGET):
            span_id = f"length-{target}-control-{replicate:02d}"
            fileid, first, after_last, count, surface = select_heldout_span(
                holdout, target, occupied, span_id)
            occupied.setdefault(fileid, []).append((first, after_last))
            control = score(model, span_id, surface)
            control.update({
                "source": "heldout_intact_prose_length_control",
                "target_scorer_tokens": target,
                "source_word_count": count,
                "word_count_difference": count - target,
                "brown_fileid": fileid,
                "sentence_indices_half_open": [first, after_last],
                "surface_sha256": hashlib.sha256(surface.encode("utf-8")).hexdigest(),
            })
            length_controls.append(control)

    control_by_candidate = {row["matched_candidate_id"]: row
                           for row in matched_controls}
    paired_gaps = [
        control_by_candidate[row["id"]]["brown_order_gain_vs_own_shuffle"]
        - row["brown_order_gain_vs_own_shuffle"]
        for row in candidate_rows
    ]
    curve = [aggregate([row for row in length_controls
                        if row["target_scorer_tokens"] == target], target)
             for target in LENGTH_TARGETS]
    gains = [row["brown_order_gain_vs_own_shuffle"] for row in candidate_rows]
    prose_gains = [row["brown_order_gain_vs_own_shuffle"]
                   for row in matched_controls]
    ranked = sorted(candidate_rows,
                    key=lambda row: row["brown_order_gain_vs_own_shuffle"],
                    reverse=True)
    return {
        "experiment": "heldout-brown-length-stratified-readability-20260925",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "programmatic_diagnostic_not_human_readability_result",
        "manifest": {
            "path": manifest_label(manifest_path),
            "snapshot_revision": manifest.get("snapshot"),
            "candidate_count": len(candidate_rows),
        },
        "method": {
            "scorer": "Brown word-bigram mean log probability; order gain is the observed score minus the mean over same-word shuffles",
            "metric_units": "nats per word transition",
            "training_split": "document-level deterministic split; no held-out control document is used to train the scorer",
            "training_file_count": len(train_ids),
            "heldout_file_count": len(heldout_ids),
            "training_fileid_manifest_sha256": hashlib.sha256(
                "\n".join(train_ids).encode()).hexdigest(),
            "heldout_fileid_manifest_sha256": hashlib.sha256(
                "\n".join(heldout_ids).encode()).hexdigest(),
            "length_targets_scorer_tokens": list(LENGTH_TARGETS),
            "controls_per_length": CONTROLS_PER_TARGET,
            "matched_prose_controls": len(matched_controls),
            "shuffles_per_item": SHUFFLES,
            "random_seed": SEED,
            "brown_sentences": len(brown.sents()),
            "nltk_version": nltk.__version__,
            "controls_are_complete_contiguous_sentences": True,
            "controls_are_nonoverlapping_within_heldout_documents": True,
        },
        "summary": {
            "candidate_mean_order_gain": statistics.fmean(gains),
            "candidate_median_order_gain": statistics.median(gains),
            "matched_heldout_prose_mean_order_gain": statistics.fmean(prose_gains),
            "matched_heldout_prose_median_order_gain": statistics.median(prose_gains),
            "matched_controls_with_higher_order_gain": sum(gap > 0 for gap in paired_gaps),
            "mean_matched_prose_minus_candidate_order_gain": statistics.fmean(paired_gaps),
            "candidate_rank_by_order_gain": [row["id"] for row in ranked],
        },
        "length_curve": curve,
        "candidates": candidate_rows,
        "matched_heldout_prose_controls": matched_controls,
        "length_stratified_heldout_controls": length_controls,
        "limits": [
            "The score diagnoses local word order only; it does not establish grammar, meaning, discourse coherence, or human readability.",
            "Brown controls and model share a corpus domain, but held-out documents prevent direct document leakage; corpus/style bias remains.",
            "Long prose controls calibrate score behavior beyond the current candidate lengths; they do not make the palindrome candidates longer or more readable.",
            "Use blinded human comparisons with intact prose and shuffled controls for any readability claim.",
        ],
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "paper/week_results.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs/readability-length-stratified-20260925.json")
    args = parser.parse_args()
    report = run(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "candidate_mean_order_gain": report["summary"]["candidate_mean_order_gain"],
        "matched_prose_mean_order_gain": report["summary"]["matched_heldout_prose_mean_order_gain"],
        "matched_controls_beating_candidates": report["summary"]["matched_controls_with_higher_order_gain"],
        "length_curve": report["length_curve"],
    }, indent=2))


if __name__ == "__main__":
    main()
