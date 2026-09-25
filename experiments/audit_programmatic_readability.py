"""Calibrated, programmatic readability diagnostics for a frozen study packet.

This is deliberately *not* an automatic readability judge.  Its central
quantity is a Brown-corpus word-bigram order gain: the mean log probability of
the observed word order minus the mean over deterministic shuffles of exactly
the same words.  That comparison holds vocabulary and length fixed, so it can
detect local English word order without mistaking common words for readable
prose.  The paired real-prose/shuffle controls test that the quantity has this
minimum discrimination power in the packet itself.

The remaining features are descriptive: word familiarity, word and bigram
repetition, and punctuation-segment density.  None measures a recoverable
subject, discourse coherence, or human readability.  In particular, this
script does not emit a composite score or select system outputs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wordfreq import zipf_frequency


WORD_RE = re.compile(r"[A-Za-z]+")
DEFAULT_STUDY = ROOT / "runs" / "readability-length-study-2026-09-12"


def tokens(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text)]


def safe_mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


@dataclass(frozen=True)
class BrownBigramModel:
    """A fixed, add-alpha Brown word-bigram diagnostic.

    The model is intentionally small and transparent.  Its absolute scores are
    corpus-dependent; only the text-minus-own-shuffle comparison is interpreted
    as a diagnostic, and even that is limited to local word order.
    """

    unigrams: Counter
    bigrams: Counter
    vocabulary_size: int
    alpha: float = 0.1

    @classmethod
    def from_brown(cls) -> "BrownBigramModel":
        from nltk.corpus import brown

        unigrams: Counter = Counter()
        bigrams: Counter = Counter()
        for sentence in brown.sents():
            words = ["<s>"] + [word.lower() for word in sentence if word.isalpha()] + ["</s>"]
            unigrams.update(words[:-1])
            bigrams.update(zip(words, words[1:]))
        return cls(unigrams, bigrams, len(unigrams))

    def score(self, words: Sequence[str]) -> float | None:
        return self.score_sentences([words])

    def score_sentences(self, sentences: Sequence[Sequence[str]]) -> float | None:
        """Mean transition log probability with a boundary pair per sentence."""
        scores = []
        for words in sentences:
            if not words:
                continue
            sequence = (["<s>"] +
                        [word if word in self.unigrams else "<unk>" for word in words] +
                        ["</s>"])
            scores.extend(
                math.log((self.bigrams[(left, right)] + self.alpha)
                         / (self.unigrams[left] + self.alpha * (self.vocabulary_size + 1)))
                for left, right in zip(sequence, sequence[1:])
            )
        return safe_mean(scores)


def seeded_rng(seed: int, item_id: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{item_id}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def order_gain(model: BrownBigramModel, words: Sequence[str], item_id: str,
               seed: int, shuffles: int) -> tuple[float | None, float | None]:
    """Return (observed bigram logprob, observed minus own-shuffle baseline)."""
    observed = model.score(words)
    if observed is None:
        return None, None
    rng = seeded_rng(seed, item_id)
    baseline = []
    for _ in range(shuffles):
        shuffled = list(words)
        rng.shuffle(shuffled)
        score = model.score(shuffled)
        if score is not None:
            baseline.append(score)
    mean_baseline = safe_mean(baseline)
    return observed, (observed - mean_baseline) if mean_baseline is not None else None


def order_gain_by_sentence(model: BrownBigramModel,
                           sentences: Sequence[Sequence[str]],
                           item_id: str, seed: int, shuffles: int
                           ) -> tuple[float | None, float | None]:
    """Score true sentence boundaries; shuffle words while preserving lengths."""
    sentence_words = [list(words) for words in sentences if words]
    observed = model.score_sentences(sentence_words)
    if observed is None:
        return None, None
    lengths = [len(words) for words in sentence_words]
    flat_words = [word for words in sentence_words for word in words]
    rng = seeded_rng(seed, item_id)
    baseline = []
    for _ in range(shuffles):
        shuffled = list(flat_words)
        rng.shuffle(shuffled)
        partitioned, cursor = [], 0
        for length in lengths:
            partitioned.append(shuffled[cursor:cursor + length])
            cursor += length
        score = model.score_sentences(partitioned)
        if score is not None:
            baseline.append(score)
    mean_baseline = safe_mean(baseline)
    return observed, (observed - mean_baseline) if mean_baseline is not None else None


def repeated_bigram_rate(words: Sequence[str]) -> float:
    bigrams = list(zip(words, words[1:]))
    if not bigrams:
        return 0.0
    seen: set[tuple[str, str]] = set()
    repeats = 0
    for bigram in bigrams:
        if bigram in seen:
            repeats += 1
        seen.add(bigram)
    return repeats / len(bigrams)


def punctuation_segments(text: str) -> int:
    return sum(bool(tokens(segment)) for segment in re.split(r"[.!?]+", text))


def item_features(row: dict, text: str, model: BrownBigramModel, seed: int,
                  shuffles: int) -> dict:
    words = tokens(text)
    if not words:
        raise ValueError(f"{row['id']} has no alphabetic words")
    bigram_score, gain = order_gain(model, words, row["id"], seed, shuffles)
    return {
        "id": row["id"],
        "source": row["source"],
        "band": row["band"],
        "letters": row["letters"],
        "words": len(words),
        "brown_bigram_logprob": bigram_score,
        "brown_order_gain_vs_own_shuffle": gain,
        "mean_zipf_frequency": safe_mean(zipf_frequency(word, "en") for word in words),
        "repeated_word_rate": 1 - len(set(words)) / len(words),
        "repeated_bigram_rate": repeated_bigram_rate(words),
        "punctuation_segments_per_100_words": 100 * punctuation_segments(text) / len(words),
        "word_multiset_signature": "\u0000".join(sorted(words)),
    }


FEATURES = (
    "brown_bigram_logprob",
    "brown_order_gain_vs_own_shuffle",
    "mean_zipf_frequency",
    "repeated_word_rate",
    "repeated_bigram_rate",
    "punctuation_segments_per_100_words",
)


def grouped_means(rows: Sequence[dict]) -> dict[str, dict[str, float | None]]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["source"], row["band"])].append(row)
    return {
        f"{source}/{band}": {"n": len(group), **{
            feature: safe_mean(row[feature] for row in group)
            for feature in FEATURES
        }}
        for (source, band), group in sorted(groups.items())
    }


def matched_control_contrasts(rows: Sequence[dict]) -> dict:
    """Pair real-prose and shuffled controls by their identical word multisets."""
    prose = {row["word_multiset_signature"]: row
             for row in rows if row["source"] == "real_prose_control"}
    shuffle = {row["word_multiset_signature"]: row
               for row in rows if row["source"] == "shuffled_word_control"}
    shared = sorted(set(prose) & set(shuffle))
    contrasts = {
        feature: [prose[key][feature] - shuffle[key][feature] for key in shared]
        for feature in FEATURES
    }
    return {
        "matched_pairs": len(shared),
        "unmatched_real_prose": len(set(prose) - set(shuffle)),
        "unmatched_shuffles": len(set(shuffle) - set(prose)),
        "real_prose_minus_shuffled": {
            feature: safe_mean(values) for feature, values in contrasts.items()
        },
        "interpretation": (
            "Only a positive Brown order-gain contrast is a local-order positive-control check. "
            "The other contrasts are descriptive and should not be treated as readability tests."
        ),
    }


def audit(study_dir: Path, seed: int = 20260912, shuffles: int = 32) -> dict:
    blind = json.loads((study_dir / "blind-items.json").read_text())
    key = json.loads((study_dir / "key.json").read_text())
    texts = {row["id"]: row["text"] for row in blind}
    if len(texts) != len(blind) or {row["id"] for row in key} != set(texts):
        raise ValueError("blind packet and source key have inconsistent item IDs")
    model = BrownBigramModel.from_brown()
    rows = [item_features(row, texts[row["id"]], model, seed, shuffles) for row in key]
    return {
        "study_dir": str(study_dir),
        "status": "diagnostic_not_human_readability_result",
        "method": {
            "brown_bigram": "add-alpha word bigram model trained on NLTK Brown",
            "order_gain": "observed mean log probability minus mean over own-word shuffles",
            "shuffle_count": shuffles,
            "random_seed": seed,
        },
        "limits": [
            "Bigram order gain measures only local word order, not grammaticality, identifiable subject, or whole-text coherence.",
            "Word frequency, repetition, and segmentation are descriptive features, not readability scores.",
            "This audit must not select examples or replace blinded human ratings.",
        ],
        "control_calibration": matched_control_contrasts(rows),
        "group_means": grouped_means(rows),
        "items": [{key: value for key, value in row.items() if key != "word_multiset_signature"}
                  for row in rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_dir", nargs="?", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--shuffles", type=int, default=32)
    args = parser.parse_args()
    if args.shuffles < 2:
        parser.error("--shuffles must be at least 2")
    report = audit(args.study_dir, args.seed, args.shuffles)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "items": len(report["items"]),
                      "matched_controls": report["control_calibration"]["matched_pairs"]}, indent=2))


if __name__ == "__main__":
    main()
