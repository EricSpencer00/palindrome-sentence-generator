"""Permit one ordinary word to cross the exact palindrome's central axis.

Word-level centre-out search usually inserts a space at the central character,
excluding otherwise valid whole-sentence palindromes.  Here the search uses a
raw one-letter palindromic centre only as an internal tape marker.  On closure,
the marker is merged into its left neighbour, right neighbour, or both before
the complete rendered sequence is checked for lexicality, novelty, exactness,
non-repetition, and a whole-sentence grammar plan.  The marker is never shown
as a one-letter output unit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.bidirectional_attested_span_mining import common_lexicon
from experiments.constrained_bilateral_decoder import (
    SUBJECT_OR_QUESTION_OPENERS, TRAILING_FUNCTION_WORDS, observed_joins,
)
from llm_palindrome.bigram import BigramModel
from llm_palindrome.admission import mechanical_admission_checks
from llm_palindrome.centerout import centerout_search
from llm_palindrome.paragraphs import is_novel_palindrome
from llm_palindrome.scoring import CoherentScorer
from llm_palindrome.search import WordTries
from llm_palindrome.sentence_plan import SentencePlan
from llm_palindrome.syntax import brown_tables
from llm_palindrome.validator import is_palindrome, normalize
from server.v3 import real_words
from wordfreq import zipf_frequency


CENTRES = ("a", "e", "i", "n", "r", "s", "t")


def render_crossing(left: tuple[str, ...], right: tuple[str, ...], center: str) -> list[list[str]]:
    """Return the three possible no-space renderings at a raw central letter."""
    if not left or not right:
        return []
    return [
        list(left[:-1]) + [left[-1] + center] + list(right),
        list(left) + [center + right[0]] + list(right[1:]),
        list(left[:-1]) + [left[-1] + center + right[0]] + list(right[1:]),
    ]


def allows_state(left: tuple[str, ...], right: tuple[str, ...], bigrams: BigramModel) -> bool:
    words = left + right
    return (len(left) <= 5 and len(right) <= 5
            and len(words) == len(set(words))
            and observed_joins(" ".join(left), bigrams)
            and observed_joins(" ".join(right), bigrams))


def sentence_checks(words: list[str], plan: SentencePlan) -> dict[str, bool]:
    text = " ".join(words)
    catalogue = set(json.loads((ROOT / "data" / "known_palindromes.json").read_text()))
    shared = mechanical_admission_checks(
        text, local_catalogue=catalogue, min_letters=30, max_letters=60
    )
    return shared | {
        "exact_palindrome": shared["exact_letter_palindrome"],
        "lexicon_words": real_words(words),
        "no_repeated_words": shared["distinct_words"],
        "no_self_palindromic_word_units": shared["no_self_palindromic_word"],
        "novel_catalogue": shared["local_catalogue_absent"],
        "subject_or_question_opening": bool(words and words[0] in SUBJECT_OR_QUESTION_OPENERS),
        "non_function_ending": bool(words and words[-1] not in TRAILING_FUNCTION_WORDS),
        "whole_sentence_plan": plan.complete(words),
    }


def run(*, seeds: int, vocabulary_size: int, min_zipf: float, beam: int) -> dict:
    table, shapes, _ = brown_tables()
    plan = SentencePlan(table, shapes, min_words=4, max_words=9)
    vocabulary = sorted((word for word in common_lexicon(min_zipf).intersection(plan.table)
                         if real_words([word])),
                        key=lambda word: (-zipf_frequency(word, "en"), word))[:vocabulary_size]
    tries = WordTries(vocabulary)
    bigrams = BigramModel.from_file(str(ROOT / "data" / "count_2w.txt"), vocab=vocabulary)
    scorer = CoherentScorer(bigrams, freq_weight=0.2, length_weight=0.04,
                            short_penalty=1.5)
    seen: set[str] = set()
    records = []
    for centre_index, center in enumerate(CENTRES):
        for seed in range(seeds):
            words = centerout_search(
                tries, scorer, min_letters=30, beam_width=beam, center=center,
                max_steps=64, candidate_limit=256,
                seed=2026094000 + 1000 * centre_index + seed, diversity=0.8,
                max_overhang=20,
                allow_state=lambda left, right: allows_state(left, right, bigrams),
                allow_closed=lambda left, right: 2 <= len(left) <= 5 and 2 <= len(right) <= 5,
            )
            if center not in words:
                continue
            split = words.index(center)
            left, right = tuple(words[:split]), tuple(words[split + 1:])
            for rendered in render_crossing(left, right, center):
                text = " ".join(rendered)
                if text in seen:
                    continue
                seen.add(text)
                gate = sentence_checks(rendered, plan)
                records.append({"centre": center, "seed": seed, "text": text,
                                "words": rendered, "letters": len(normalize(text)),
                                "checks": gate,
                                "rejection_codes": [key for key, value in gate.items() if not value],
                                "independent_exact_audit": is_palindrome(text)})
    return {
        "status": "complete_centre_crossing_whole_sentence_decoder_run",
        "config": {"centres": CENTRES, "seeds_per_centre": seeds,
                   "vocabulary_size_requested": vocabulary_size, "min_zipf": min_zipf,
                   "beam": beam},
        "vocabulary_size": len(vocabulary),
        "vocabulary_sha256": hashlib.sha256("\n".join(vocabulary).encode()).hexdigest(),
        "records": records,
        "mechanically_admitted": [row for row in records if not row["rejection_codes"]],
        "reader_gate": (
            "Mechanical admission is not reader evidence. Any rendered candidate must be tested "
            "with intact prose controls, matched shuffles, randomized blinded order, and independent "
            "human ratings before it is called readable."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--vocabulary-size", type=int, default=16000)
    parser.add_argument("--min-zipf", type=float, default=3.0)
    parser.add_argument("--beam", type=int, default=160)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = run(seeds=args.seeds, vocabulary_size=args.vocabulary_size,
                 min_zipf=args.min_zipf, beam=args.beam)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "records": len(result["records"]),
                      "admitted": len(result["mechanically_admitted"])}, indent=2))


if __name__ == "__main__":
    main()
