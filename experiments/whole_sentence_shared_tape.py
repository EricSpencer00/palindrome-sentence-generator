"""Construct one grammatical whole sentence over an exact shared palindrome tape.

Earlier bilateral experiments mistakenly required both halves of a palindrome to
be separate sentences.  This decoder instead treats the complete rendered
palindrome as the linguistic object: character equality is enforced during
center-out construction, while an observed subject-and-verb sentence plan is
required only of the complete word sequence.  This broadens the construction
space without weakening exactness, novelty, non-repetition, or later blinded
reader evaluation.
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


def allow_state(left: tuple[str, ...], right: tuple[str, ...], bigrams: BigramModel) -> bool:
    """Keep only lexical, non-repeating partial tape readings."""
    words = list(left + right)
    return (len(words) <= 9 and len(words) == len(set(words))
            and observed_joins(" ".join(left), bigrams)
            and observed_joins(" ".join(right), bigrams))


def allow_closed(left: tuple[str, ...], right: tuple[str, ...], plan: SentencePlan) -> bool:
    """Gate the complete palindrome, not its individually mirrored halves."""
    words = left + right
    return (4 <= len(words) <= 9
            and words[0] in SUBJECT_OR_QUESTION_OPENERS
            and words[-1] not in TRAILING_FUNCTION_WORDS
            and plan.complete(words))


def checks(words: list[str], catalogue: set[str] | None = None) -> dict[str, bool]:
    text = " ".join(words)
    left, right = " ".join(words[:len(words) // 2]), " ".join(words[len(words) // 2:])
    catalogue = catalogue if catalogue is not None else set(json.loads(
        (ROOT / "data" / "known_palindromes.json").read_text()
    ))
    shared = mechanical_admission_checks(
        text, local_catalogue=catalogue, min_letters=30, max_letters=60
    )
    return shared | {
        "exact_palindrome": shared["exact_letter_palindrome"],
        "lexicon_words": real_words(words),
        "no_repeated_words": shared["distinct_words"],
        "no_self_palindromic_word_units": shared["no_self_palindromic_word"],
        "novel_catalogue": shared["local_catalogue_absent"],
        "nonrepeated_half_tapes": normalize(left) != normalize(right),
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
    for seed in range(seeds):
        sequence = centerout_search(
            tries, scorer, min_letters=30, beam_width=beam, max_steps=64,
            candidate_limit=256, seed=seed, diversity=0.8, max_overhang=20,
            allow_state=lambda left, right: allow_state(left, right, bigrams),
            allow_closed=lambda left, right: allow_closed(left, right, plan),
        )
        text = " ".join(sequence)
        if not sequence or text in seen:
            continue
        seen.add(text)
        gate = checks(sequence)
        records.append({"seed": seed, "text": text, "words": sequence,
                        "letters": len(normalize(text)), "checks": gate,
                        "rejection_codes": [key for key, value in gate.items() if not value],
                        "independent_exact_audit": is_palindrome(text)})
    return {
        "status": "complete_whole_sentence_shared_tape_run",
        "config": {"seeds": list(range(seeds)), "vocabulary_size_requested": vocabulary_size,
                   "min_zipf": min_zipf, "beam": beam},
        "vocabulary_size": len(vocabulary),
        "vocabulary_sha256": hashlib.sha256("\n".join(vocabulary).encode()).hexdigest(),
        "mechanically_admitted": [row for row in records if not row["rejection_codes"]],
        "records": records,
        "reader_gate": (
            "A mechanically admitted whole sentence is a reader-study candidate, not a readability "
            "claim. It requires intact prose controls, matched shuffles, randomized blind order, and "
            "independent human ratings of grammar, intent, and coherence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seeds", type=int, default=128)
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
