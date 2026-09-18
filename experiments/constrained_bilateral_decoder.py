"""Decode both word-boundary readings of one exact palindrome simultaneously.

This replaces one-sided proposal-and-respacing with a shared-tape search.  The
decoder grows outwards from an empty centre.  A word added to either side must
consume the exact character debt induced by the other side, so every closed
state has two independently segmented lexicalizations of the same tape.
Separate forward and backward bigram contexts order proposals on the two sides;
they never certify readability or admit an invalid candidate.

Every emitted lead is checked independently for exactness, lexical form,
length, non-repetition, and novelty.  Human readers, using a frozen blinded
study with intact controls, remain the only acceptance gate for English
quality.
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
from experiments.joint_dual_lexicalization import existing_v3_pairs, screen_candidate
from llm_palindrome.bigram import BigramModel
from llm_palindrome.centerout import centerout_search
from llm_palindrome.scoring import CoherentScorer
from llm_palindrome.search import WordTries
from llm_palindrome.sentence_plan import SentencePlan
from llm_palindrome.syntax import brown_tables
from server.v3 import real_words
from llm_palindrome.validator import is_palindrome, normalize
from wordfreq import zipf_frequency


SUBJECT_OR_QUESTION_OPENERS = frozenset({
    "he", "she", "they", "we", "it", "this", "that", "these", "those",
    "people", "men", "women", "children", "time", "there", "who", "what",
    "was", "were", "is", "are", "do", "does", "did", "can", "will", "may",
    "has", "have", "had",
})
TRAILING_FUNCTION_WORDS = frozenset({
    "a", "an", "the", "and", "as", "at", "by", "for", "from", "in", "into",
    "is", "it", "no", "of", "on", "or", "that", "the", "to", "with",
})


def split_bilateral(words: list[str]) -> tuple[str, str] | None:
    """Recover the two nonempty lexicalizations of a no-centre closure."""
    for cut in range(3, len(words) - 2):
        left, right = " ".join(words[:cut]), " ".join(words[cut:])
        if normalize(left) and normalize(left) == normalize(right)[::-1]:
            return left, right
    return None


def observed_joins(text: str, bigrams: BigramModel) -> bool:
    """A strict proposal filter: every adjacent word pair must be attested."""
    words = text.split()
    return all(bigrams.observed(left, right) for left, right in zip(words, words[1:]))


def _allow_state(left: tuple[str, ...], right: tuple[str, ...],
                 plan: SentencePlan | None = None,
                 bigrams: BigramModel | None = None) -> bool:
    words = [word for unit in left + right for word in unit.split()]
    base = len(left) <= 8 and len(right) <= 8 and len(words) == len(set(words))
    syntactic = plan is None or plan.state_possible(left, right)
    local_english = bigrams is None or (
        observed_joins(" ".join(left), bigrams)
        and observed_joins(" ".join(right), bigrams)
    )
    return base and syntactic and local_english


def _allow_closed(left: tuple[str, ...], right: tuple[str, ...],
                  plan: SentencePlan | None = None) -> bool:
    base = (3 <= len(left) <= 8 and 3 <= len(right) <= 8
            and not set(left).intersection(right))
    if not base:
        return False
    boundary = (left[0] in SUBJECT_OR_QUESTION_OPENERS
                and right[0] in SUBJECT_OR_QUESTION_OPENERS
                and left[-1] not in TRAILING_FUNCTION_WORDS
                and right[-1] not in TRAILING_FUNCTION_WORDS)
    return boundary and (plan is None or (plan.complete(left) and plan.complete(right)))


def run(*, seeds: int, vocabulary_size: int, min_zipf: float,
        min_letters: int, max_letters: int, beam: int) -> dict:
    """Run fixed-seed bilateral decoding and retain every mechanically closed lead."""
    # This order becomes the bounded trie menu.  Frequency must come first:
    # ordering by spelling or length starves function words and prevents either
    # side from forming ordinary clause structure before the shared tape closes.
    table, shapes, _ = brown_tables()
    plan = SentencePlan(table, shapes, min_words=3, max_words=8)
    vocabulary = sorted((word for word in common_lexicon(min_zipf).intersection(plan.table)
                         if real_words([word])),
                        key=lambda word: (-zipf_frequency(word, "en"), word))[:vocabulary_size]
    tries = WordTries(vocabulary)
    bigrams = BigramModel.from_file(str(ROOT / "data" / "count_2w.txt"), vocab=vocabulary)
    scorer = CoherentScorer(bigrams, freq_weight=0.15, length_weight=0.05,
                            short_penalty=1.5)
    known_pairs = existing_v3_pairs()
    closed: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for seed in range(seeds):
        raw_closures: list[list[str]] = []
        words = centerout_search(
            tries, scorer, min_letters=min_letters, beam_width=beam,
            max_steps=64, candidate_limit=256, seed=seed, diversity=0.75,
            max_overhang=20,
            allow_state=lambda left, right: _allow_state(left, right, plan, bigrams),
            allow_closed=lambda left, right: _allow_closed(left, right, plan),
            on_closed=lambda sequence: raw_closures.append(sequence),
        )
        for sequence in ([words] if words else []) + raw_closures:
            split = split_bilateral(sequence)
            if split is None:
                continue
            left, right = split
            if not min_letters <= len(normalize(left) + normalize(right)) <= max_letters:
                continue
            if not (observed_joins(left, bigrams) and observed_joins(right, bigrams)):
                continue
            key = (left, right)
            if key in seen:
                continue
            seen.add(key)
            row = screen_candidate(left, right, "shared_tape_bilateral_decoder",
                                   existing_pairs=known_pairs)
            row["seed"] = seed
            row["word_sequence"] = sequence
            row["independent_exact_audit"] = (
                is_palindrome(row["text"])
                and normalize(left) == normalize(right)[::-1]
            )
            closed.append(row)
    admitted = [row for row in closed if not row["rejection_codes"]]
    return {
        "status": "complete_shared_tape_bilateral_decoder_run",
        "config": {
            "seeds": list(range(seeds)), "vocabulary_size_requested": vocabulary_size,
            "min_zipf": min_zipf, "min_letters": min_letters,
            "max_letters": max_letters, "beam": beam,
        },
        "vocabulary_size": len(vocabulary),
        "vocabulary_sha256": hashlib.sha256("\n".join(vocabulary).encode()).hexdigest(),
        "decoder": {
            "tape": "one character tape shared by both lexicalizations",
            "hard_constraints": ["exact reversal", "3--8 words per side", "30--60 letters",
                                 "no repeated word", "no identical side", "lexicon", "novelty",
                                 "attested local joins on both readings"],
            "structural_prefilter": "both sides must match an attested Brown subject-and-verb plan and every partial local join must be attested; not a reader-quality score",
            "proposal_order": "two directional Brown/Norvig bigram contexts; not an acceptance score",
        },
        "closed_leads": closed,
        "mechanically_admitted_leads": admitted,
        "reader_gate": (
            "Mechanically admitted leads are not called readable. Each must be rendered as intact "
            "prose and judged by independent blinded readers alongside complete prose and matched "
            "word-shuffle controls in randomized order."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seeds", type=int, default=48)
    parser.add_argument("--vocabulary-size", type=int, default=8000)
    parser.add_argument("--min-zipf", type=float, default=3.6)
    parser.add_argument("--min-letters", type=int, default=30)
    parser.add_argument("--max-letters", type=int, default=60)
    parser.add_argument("--beam", type=int, default=72)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = run(seeds=args.seeds, vocabulary_size=args.vocabulary_size,
                 min_zipf=args.min_zipf, min_letters=args.min_letters,
                 max_letters=args.max_letters, beam=args.beam)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "closed": len(result["closed_leads"]),
                      "admitted": len(result["mechanically_admitted_leads"])}, indent=2))


if __name__ == "__main__":
    main()
