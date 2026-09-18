"""Outside-in exact palindrome decoding constrained by one whole sentence plan.

The left reading grows from the displayed sentence opening and the right reading
grows from its displayed ending.  At every exact-overhang state, the current
left words must still be a prefix of an observed subject-and-verb sentence
plan, and the current right words a suffix of one.  Thus grammar constrains the
same whole text that is rendered, while character equality remains a hard
property of the search transition.
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
from experiments.constrained_bilateral_decoder import SUBJECT_OR_QUESTION_OPENERS
from experiments.whole_sentence_shared_tape import checks
from llm_palindrome.bigram import BigramModel
from llm_palindrome.scoring import CoherentScorer
from llm_palindrome.search import WordTries, beam_search
from llm_palindrome.sentence_plan import SentencePlan
from llm_palindrome.syntax import brown_tables
from server.v3 import real_words
from wordfreq import zipf_frequency


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

    def prune(states):
        return [state for state in states
                if plan.prefix_possible(state.left) and plan.suffix_possible(state.right)]

    seen, records = set(), []
    for seed in range(seeds):
        words = beam_search(tries, scorer, min_letters=30, beam_width=beam,
                            max_steps=96, candidate_limit=384, seed=seed,
                            diversity=0.8, prune=prune, prune_every=1,
                            opening_words=set(SUBJECT_OR_QUESTION_OPENERS),
                            max_word_uses=1)
        text = " ".join(words)
        if not words or text in seen or not plan.complete(words):
            continue
        seen.add(text)
        gate = checks(words)
        records.append({"seed": seed, "text": text, "words": words,
                        "letters": len("".join(words)), "checks": gate,
                        "rejection_codes": [key for key, value in gate.items() if not value]})
    return {
        "status": "complete_outside_in_whole_sentence_decoder_run",
        "config": {"seeds": list(range(seeds)), "vocabulary_size_requested": vocabulary_size,
                   "min_zipf": min_zipf, "beam": beam},
        "vocabulary_size": len(vocabulary),
        "vocabulary_sha256": hashlib.sha256("\n".join(vocabulary).encode()).hexdigest(),
        "records": records,
        "mechanically_admitted": [row for row in records if not row["rejection_codes"]],
        "reader_gate": "Mechanical admission requires blinded human evaluation with intact controls before any readability claim.",
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
