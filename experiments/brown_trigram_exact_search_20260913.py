"""Exact lattice search scored by Brown trigram context.

This is a representation-preserving language upgrade: the palindrome lattice
still owns every accepted letter, while a corpus trigram model scores the
actual left-to-right joins created by each outside-in transition.  It does
not claim that a corpus score certifies readability.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import (ORDINARY_TWO_LETTER_WORDS,
    mechanical_admission_checks, normalize_letters)
from llm_palindrome.generate import build_vocab
from llm_palindrome.search import WordTries, beam_search
from llm_palindrome.textify import textify


def brown_trigrams() -> tuple[Counter, Counter, Counter, int]:
    from nltk.corpus import brown
    trigram = Counter()
    context = Counter()
    unigram = Counter()
    sentences = []
    for sentence in brown.sents():
        words = [word.casefold() for word in sentence if word.isascii() and word.isalpha()]
        if not words:
            continue
        sentences.append(words)
        unigram.update(words)
        for a, b, c in zip(words, words[1:], words[2:]):
            trigram[(a, b, c)] += 1
            context[(a, b)] += 1
    return trigram, context, unigram, len(sentences)


class BrownTrigramScorer:
    def __init__(self, trigram: Counter, context: Counter, unigram: Counter,
                 *, short_penalty: float = 2.0):
        self.trigram = trigram
        self.context = context
        self.unigram = unigram
        self.vocab_size = max(1, len(unigram))
        self.short_penalty = short_penalty

    def trigram_score(self, a: str, b: str, c: str) -> float:
        n = self.trigram.get((a, b, c), 0)
        den = self.context.get((a, b), 0)
        return math.log((n + 0.05) / (den + 0.05 * self.vocab_size))

    def word_delta(self, left, right, placement, word, growth):
        inner = word.split()
        score = sum(math.log(self.unigram.get(w, 1) + 0.05) for w in inner) * 0.08
        if placement == "L":
            seq = tuple(left)
            if len(seq) >= 2:
                score += self.trigram_score(seq[-2], seq[-1], word)
        else:
            seq = tuple(right)
            if len(seq) >= 2:
                score += self.trigram_score(word, seq[0], seq[1])
        if len(inner) == 1 and len(inner[0]) <= 2:
            score -= self.short_penalty
        return score + 0.04 * len(word)


def audit(words: list[str], seed: int) -> dict:
    rendered = textify(words)
    tape = normalize_letters(rendered)
    independent_tape = "".join(char.lower() for char in rendered
                                 if "A" <= char <= "Z" or "a" <= char <= "z")
    checks = mechanical_admission_checks(rendered, min_letters=100, max_letters=180)
    independent = (bool(independent_tape) and independent_tape == independent_tape[::-1]
                   and independent_tape == tape)
    checks["independent_exact_audit"] = independent
    return {"seed": seed, "rendered": rendered, "words": words,
            "letters": len(tape), "mechanical_checks": checks,
            "mechanically_eligible": all(checks.values()),
            "independent_normalized_letters": independent_tape,
            "render_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
            "reader_status": "not_run"}


def run(*, seeds: int, vocabulary_size: int, beam: int, candidate_limit: int) -> dict:
    trigram, context, unigram, sentence_count = brown_trigrams()
    vocabulary = [word for word in build_vocab(vocabulary_size)
                  if word != word[::-1]
                  and (len(word) > 2 or word in ORDINARY_TWO_LETTER_WORDS)]
    tries = WordTries(vocabulary)
    scorer = BrownTrigramScorer(trigram, context, unigram)
    records = []
    for seed in range(seeds):
        words = beam_search(tries, scorer, min_letters=100, max_steps=220,
                            beam_width=beam, candidate_limit=candidate_limit,
                            seed=seed, diversity=1.8, max_word_uses=2)
        if words:
            records.append(audit(words, seed))
    return {"status": "complete_brown_trigram_exact_search",
            "config": {"seeds": seeds, "vocabulary_size": len(vocabulary),
                       "beam": beam, "candidate_limit": candidate_limit,
                       "minimum_letters": 100, "machine_readability_certification": False},
            "brown_sentences": sentence_count, "brown_trigrams": len(trigram),
            "records": records,
            "mechanically_eligible": [r for r in records if r["mechanically_eligible"]],
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "corpus": "NLTK Brown sentences, local copy", "vocabulary": "build_vocab"},
            "reader_gate": "No readability claim; eligible text requires randomized blinded readers with intact prose and shuffled controls."}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", type=int, default=24)
    p.add_argument("--vocabulary-size", type=int, default=12000)
    p.add_argument("--beam", type=int, default=120)
    p.add_argument("--candidate-limit", type=int, default=300)
    args = p.parse_args()
    if args.out.exists():
        p.error(f"refusing to overwrite {args.out}")
    result = run(seeds=args.seeds, vocabulary_size=args.vocabulary_size,
                 beam=args.beam, candidate_limit=args.candidate_limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "records": len(result["records"]),
                      "mechanically_eligible": len(result["mechanically_eligible"])}, sort_keys=True))


if __name__ == "__main__":
    main()
