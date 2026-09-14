"""Exact palindrome search with soft Brown POS-transition scoring.

Unlike the fixed-skeleton pilot, this scorer permits any legal word sequence
but rewards locally plausible part-of-speech transitions in both reading
directions. It ranks exact lattice states only; it never certifies readability.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wordfreq import zipf_frequency

from llm_palindrome.admission import (
    ORDINARY_TWO_LETTER_WORDS, REPEATABLE_FUNCTION_WORDS,
    has_only_ordinary_short_words, mechanical_admission_checks, normalize_letters,
)
from llm_palindrome.generate import build_vocab
from llm_palindrome.search import WordTries, beam_search
from llm_palindrome.scoring import adjacent
from llm_palindrome.textify import textify

MIN_LETTERS, MAX_LETTERS = 100, 180


def brown_pos_model(vocabulary: set[str]) -> tuple[dict[str, frozenset[str]], Counter, Counter, Counter, int]:
    from nltk.corpus import brown

    word_tags: dict[str, set[str]] = defaultdict(set)
    pairs: Counter = Counter()
    contexts: Counter = Counter()
    starts: Counter = Counter()
    sentence_count = 0
    for sentence in brown.tagged_sents(tagset="universal"):
        rows = [(word.casefold(), tag) for word, tag in sentence
                if word.isascii() and word.isalpha()]
        if not rows:
            continue
        sentence_count += 1
        for word, tag in rows:
            if word in vocabulary:
                word_tags[word].add(tag)
        tags = [tag for _, tag in rows]
        starts[tags[0]] += 1
        for left, right in zip(tags, tags[1:]):
            pairs[(left, right)] += 1
            contexts[left] += 1
    return {word: frozenset(tags) for word, tags in word_tags.items()}, pairs, contexts, starts, sentence_count


class BrownPOSScorer:
    wants_overhang = False

    def __init__(self, word_tags: dict[str, frozenset[str]], pairs: Counter,
                 contexts: Counter, starts: Counter, sentence_count: int):
        self.word_tags = word_tags
        self.pairs = pairs
        self.contexts = contexts
        self.starts = starts
        self.tagset = sorted(set(starts) | {tag for pair in pairs for tag in pair})
        self.alpha = 0.05
        self.tag_vocab = max(1, len(self.tagset))
        self.sentence_count = sentence_count

    def tags(self, word: str) -> frozenset[str]:
        return self.word_tags.get(word, frozenset({"X"}))

    def transition(self, left: str | None, right: str) -> float:
        if left is None:
            return math.log((self.starts.get(right, 0) + self.alpha)
                            / (sum(self.starts.values()) + self.alpha * self.tag_vocab))
        return math.log((self.pairs.get((left, right), 0) + self.alpha)
                        / (self.contexts.get(left, 0) + self.alpha * self.tag_vocab))

    def word_delta(self, left, right, placement, word, growth):
        inner = word.split()
        candidate_tags = self.tags(inner[0])
        if placement == "L":
            previous = left[-2] if len(left) >= 2 else None
            neighbor_tags = self.tags(previous) if previous else frozenset()
            transitions = [self.transition(a, b) for a in (neighbor_tags or {None}) for b in candidate_tags]
        else:
            neighbor = right[1] if len(right) >= 2 else None
            neighbor_tags = self.tags(neighbor) if neighbor else frozenset()
            transitions = [self.transition(a, b) for a in candidate_tags for b in (neighbor_tags or {None})]
        pos = max(transitions) if transitions else -12.0
        # Preserve ordinary-word preference without letting frequency dominate
        # the syntactic signal or reward short filler units.
        freq = sum(zipf_frequency(token, "en") for token in inner)
        short_penalty = 2.5 if len(inner) == 1 and len(inner[0]) <= 2 else 0.0
        return 1.4 * pos + 0.08 * freq + 0.06 * len(word) - short_penalty


def audit(words: list[str], seed: int, scorer: BrownPOSScorer) -> dict:
    rendered = textify(words)
    tape = normalize_letters(rendered)
    independent_tape = "".join(ch.lower() for ch in rendered if ch.isascii() and ch.isalpha())
    checks = mechanical_admission_checks(rendered, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    checks["independent_exact_audit"] = bool(independent_tape) and independent_tape == independent_tape[::-1] and independent_tape == tape
    return {
        "seed": seed,
        "rendered": rendered,
        "words": words,
        "letters": len(tape),
        "mechanical_checks": checks,
        "mechanically_eligible": all(checks.values()),
        "independent_normalized_letters": independent_tape,
        "render_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "brown_pos_sentences": scorer.sentence_count,
        "reader_status": "not_run",
    }


def run(*, seeds: int, vocabulary_size: int, beam: int, candidate_limit: int) -> dict:
    vocabulary = [word for word in build_vocab(vocabulary_size)
                  if (len(word) > 2 or word in ORDINARY_TWO_LETTER_WORDS)
                  and word not in REPEATABLE_FUNCTION_WORDS]
    vocabulary_set = set(vocabulary)
    word_tags, pairs, contexts, starts, sentence_count = brown_pos_model(vocabulary_set)
    tries = WordTries(vocabulary)
    scorer = BrownPOSScorer(word_tags, pairs, contexts, starts, sentence_count)
    records = []
    for seed in range(seeds):
        words = beam_search(tries, scorer, min_letters=MIN_LETTERS,
                            max_steps=260, beam_width=beam,
                            candidate_limit=candidate_limit, seed=seed,
                            diversity=1.4, max_word_uses=2)
        if words:
            records.append(audit(words, seed, scorer))
    return {
        "status": "complete_brown_pos_soft_exact_search",
        "config": {"seeds": seeds, "vocabulary_size": len(vocabulary),
                   "beam": beam, "candidate_limit": candidate_limit,
                   "minimum_letters": MIN_LETTERS, "maximum_letters": MAX_LETTERS,
                   "machine_readability_certification": False},
        "brown_pos_sentences": sentence_count,
        "records": records,
        "mechanically_eligible": [row for row in records if row["mechanically_eligible"]],
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "corpus": "NLTK Brown universal POS tags, local copy",
                       "vocabulary": "build_vocab with ordinary short-word filter"},
        "reader_gate": "No readability claim; an eligible surface requires randomized blinded human readers with intact prose and shuffled controls.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--vocabulary-size", type=int, default=16000)
    parser.add_argument("--beam", type=int, default=320)
    parser.add_argument("--candidate-limit", type=int, default=600)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    result = run(seeds=args.seeds, vocabulary_size=args.vocabulary_size,
                 beam=args.beam, candidate_limit=args.candidate_limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "records": len(result["records"]),
                      "mechanically_eligible": len(result["mechanically_eligible"])}, sort_keys=True))


if __name__ == "__main__":
    main()
