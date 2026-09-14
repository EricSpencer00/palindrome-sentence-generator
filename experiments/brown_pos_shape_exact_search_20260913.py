"""Exact search under attested Brown sentence part-of-speech skeletons.

The lattice still owns every letter.  The new operator constrains each word
position to a universal POS tag sequence used by an attested Brown sentence,
then scores only the legal exact transitions.  The POS skeleton is a
construction prior, never a readability certificate; every output is audited
independently and must still pass the human-reader gate.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wordfreq import zipf_frequency

from llm_palindrome.admission import (
    REPEATABLE_FUNCTION_WORDS, has_only_ordinary_short_words,
    mechanical_admission_checks, normalize_letters,
)
from llm_palindrome.bigram import BigramModel
from llm_palindrome.generate import build_vocab
from llm_palindrome.scoring import CoherentScorer
from llm_palindrome.search import State, WordTries, _expand
from llm_palindrome.textify import textify


MIN_LETTERS, MAX_LETTERS = 100, 180


def brown_shapes(min_words: int = 12, max_words: int = 20) -> tuple[dict[str, frozenset[str]], list[tuple[str, ...]], int]:
    from nltk.corpus import brown

    tags: dict[str, set[str]] = defaultdict(set)
    shapes: list[tuple[str, ...]] = []
    sentence_count = 0
    for sentence in brown.tagged_sents(tagset="universal"):
        words = [(word.casefold(), tag) for word, tag in sentence
                 if word.isascii() and word.isalpha()]
        for word, tag in words:
            tags[word].add(tag)
        if min_words <= len(words) <= max_words:
            shapes.append(tuple(tag for _, tag in words))
            sentence_count += 1
    return {word: frozenset(values) for word, values in tags.items()}, list(dict.fromkeys(shapes)), sentence_count


def _words(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(word for unit in left + right for word in unit.split())


def exact_shape_search(shape: tuple[str, ...], tags: dict[str, frozenset[str]],
                       tries: WordTries, scorer: CoherentScorer, *, beam_width: int,
                       candidate_limit: int, min_zipf: float) -> State | None:
    count = len(shape)
    beam = [State(0.0, (), (), "", "L", 0.0)]
    for _ in range(count):
        pool: list[State] = []
        for state in beam:
            for placement, word, overhang, side in _expand(state, tries, candidate_limit):
                index = len(state.left) if placement == "L" else count - 1 - len(state.right)
                if index < 0 or index >= count or shape[index] not in tags.get(word, ()):
                    continue
                left = state.left + (word,) if placement == "L" else state.left
                right = state.right if placement == "L" else (word,) + state.right
                words = _words(left, right)
                if (len(set(words)) != len(words)
                        or not has_only_ordinary_short_words(words)
                        or any(word == word[::-1] and word not in REPEATABLE_FUNCTION_WORDS
                               for word in words)
                        or any(zipf_frequency(word, "en") < min_zipf for word in words)):
                    continue
                delta = scorer.word_delta(left, right, placement, word,
                                          "append" if placement == "L" else "prepend")
                pool.append(State(-(state.score + delta), left, right, overhang, side,
                                  state.score + delta))
        if not pool:
            return None
        pool.sort(key=lambda state: (state.sort_key, state.left, state.right))
        beam = pool[:beam_width]
    return max((state for state in beam if state.overhang == state.overhang[::-1]),
               key=lambda state: state.letters, default=None)


def audit(state: State, shape: tuple[str, ...], seed: int) -> dict:
    rendered = textify(list(state.left) + list(state.right))
    tape = normalize_letters(rendered)
    independent = bool(tape) and tape == tape[::-1] and all("a" <= char <= "z" for char in tape)
    checks = mechanical_admission_checks(rendered, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    checks["independent_exact_audit"] = independent
    return {"seed": seed, "shape": shape, "rendered": rendered,
            "words": list(state.left) + list(state.right), "letters": len(tape),
            "mechanical_checks": checks, "mechanically_eligible": all(checks.values()),
            "render_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
            "reader_status": "not_run"}


def run(*, shape_limit: int, vocabulary_size: int, beam_width: int,
        candidate_limit: int, min_zipf: float) -> dict:
    tags, shapes, sentence_count = brown_shapes()
    shapes = shapes[:shape_limit]
    vocabulary = [word for word in build_vocab(vocabulary_size)
                  if len(word) > 2 or word in {"a", "i"}]
    vocabulary = [word for word in vocabulary if zipf_frequency(word, "en") >= min_zipf]
    tries = WordTries(vocabulary)
    bigrams = BigramModel.from_file(str(ROOT / "data" / "count_2w.txt"), vocab=set(vocabulary))
    scorer = CoherentScorer(bigrams, freq_weight=0.1, length_weight=0.15,
                            phrase_weight=1.0, short_penalty=4.0)
    records = []
    for seed, shape in enumerate(shapes):
        state = exact_shape_search(shape, tags, tries, scorer,
                                   beam_width=beam_width,
                                   candidate_limit=candidate_limit,
                                   min_zipf=min_zipf)
        if state is not None:
            records.append(audit(state, shape, seed))
    records.sort(key=lambda row: (-row["letters"], row["render_sha256"]))
    return {"status": "complete_brown_pos_shape_exact_search",
            "config": {"shape_limit": shape_limit, "vocabulary_size": len(vocabulary),
                       "beam_width": beam_width, "candidate_limit": candidate_limit,
                       "min_zipf": min_zipf, "minimum_letters": MIN_LETTERS,
                       "machine_readability_certification": False},
            "brown_sentences_in_band": sentence_count, "shapes_searched": len(shapes),
            "records": records,
            "mechanically_eligible": [row for row in records if row["mechanically_eligible"]],
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "corpus": "NLTK Brown universal POS tags, local copy",
                           "vocabulary": "build_vocab filtered by wordfreq Zipf"},
            "reader_gate": "No readability claim; an eligible surface requires randomized blinded human readers with intact prose and shuffled controls."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shape-limit", type=int, default=300)
    parser.add_argument("--vocabulary-size", type=int, default=18000)
    parser.add_argument("--beam-width", type=int, default=500)
    parser.add_argument("--candidate-limit", type=int, default=800)
    parser.add_argument("--min-zipf", type=float, default=4.0)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    result = run(shape_limit=args.shape_limit, vocabulary_size=args.vocabulary_size,
                 beam_width=args.beam_width, candidate_limit=args.candidate_limit,
                 min_zipf=args.min_zipf)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "records": len(result["records"]),
                      "mechanically_eligible": len(result["mechanically_eligible"])}, sort_keys=True))


if __name__ == "__main__":
    main()
