"""Decode exact palindromes whose two readings are coordinated clauses.

This is a representation change from the free POS pilot.  Instead of ranking
an arbitrary word stream by local tags, the decoder intersects one shared
character tape with a finite inventory of *two-clause sentence shapes* on both
readings.  Every partial state must remain a prefix/suffix of a complete
subject--verb clause sequence; exact reversal is still enforced by the
center-out lattice.  The grammar is only a structural proposal filter: a
survivor is not called readable until blinded readers compare it with intact
prose and shuffled controls.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import (
    ORDINARY_TWO_LETTER_WORDS,
    REPEATABLE_FUNCTION_WORDS,
    is_lexical_word,
    mechanical_admission_checks,
    normalize_letters,
)
from llm_palindrome.clause_ngram import ClauseNgramScorer
from llm_palindrome.centerout import centerout_search
from llm_palindrome.generate import build_vocab
from llm_palindrome.search import WordTries
from llm_palindrome.syntax import brown_tables
from llm_palindrome.textify import textify


MIN_LETTERS, MAX_LETTERS = 100, 180

# A compact, authored inventory of ordinary clause shapes.  The conjunction
# belongs to the complete surface, so a two-clause reading cannot collapse to
# an arbitrary POS salad.  Noun/verb ambiguity is resolved by the Brown table
# for each concrete word during the prefix/suffix checks below.
CLAUSE_SHAPES: tuple[tuple[str, ...], ...] = (
    ("PRON", "VERB", "DET", "NOUN"),
    ("PRON", "VERB", "ADP", "DET", "NOUN"),
    ("PRON", "VERB", "DET", "ADJ", "NOUN"),
    ("DET", "NOUN", "VERB", "DET", "NOUN"),
    ("DET", "ADJ", "NOUN", "VERB", "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "ADP", "DET", "NOUN"),
    ("DET", "ADJ", "NOUN", "VERB", "ADP", "DET", "NOUN"),
    ("NOUN", "VERB", "DET", "NOUN"),
    ("NOUN", "VERB", "ADP", "DET", "NOUN"),
    ("ADV", "PRON", "VERB", "DET", "NOUN"),
)


def two_clause_shapes() -> tuple[tuple[str, ...], ...]:
    """Return distinct clause conjunctions, keeping lengths in search range."""
    out: set[tuple[str, ...]] = set()
    for left in CLAUSE_SHAPES:
        for right in CLAUSE_SHAPES:
            shape = left + ("CONJ",) + right
            if 9 <= len(shape) <= 17:
                out.add(shape)
    return tuple(sorted(out, key=lambda shape: (len(shape), shape)))


class ClauseShapePlan:
    """Prefix/suffix feasibility for the authored two-clause inventory."""

    def __init__(self, table: dict[str, frozenset[str]],
                 shapes: Iterable[Sequence[str]]):
        self.table = table
        self.shapes = {tuple(shape) for shape in shapes}
        self.min_words = min(len(shape) for shape in self.shapes)
        self.max_words = max(len(shape) for shape in self.shapes)
        self.prefixes: dict[int, set[tuple[str, ...]]] = {}
        self.suffixes: dict[int, set[tuple[str, ...]]] = {}
        for shape in self.shapes:
            for width in range(1, len(shape) + 1):
                self.prefixes.setdefault(width, set()).add(shape[:width])
                self.suffixes.setdefault(width, set()).add(shape[-width:])

    def _pools(self, words: Sequence[str]):
        pools = []
        for word in words:
            tags = self.table.get(word.lower())
            if not tags:
                return None
            pools.append(tags)
        return pools

    @staticmethod
    def _matches(pools, allowed: set[tuple[str, ...]]) -> bool:
        if pools is None or not allowed:
            return False
        # The shape inventory is short; a direct product is still bounded by
        # the Brown ambiguity of the selected lexical words.  Keep it explicit
        # and deterministic for replay/audit.
        from itertools import product
        return any(tuple(reading) in allowed for reading in product(*pools))

    def _possible(self, words: Sequence[str], index: dict[int, set[tuple[str, ...]]]) -> bool:
        if not words or len(words) > self.max_words:
            return bool(not words and self.shapes)
        return self._matches(self._pools(words), index.get(len(words), set()))

    def state_possible(self, left: Sequence[str], right: Sequence[str]) -> bool:
        return self._possible(left, self.suffixes) and self._possible(right, self.prefixes)

    def complete(self, words: Sequence[str]) -> bool:
        return self._matches(self._pools(words), self.shapes)


def brown_words() -> list[list[str]]:
    from nltk.corpus import brown
    return [[word.casefold() for word in sentence
             if word.isascii() and word.isalpha()] for sentence in brown.sents()]


def native_proper_names() -> set[str]:
    from nltk.corpus import brown
    return {word.casefold() for sentence in brown.tagged_sents()
            for word, tag in sentence
            if tag.startswith("NP") and word.isascii() and word.isalpha()}


def lexical_vocabulary(size: int) -> list[str]:
    proper = native_proper_names()
    words = []
    for word in build_vocab(size):
        word = word.casefold()
        if word in proper or word in REPEATABLE_FUNCTION_WORDS:
            # Function words are restored below; this exclusion only removes
            # proper names and repeatable content-like entries from the first
            # frequency pass.
            pass
        if (word not in proper and
                (len(word) > 2 or word in ORDINARY_TWO_LETTER_WORDS) and
                is_lexical_word(word) and word != word[::-1]):
            words.append(word)
    # Keep ordinary syntax available even when the top-N list is content-heavy.
    for word in sorted(REPEATABLE_FUNCTION_WORDS):
        if word in proper or word == word[::-1] or not is_lexical_word(word):
            continue
        if len(word) > 2 or word in ORDINARY_TWO_LETTER_WORDS:
            words.append(word)
    return list(dict.fromkeys(words))


def allow_state(left: tuple[str, ...], right: tuple[str, ...], plan: ClauseShapePlan) -> bool:
    words = [word for unit in left + right for word in unit.split()]
    content = [word for word in words if word not in REPEATABLE_FUNCTION_WORDS]
    return (len(content) == len(set(content)) and plan.state_possible(left, right))


def split_tape(words: Sequence[str]) -> tuple[list[str], list[str]] | None:
    """Recover center-out's left/right lexicalizations by equal tape length."""
    from llm_palindrome.validator import normalize
    total = normalize(" ".join(words))
    prefix = ""
    for index, word in enumerate(words):
        prefix += normalize(word)
        if len(prefix) * 2 == len(total):
            return list(words[:index + 1]), list(words[index + 1:])
        if len(prefix) * 2 > len(total):
            return None
    return None


def audit(words: list[str], seed: int, plan: ClauseShapePlan) -> dict[str, object]:
    rendered = textify(words)
    tape = normalize_letters(rendered)
    independent = "".join(ch.casefold() for ch in rendered if ch.isascii() and ch.isalpha())
    checks = mechanical_admission_checks(rendered, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    checks["independent_exact_audit"] = bool(independent) and independent == independent[::-1] and independent == tape
    split = split_tape(words)
    checks["has_equal_tape_split"] = split is not None
    checks["two_clause_shape_left"] = bool(split) and plan.complete(split[0])
    checks["two_clause_shape_right"] = bool(split) and plan.complete(split[1])
    return {
        "seed": seed,
        "rendered": rendered,
        "words": words,
        "letters": len(tape),
        "mechanical_checks": checks,
        "mechanically_eligible": all(checks.values()),
        "independent_normalized_letters": independent,
        "render_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "reader_status": "not_run",
    }


def run(*, seeds: int, vocabulary_size: int, beam: int, candidate_limit: int) -> dict[str, object]:
    table, _unused_shapes, _unused_trigrams = brown_tables()
    plan = ClauseShapePlan(table, two_clause_shapes())
    vocabulary = lexical_vocabulary(vocabulary_size)
    vocabulary = [word for word in vocabulary if word in table]
    tries = WordTries(vocabulary)
    scorer = ClauseNgramScorer(brown_words(), order=4)
    records = []
    for seed in range(seeds):
        words = centerout_search(
            tries, scorer, min_letters=MIN_LETTERS, max_steps=100,
            beam_width=beam, candidate_limit=candidate_limit,
            per_parent=12, seed=seed, diversity=0.65, max_overhang=20,
            allow_state=lambda left, right: allow_state(left, right, plan),
            allow_closed=lambda left, right: plan.complete(left) and plan.complete(right),
        )
        if words:
            records.append(audit(words, seed, plan))
    return {
        "status": "complete_clause_shaped_shared_tape_search",
        "config": {
            "seeds": seeds, "vocabulary_size_requested": vocabulary_size,
            "vocabulary_size": len(vocabulary), "beam": beam,
            "candidate_limit": candidate_limit, "minimum_letters": MIN_LETTERS,
            "maximum_letters": MAX_LETTERS, "machine_readability_certification": False,
        },
        "grammar": {"shape_count": len(plan.shapes), "shape_inventory": [list(shape) for shape in sorted(plan.shapes)]},
        "records": records,
        "mechanically_eligible": [row for row in records if row["mechanically_eligible"]],
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "corpus": "NLTK Brown word/tag data for proposal scoring and lexical tags",
            "grammar": "authored subject-verb two-clause shape inventory; no corpus sentence copied",
            "vocabulary_sha256": hashlib.sha256("\\n".join(vocabulary).encode()).hexdigest(),
        },
        "reader_gate": "No readability claim; any survivor requires randomized blinded intact-prose and shuffled-control readers.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--vocabulary-size", type=int, default=14000)
    parser.add_argument("--beam", type=int, default=256)
    parser.add_argument("--candidate-limit", type=int, default=1000)
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
