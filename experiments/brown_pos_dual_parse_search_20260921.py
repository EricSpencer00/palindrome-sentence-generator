"""Exhaustive POS/bigram dual-parse search over the exact word residual.

This experiment compiles two independent Brown-derived sentence grammars into
word slots.  It intersects them outside-in while the unmatched character tape
is live.  A word enters a state only if its POS role and its reading-order
bigram are corpus-attested; no complete sentence is generated before exact
closure.  Programmatic constraints only propose or reject candidates.  They
never certify readability.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import (
    REPEATABLE_FUNCTION_WORDS,
    is_lexical_word,
    mechanical_admission_checks,
    normalize_letters,
)
from llm_palindrome.dual_parse import word_residual_search
from llm_palindrome.syntax import OPENING_TAGS


ID = "brown-pos-dual-parse-search-20260921"
DROP_TAGS = frozenset({".", "X"})


def _ordinary(word: str) -> bool:
    return (
        bool(re.fullmatch(r"[a-z]+", word))
        and is_lexical_word(word)
        and (len(word) != 2 or word in {
            "ah", "am", "an", "as", "at", "be", "by", "do", "go", "he",
            "if", "in", "is", "it", "me", "my", "no", "of", "oh", "on",
            "or", "ox", "so", "to", "up", "us", "we",
        })
    )


def extract_inventory(*, words_per_tag: int, shape_limit: int,
                      min_bigram_count: int) -> dict:
    """Create a compact replayable inventory from Brown's forward prose."""
    from nltk.corpus import brown

    word_counts: dict[str, Counter] = defaultdict(Counter)
    shape_counts: Counter = Counter()
    bigram_counts: Counter = Counter()
    for tagged in brown.tagged_sents(tagset="universal"):
        row = [
            (word.casefold(), tag)
            for word, tag in tagged
            if tag not in DROP_TAGS and word.isascii() and word.isalpha()
        ]
        row = [(word, tag) for word, tag in row if _ordinary(word)]
        if not row:
            continue
        for word, tag in row:
            word_counts[tag][word] += 1
        bigram_counts.update((a[0], b[0]) for a, b in zip(row, row[1:]))
        shape = tuple(tag for _, tag in row)
        if 4 <= len(shape) <= 9 and shape[0] in OPENING_TAGS and "VERB" in shape:
            shape_counts[shape] += 1

    vocabulary = {
        tag: [word for word, _count in counts.most_common(words_per_tag)]
        for tag, counts in word_counts.items()
    }
    vocabulary_set = {word for words in vocabulary.values() for word in words}
    shapes = [
        {"tags": list(shape), "count": count}
        for shape, count in shape_counts.most_common(shape_limit)
        if all(vocabulary.get(tag) for tag in shape)
    ]
    bigrams = [
        [left, right, count]
        for (left, right), count in bigram_counts.items()
        if count >= min_bigram_count
        and left in vocabulary_set
        and right in vocabulary_set
    ]
    payload = {
        "source": "NLTK Brown tagged sentences; forward prose only",
        "words_per_tag": words_per_tag,
        "shape_limit": shape_limit,
        "min_bigram_count": min_bigram_count,
        "vocabulary": vocabulary,
        "shapes": shapes,
        "bigrams": bigrams,
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()
    return payload


def _slots(shape: Iterable[str], vocabulary: dict[str, list[str]], side: str):
    return tuple(
        (f"{side}:{index}:{tag}", tuple(vocabulary[tag]))
        for index, tag in enumerate(shape)
    )


def _audit(text: str) -> dict:
    tape = normalize_letters(text)
    reverse = tape[::-1]
    mismatch = next(
        ((index, tape[index], tape[-1 - index])
         for index in range(len(tape) // 2)
         if tape[index] != tape[-1 - index]),
        None,
    )
    return {
        "letters": len(tape),
        "two_pointer_exact": bool(tape) and mismatch is None,
        "first_mismatch": mismatch,
        "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(reverse.encode()).hexdigest(),
    }


def search_inventory(inventory: dict, *, max_shape_pairs: int,
                     states_per_pair: int, max_results: int) -> dict:
    vocabulary = inventory["vocabulary"]
    bigrams = {(left, right) for left, right, _count in inventory["bigrams"]}
    shapes = [tuple(row["tags"]) for row in inventory["shapes"]]
    rows: list[dict] = []
    frontiers: list[dict] = []
    states = transitions = pairs = capped = 0

    def allow_choice(side: str, word: str, neighbor: str | None, _role: str) -> bool:
        if neighbor is None:
            return True
        pair = (neighbor, word) if side == "left" else (word, neighbor)
        return pair in bigrams

    def allow_partial(left_words: tuple[str, ...], right_words: tuple[str, ...]) -> bool:
        content = [
            word for word in left_words + right_words
            if word not in REPEATABLE_FUNCTION_WORDS
        ]
        return (
            len(content) == len(set(content))
            and all(word != word[::-1] for word in content)
        )

    for left_shape in shapes:
        for right_shape in shapes:
            if pairs >= max_shape_pairs or len(rows) >= max_results:
                break
            pairs += 1
            result = word_residual_search(
                _slots(left_shape, vocabulary, "A/B"),
                _slots(right_shape, vocabulary, "B-prime/A-prime"),
                max_states=states_per_pair,
                max_results=max_results - len(rows),
                allow_choice=allow_choice,
                allow_partial=allow_partial,
            )
            states += result["states"]
            transitions += result["transitions"]
            capped += int(result["cap_reached"])
            for frontier in result["dead_frontiers"][:3]:
                frontiers.append({
                    **frontier,
                    "left_shape": list(left_shape),
                    "right_shape": list(right_shape),
                })
            for closure in result["results"]:
                rendered = closure["rendered"][:1].upper() + closure["rendered"][1:] + "."
                audit = _audit(rendered)
                admission = mechanical_admission_checks(
                    rendered, min_letters=39, max_letters=220
                )
                rows.append({
                    **closure,
                    "rendered": rendered,
                    "left_shape": list(left_shape),
                    "right_shape": list(right_shape),
                    "audit": audit,
                    "mechanical_admission": admission,
                    "mechanically_admitted": all(admission.values()),
                })
        if pairs >= max_shape_pairs or len(rows) >= max_results:
            break

    frontiers.sort(key=lambda row: (-row["matched_letters"], len(row["residual"])))
    admitted = [row for row in rows if row["mechanically_admitted"]]
    return {
        "experiment_id": ID,
        "method": "exact word-residual product of independent POS plans with forward-bigram transition gates",
        "stats": {
            "shape_pairs": pairs,
            "states": states,
            "transitions": transitions,
            "capped_shape_pairs": capped,
            "exact_closures": len(rows),
            "mechanically_admitted_gt38": len(admitted),
        },
        "exact_candidates": rows,
        "mechanically_admitted_candidates": admitted,
        "deepest_frontiers": frontiers[:40],
        "inventory": {
            "sha256": inventory["sha256"],
            "words_per_tag": inventory["words_per_tag"],
            "shape_count": len(shapes),
            "bigram_count": len(bigrams),
            "source": inventory["source"],
        },
        "provenance": {
            "complete_sentence_enumeration": False,
            "finished_tape_reversal": False,
            "post_hoc_repair": False,
            "catalogue_text": False,
            "per_candidate_rlaif": False,
            "central_mechanical_admission": True,
            "readability_claim": "requires blinded human readers",
        },
        "reader_packet": [],
        "status": (
            "exact admitted candidates require blinded study"
            if admitted else
            "no exact admitted candidate in this bounded grammar product"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--write-inventory", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--words-per-tag", type=int, default=240)
    parser.add_argument("--shape-limit", type=int, default=64)
    parser.add_argument("--min-bigram-count", type=int, default=2)
    parser.add_argument("--max-shape-pairs", type=int, default=4096)
    parser.add_argument("--states-per-pair", type=int, default=150_000)
    parser.add_argument("--max-results", type=int, default=500)
    args = parser.parse_args()

    if args.inventory:
        inventory = json.loads(args.inventory.read_text())
    else:
        inventory = extract_inventory(
            words_per_tag=args.words_per_tag,
            shape_limit=args.shape_limit,
            min_bigram_count=args.min_bigram_count,
        )
    if args.write_inventory:
        args.write_inventory.write_text(json.dumps(inventory, separators=(",", ":")) + "\n")
    result = search_inventory(
        inventory,
        max_shape_pairs=args.max_shape_pairs,
        states_per_pair=args.states_per_pair,
        max_results=args.max_results,
    )
    result["generator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))


if __name__ == "__main__":
    main()
