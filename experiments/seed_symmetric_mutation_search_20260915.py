"""Search a readable-seed neighbourhood by palindrome-preserving edits.

The seed is useful because it is the only current reader-worthy short item.  A
simple way to grow it is to change or insert a character at a position and the
mirrored position at the same time; exactness is then guaranteed by
construction, while a fresh lexical segmentation must recover the prose.  This
probe exhausts one and two mirrored substitutions and one mirrored insertion,
then applies the independent admission and Brown sentence-shape gates.  It is
not allowed to promote a wrapper around the seed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from functools import lru_cache
from itertools import combinations
from pathlib import Path
import sys

from nltk.corpus import brown
from wordfreq import top_n_list, zipf_frequency

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters
from llm_palindrome.syntax import sentence_like, sentence_shapes, tag_table

SEED = "anaideripsninememossomemeninspirediana"
SHORT_WORDS = frozenset(
    "ah am an as at be by do go he if in is it me my no of oh on or ox so to up us we".split()
)


def lexical_inventory() -> frozenset[str]:
    common = {
        word for word in top_n_list("en", 90_000)
        if word.isascii() and word.isalpha() and len(word) >= 2
        and zipf_frequency(word, "en") >= 3.2
    }
    project = {
        line.strip().casefold() for line in (ROOT / "data" / "lexicon.txt").read_text().splitlines()
        if line.strip()
    }
    return frozenset(word for word in common | project
                     if len(word) > 2 or word in SHORT_WORDS)


def mirrored_substitute(tape: str, position: int, letter: str) -> str:
    other = len(tape) - 1 - position
    chars = list(tape)
    chars[position] = chars[other] = letter
    return "".join(chars)


def mirrored_insert(tape: str, position: int, letter: str) -> str:
    """Insert a mirrored pair and return an exact palindrome."""
    n = len(tape)
    after_left = tape[:position] + letter + tape[position:]
    mirror = n + 1 - position
    return after_left[:mirror] + letter + after_left[mirror:]


def segmenter(lexicon: frozenset[str], max_words: int = 16):
    @lru_cache(maxsize=None)
    def segment(tape: str) -> tuple[tuple[str, ...], ...]:
        if not tape:
            return ((),)
        rows: list[tuple[str, ...]] = []
        for end in range(2, min(len(tape), 13) + 1):
            word = tape[:end]
            if word not in lexicon or word == word[::-1]:
                continue
            for tail in segment(tape[end:]):
                if len(tail) + 1 <= max_words:
                    rows.append((word,) + tail)
                    if len(rows) >= 2_500:
                        return tuple(rows)
        return tuple(rows)

    return segment


def audit(tape: str, words: tuple[str, ...], table, shapes) -> dict:
    text = " ".join(words).capitalize() + "."
    checks = mechanical_admission_checks(text, min_letters=39, max_letters=100)
    return {
        "text": text,
        "letters": len(normalize_letters(text)),
        "exact": normalize_letters(text) == tape == tape[::-1],
        "sentence_like": sentence_like(words, table, shapes),
        "checks": checks,
        "admitted": all(checks.values()) and sentence_like(words, table, shapes),
    }


def run() -> dict:
    tagged = list(brown.tagged_sents(tagset="universal"))
    table = tag_table(tagged)
    shapes = sentence_shapes(tagged, min_words=4, max_words=16)
    lexicon = lexical_inventory()
    segment = segmenter(lexicon)
    seen: set[str] = set()
    records: list[dict] = []
    stats = {"substitution_tapes": 0, "insertion_tapes": 0,
             "lexically_segmentable": 0, "sentence_like": 0,
             "admitted": 0}

    def inspect(tape: str, operation: str) -> None:
        if tape in seen:
            return
        seen.add(tape)
        segmentations = segment(tape)
        if segmentations:
            stats["lexically_segmentable"] += 1
        for words in segmentations:
            row = audit(tape, words, table, shapes)
            if not row["sentence_like"]:
                continue
            stats["sentence_like"] += 1
            row["operation"] = operation
            records.append(row)
            if row["admitted"]:
                stats["admitted"] += 1

    n = len(SEED)
    for position in range(n // 2):
        other = n - 1 - position
        for letter in "abcdefghijklmnopqrstuvwxyz":
            if letter == SEED[position]:
                continue
            stats["substitution_tapes"] += 1
            inspect(mirrored_substitute(SEED, position, letter),
                    f"substitute:{position}:{other}:{letter}")

    substitution_ops = [
        (position, letter)
        for position in range(n // 2)
        for letter in "abcdefghijklmnopqrstuvwxyz"
        if letter != SEED[position]
    ]
    for (left, letter_left), (right, letter_right) in combinations(substitution_ops, 2):
        if left == right:
            continue
        chars = list(SEED)
        chars[left] = chars[n - 1 - left] = letter_left
        chars[right] = chars[n - 1 - right] = letter_right
        inspect("".join(chars), f"substitute2:{left}:{letter_left}:{right}:{letter_right}")

    for position in range(n // 2 + 1):
        for letter in "abcdefghijklmnopqrstuvwxyz":
            stats["insertion_tapes"] += 1
            inspect(mirrored_insert(SEED, position, letter),
                    f"insert:{position}:{letter}")

    unique = {row["text"]: row for row in records}
    return {
        "status": "complete_seed_symmetric_mutation_search",
        "seed": SEED,
        "seed_letters": len(SEED),
        "config": {
            "substitutions": "one and two mirrored character pairs",
            "insertions": "one mirrored character pair",
            "minimum_letters": 39,
            "forbid_seed_wrapper": True,
        },
        "stats": stats | {"unique_sentence_like": len(unique)},
        "admitted": [row for row in unique.values() if row["admitted"]],
        "near_misses": list(unique.values())[:100],
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "lexicon": "wordfreq common types plus data/lexicon.txt",
            "grammar": "NLTK Brown universal POS sentence shapes",
            "source_text_copied": False,
        },
        "next_repair_operator": (
            "Use the best reverse-prefix slot mutation from this neighbourhood as a seed, "
            "then allow typed word-boundary shifts and lexical replacements jointly; do not "
            "wrap or retain the 38-letter seed as a proper interior palindrome."
        ),
        "reader_gate": (
            "No row is reader evidence. A future admitted row requires rendered provenance, "
            "independent exact audit, and randomized blinded intact-prose/shuffled-control ratings."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite output: {args.out}")
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stats": result["stats"], "admitted": len(result["admitted"])}, indent=2))


if __name__ == "__main__":
    main()
