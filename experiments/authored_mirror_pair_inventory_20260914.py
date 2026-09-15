"""Enumerate original mirror-pairs from an authored semantic lexicon.

Unlike a corpus-span miner, this inventory contains only hand-authored lexical
roles and clause templates.  The solver matches their character tapes exactly
from the clause centres outward, then ranks complete pairs by ordinary word
frequency and observed joins.  It never copies a sentence and it does not call
the resulting pairs readable until they are assembled and shown to blinded
readers.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import (
    REPEATABLE_FUNCTION_WORDS,
    mechanical_admission_checks,
    normalize_letters,
    tokenize,
)
from llm_palindrome.bigram import BigramModel
from llm_palindrome.centerout import _expand, COState
from llm_palindrome.search import WordTries
from llm_palindrome.validator import normalize
from wordfreq import zipf_frequency


# Small, transparent semantic roles.  The inventory is intentionally authored;
# names are ordinary lexical options but no corpus sentence is imported.
LEXICON: dict[str, tuple[str, ...]] = {
    "DET": "a an the some one my his her our their this that".split(),
    "PRON": "i we he she they it you me us them who".split(),
    "NUM": "one two three four five six seven eight nine ten".split(),
    "ADP": "at in on by for with from near after before under over".split(),
    "ADJ": "kind small young old red warm fresh careful patient quiet busy".split(),
    "ADV": "today again well slowly carefully outside inside away home early late now often".split(),
    "NOUN": (
        "aide artist baker child cook driver farmer friend guard nurse parent poet pupil teacher worker men "
        "woman man girl boy mother father sister brother doctor helper writer reader singer dancer "
        "note book map maps letter memo memos loaf meal gift plan key door song story task test cup cake room garden "
        "bread tool horse dog cat bird gate road class table chair house river flower apple drawer reward "
        "devil mood doom pals slap dial laid spit tips loop pool flow wolf stop pots step pets part trap "
        "star rats time denim mined diaper straw warts dessert desserts stress".split()
    ),
    "VERB": (
        "aid asks bakes builds calls carries checks cleans closes cooks draws drives eats feels finds "
        "gives grows guards hears helps holds keeps likes lives loves makes meets moves opens paints "
        "plans reads repairs saves sees sends shares sings starts stops takes tells tests uses visits "
        "waits walks watches writes works rips inspire inspires serves serve record records learns teaches "
        "draw draws drew keep keeps peek peeks live lived saw sees was emit emits repaid delivers reviled "
        "stressed stress stops stop taps pat patrol".split()
    ),
    "PROPN": "anna diana emma grace helen jane laura maria maya nina olivia sara sophia".split(),
    "CONJ": "and but or yet".split(),
}

# Clause templates are deliberately short enough to enumerate, but long enough
# that concatenating distinct pairs can reach the 100-letter target.
SHAPES: tuple[tuple[str, ...], ...] = (
    ("DET", "NOUN", "VERB"),
    ("PRON", "VERB"),
    ("PRON", "VERB", "ADV"),
    ("ADJ", "NOUN", "VERB"),
    ("DET", "ADJ", "NOUN", "VERB"),
    ("DET", "NOUN", "VERB", "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "NOUN"),
    ("DET", "NOUN", "VERB", "NUM", "NOUN"),
    ("PRON", "VERB", "DET", "NOUN"),
    ("PRON", "VERB", "NOUN"),
    ("DET", "ADJ", "NOUN", "VERB", "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "DET", "ADJ", "NOUN"),
    ("DET", "NOUN", "VERB", "ADP", "DET", "NOUN"),
    ("NOUN", "VERB", "DET", "NOUN"),
    ("NOUN", "VERB", "ADP", "DET", "NOUN"),
    ("PRON", "VERB", "ADP", "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "ADV"),
    ("DET", "NOUN", "VERB", "PROPN"),
    ("PRON", "VERB", "PROPN"),
    ("NOUN", "NOUN", "VERB", "PROPN"),
    ("DET", "NOUN", "VERB", "DET", "PROPN"),
    # Longer productive frames keep the two readings as ordinary clauses
    # while allowing the character seam to fall inside a word.  These are
    # authored role sequences, not copied corpus sentences.
    ("DET", "ADJ", "NOUN", "VERB", "DET", "NOUN", "ADP", "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "DET", "ADJ", "NOUN", "ADP", "DET", "NOUN"),
    ("PRON", "VERB", "DET", "NOUN", "ADP", "DET", "NOUN"),
    ("PROPN", "VERB", "DET", "NOUN", "ADP", "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "ADV", "ADP", "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "DET", "NOUN", "CONJ", "DET", "NOUN"),
    ("DET", "ADJ", "NOUN", "VERB", "ADV"),
    ("PRON", "VERB", "DET", "ADJ", "NOUN", "ADV"),
    ("PROPN", "VERB", "DET", "ADJ", "NOUN"),
    ("DET", "NOUN", "VERB", "PROPN", "ADV"),
)


def vocabulary() -> list[str]:
    return list(dict.fromkeys(word for words in LEXICON.values() for word in words
                              if word != word[::-1]))


def category_ok(word: str, category: str) -> bool:
    return word in LEXICON.get(category, ())


def compatible_tags(word: str) -> set[str]:
    return {tag for tag, words in LEXICON.items() if word in words}


def solve_pair(left_shape: tuple[str, ...], right_shape: tuple[str, ...], *, limit: int,
               state_budget: int) -> tuple[list[tuple[tuple[str, ...], tuple[str, ...]]], int]:
    tries = WordTries(vocabulary())
    out: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    states = 0

    def tag_ok(word: str, category: str) -> bool:
        return category in compatible_tags(word)

    def visit(state: COState) -> None:
        nonlocal states
        states += 1
        if states > state_budget or len(out) >= limit:
            return
        if (len(state.left) == len(left_shape) and
                len(state.right) == len(right_shape)):
            if not state.overhang:
                key = (state.left, state.right)
                if key not in seen:
                    seen.add(key)
                    out.append(key)
            return
        for placement, word, overhang, owner in _expand(state, tries, 5000):
            if placement == "L":
                position = len(left_shape) - 1 - len(state.left)
                if position < 0 or not tag_ok(word, left_shape[position]):
                    continue
                if (word not in REPEATABLE_FUNCTION_WORDS and
                        word in state.left + state.right):
                    continue
                child = COState(0.0, (word,) + state.left, state.right,
                                overhang, owner, 0.0)
            else:
                position = len(state.right)
                if position >= len(right_shape) or not tag_ok(word, right_shape[position]):
                    continue
                if (word not in REPEATABLE_FUNCTION_WORDS and
                        word in state.left + state.right):
                    continue
                child = COState(0.0, state.left, state.right + (word,),
                                overhang, owner, 0.0)
            visit(child)

    # The first emitted word sits immediately beside the centre on the left.
    for word in LEXICON.get(left_shape[-1], ()):
        if word == word[::-1]:
            continue
        visit(COState(0.0, (word,), (), normalize_letters(word)[::-1], "R", 0.0))
        if len(out) >= limit or states > state_budget:
            break
    return out, states


def score_pair(pair: tuple[tuple[str, ...], tuple[str, ...]], bigrams: BigramModel) -> float:
    left, right = pair
    words = list(left) + list(right)
    joins = sum(bigrams.forward(a, b) for a, b in zip(left, left[1:]))
    joins += sum(bigrams.forward(a, b) for a, b in zip(right, right[1:]))
    return joins + 0.18 * sum(zipf_frequency(word, "en") for word in words)


def audit_pair(pair: tuple[tuple[str, ...], tuple[str, ...]], bigrams: BigramModel) -> dict[str, object]:
    left, right = pair
    text = " ".join(left + right)
    tape = normalize_letters(text)
    checks = mechanical_admission_checks(text, min_letters=6, max_letters=80)
    checks["pair_exact_reverse"] = normalize_letters(" ".join(left)) == normalize_letters(" ".join(right))[::-1]
    checks["independent_exact_audit"] = tape == tape[::-1]
    content = [word for word in tokenize(text)
               if word not in REPEATABLE_FUNCTION_WORDS]
    checks["distinct_pair_content"] = len(set(content)) == len(content)
    return {"left": " ".join(left), "right": " ".join(right), "text": text,
            "letters": len(tape), "score": score_pair(pair, bigrams),
            "mechanical_checks": checks,
            "mechanically_eligible": all(checks.values()),
            "render_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "reader_status": "not_run"}


def run(*, pair_limit: int, state_budget: int) -> dict[str, object]:
    words = vocabulary()
    tries = WordTries(words)
    bigrams = BigramModel.from_file(str(ROOT / "data" / "count_2w.txt"), vocab=set(words))
    rows = []
    stats = []
    for left_shape in SHAPES:
        for right_shape in SHAPES:
            found, states = solve_pair(left_shape, right_shape, limit=pair_limit,
                                       state_budget=state_budget)
            stats.append({"left_shape": left_shape, "right_shape": right_shape,
                          "states": states, "closures": len(found)})
            rows.extend(audit_pair(pair, bigrams) for pair in found)
    rows.sort(key=lambda row: (-row["score"], -row["letters"], row["text"]))
    return {"status": "complete_authored_mirror_pair_inventory",
            "config": {"pair_limit_per_shape_pair": pair_limit,
                       "state_budget_per_shape_pair": state_budget,
                       "shape_count": len(SHAPES),
                       "machine_readability_certification": False},
            "lexicon": {"categories": {k: list(v) for k, v in LEXICON.items()},
                        "vocabulary_size": len(words),
                        "sha256": hashlib.sha256("\n".join(words).encode()).hexdigest()},
            "shape_stats": stats, "pairs": rows,
            "mechanically_eligible": [row for row in rows if row["mechanically_eligible"]],
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "material": "authored lexical roles and clause templates only; no corpus sentence copied"},
            "reader_gate": "Pairs are not readable evidence. Assemble distinct pairs into intact prose, then run exact independent audit and blinded human readers with shuffled controls."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--pair-limit", type=int, default=80)
    parser.add_argument("--state-budget", type=int, default=120000)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    result = run(pair_limit=args.pair_limit, state_budget=args.state_budget)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "pairs": len(result["pairs"]),
                      "mechanically_eligible": len(result["mechanically_eligible"])}, sort_keys=True))


if __name__ == "__main__":
    main()
