"""Search a Brown-shaped lexical lattice for exact, original prose.

This is a constructive branch after the fixed authored-template search.  It
does not copy Brown sentences: Brown supplies only universal-POS *shapes* and
word tags.  Lexical material comes from a fixed frequency vocabulary plus a
small authored supplement.  Every closure is independently audited and every
failure is retained with its next construction target.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable

from nltk.corpus import brown

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters
from llm_palindrome.centerout import COState, _expand
from llm_palindrome.search import WordTries
from wordfreq import zipf_frequency, top_n_list


TAGS = ("DET", "PRON", "NUM", "ADP", "ADJ", "ADV", "NOUN", "VERB", "CONJ")
AUTHORED: dict[str, tuple[str, ...]] = {
    "DET": "a an the some one my his her our their this that".split(),
    "PRON": "i we he she they it you me us them who".split(),
    "NUM": "one two three four five six seven eight nine ten".split(),
    "ADP": "at in on by for with from near after before under over".split(),
    "ADJ": "kind small young old red warm fresh careful patient quiet busy".split(),
    "ADV": "today again well slowly carefully outside inside away home early late now often".split(),
    "NOUN": (
        "aide artist baker child cook driver farmer friend guard nurse parent poet pupil teacher worker men "
        "woman man girl boy mother father sister brother doctor helper writer reader singer dancer note book "
        "map maps letter memo memos loaf meal gift plan key door song story task test cup cake room garden "
        "bread tool horse dog cat bird gate road class table chair house river flower apple drawer reward devil "
        "mood doom pals slap dial laid spit tips loop pool flow wolf stop pots step pets part trap star rats "
        "time denim mined diaper straw warts dessert desserts stress"
    ).split(),
    "VERB": (
        "aid asks bakes builds calls carries checks cleans closes cooks draws drives eats feels finds gives grows "
        "guards hears helps holds keeps likes lives loves makes meets moves opens paints plans reads repairs saves "
        "sees sends shares sings starts stops takes tells tests uses visits waits walks watches writes works rips "
        "inspire serves serve records learns teaches draw drew keep peek live lived saw was emit emits repaid "
        "delivers reviled stressed taps patrol"
    ).split(),
    "CONJ": "and but or yet".split(),
}
PROPER_NAMES = "diana anna emma grace helen jane laura maria maya nina olivia sara sophia".split()

# Whole-sentence Brown shapes are useful diagnostics, but their most frequent
# entries are short.  These authored multi-clause frames make the next
# construction attempt genuinely different: each side must carry two linked
# subject--predicate units, with a shared conjunction or adjunct slot.  The
# words are still selected by the exact lattice, never copied from a Brown
# sentence.
CONSTRUCTIVE_SHAPES: tuple[tuple[str, ...], ...] = (
    ("DET", "NOUN", "VERB", "DET", "NOUN", "CONJ", "DET", "NOUN", "VERB"),
    ("PRON", "VERB", "DET", "NOUN", "CONJ", "PRON", "VERB", "NOUN"),
    ("DET", "ADJ", "NOUN", "VERB", "ADP", "DET", "NOUN", "CONJ", "PRON", "VERB"),
    ("PRON", "VERB", "ADP", "DET", "NOUN", "CONJ", "DET", "NOUN", "VERB"),
    ("DET", "NOUN", "VERB", "ADV", "CONJ", "DET", "NOUN", "VERB", "DET", "NOUN"),
    ("NOUN", "VERB", "DET", "NOUN", "CONJ", "PRON", "VERB", "ADP", "DET", "NOUN"),
    ("DET", "NOUN", "VERB", "DET", "ADJ", "NOUN", "CONJ", "DET", "NOUN", "VERB"),
)


def brown_roles(limit: int) -> tuple[dict[str, list[str]], list[tuple[str, ...]], dict[str, frozenset[str]]]:
    counts: Counter[str] = Counter()
    tags: dict[str, Counter[str]] = defaultdict(Counter)
    shape_counts: Counter[tuple[str, ...]] = Counter()
    for sentence in brown.tagged_sents(tagset="universal"):
        shape = tuple(tag for _, tag in sentence if tag not in {".", "X", "PRT"})
        if 3 <= len(shape) <= 9:
            shape_counts[shape] += 1
        for raw, tag in sentence:
            word = raw.lower()
            if word.isalpha() and word.isascii() and len(word) > 1:
                counts[word] += 1
                tags[word][tag] += 1
    roles: dict[str, list[str]] = {}
    for tag in TAGS:
        ranked = sorted(
            ((tags[word][tag], counts[word], word) for word in counts if tags[word][tag] >= 2),
            reverse=True,
        )[:limit]
        words = [word for _, _, word in ranked if word != word[::-1]]
        words.extend(AUTHORED.get(tag, ()))
        roles[tag] = list(dict.fromkeys(words))
    roles["PROPN"] = PROPER_NAMES
    shapes = [shape for shape, _ in shape_counts.most_common(120)]
    # Keep Brown's observed shapes first for the diagnostic branch, then add
    # authored multi-clause frames for the constructive branch.  Stable
    # de-duplication makes the run replayable across NLTK versions.
    shapes = list(dict.fromkeys(shapes + list(CONSTRUCTIVE_SHAPES)))
    # The supplement is not a corpus sentence; it only restores useful tags
    # for ordinary authored words Brown saw too rarely or not at all.
    compatibility: dict[str, set[str]] = defaultdict(set)
    for tag, words in roles.items():
        for word in words:
            compatibility[word].add(tag)
    return roles, shapes, {word: frozenset(tags) for word, tags in compatibility.items()}


def solve_pair(
    left_shape: tuple[str, ...], right_shape: tuple[str, ...], roles: dict[str, list[str]],
    compatibility: dict[str, frozenset[str]], tries: WordTries, *, limit: int, state_budget: int,
) -> tuple[list[tuple[tuple[str, ...], tuple[str, ...]]], int]:
    found: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    states = 0

    def visit(state: COState) -> None:
        nonlocal states
        states += 1
        if states > state_budget or len(found) >= limit:
            return
        if len(state.left) == len(left_shape) and len(state.right) == len(right_shape):
            if not state.overhang:
                key = (state.left, state.right)
                if key not in seen:
                    seen.add(key)
                    found.append(key)
            return
        for placement, word, overhang, owner in _expand(state, tries, 1200):
            if word in state.left or word in state.right:
                continue
            if placement == "L":
                position = len(left_shape) - 1 - len(state.left)
                if position < 0 or left_shape[position] not in compatibility.get(word, ()):
                    continue
                child = COState(0.0, (word,) + state.left, state.right, overhang, owner, 0.0)
            else:
                position = len(state.right)
                if position >= len(right_shape) or right_shape[position] not in compatibility.get(word, ()):
                    continue
                child = COState(0.0, state.left, state.right + (word,), overhang, owner, 0.0)
            visit(child)

    for word in roles.get(left_shape[-1], ()):
        visit(COState(0.0, (word,), (), normalize_letters(word)[::-1], "R", 0.0))
        if states > state_budget or len(found) >= limit:
            break
    return found, states


def audit(pair: tuple[tuple[str, ...], tuple[str, ...]]) -> dict[str, object]:
    text = " ".join(pair[0] + pair[1])
    checks = mechanical_admission_checks(text, min_letters=30, max_letters=500)
    checks["independent_exact_audit"] = normalize_letters(text) == normalize_letters(text)[::-1]
    checks["shape_pair_reverse"] = normalize_letters(" ".join(pair[0])) == normalize_letters(" ".join(pair[1]))[::-1]
    return {
        "text": text,
        "left": " ".join(pair[0]),
        "right": " ".join(pair[1]),
        "letters": len(normalize_letters(text)),
        "checks": checks,
        "mechanically_eligible": all(checks.values()),
        "reader_status": "not_run",
    }


def run(*, role_limit: int, shape_limit: int, pair_limit: int, state_budget: int,
        custom_only: bool = False) -> dict[str, object]:
    roles, all_shapes, compatibility = brown_roles(role_limit)
    shapes = list(CONSTRUCTIVE_SHAPES) if custom_only else all_shapes[:shape_limit]
    words = list(dict.fromkeys(word for values in roles.values() for word in values))
    tries = WordTries(words)
    rows: list[dict[str, object]] = []
    shape_stats: list[dict[str, object]] = []
    for left_shape in shapes:
        for right_shape in shapes:
            found, states = solve_pair(
                left_shape, right_shape, roles, compatibility, tries,
                limit=pair_limit, state_budget=state_budget,
            )
            shape_stats.append({"left_shape": left_shape, "right_shape": right_shape,
                                "states": states, "closures": len(found)})
            rows.extend(audit(pair) for pair in found)
    unique: dict[str, dict[str, object]] = {}
    for row in rows:
        unique.setdefault(normalize_letters(str(row["text"])), row)
    rows = sorted(unique.values(), key=lambda row: (-int(row["letters"]), str(row["text"])))
    return {
        "status": "complete_brown_shape_lattice_exact_search",
        "config": {"role_limit": role_limit, "shape_limit": shape_limit,
                   "pair_limit": pair_limit, "state_budget": state_budget,
                   "shape_source": ("authored multi-clause POS frames" if custom_only
                                    else "Brown universal POS sequences plus authored frames"),
                   "machine_readability_certification": False},
        "lexicon": {"roles": roles, "vocabulary_size": len(words),
                    "sha256": hashlib.sha256("\n".join(words).encode()).hexdigest()},
        "shape_count": len(shapes),
        "shape_stats": shape_stats,
        "records": rows,
        "mechanically_eligible": [row for row in rows if row["mechanically_eligible"]],
        "next_construction_operator_if_empty": (
            "Keep only exact lexical closures, then use a clause-boundary repair pass that changes "
            "one POS-tagged word at a time while preserving the shared character tape."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--role-limit", type=int, default=160)
    parser.add_argument("--shape-limit", type=int, default=36)
    parser.add_argument("--pair-limit", type=int, default=3)
    parser.add_argument("--state-budget", type=int, default=7000)
    parser.add_argument("--custom-only", action="store_true",
                        help="run only the authored multi-clause frames")
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    result = run(role_limit=args.role_limit, shape_limit=args.shape_limit,
                 pair_limit=args.pair_limit, state_budget=args.state_budget,
                 custom_only=args.custom_only)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "records": len(result["records"]),
                      "mechanically_eligible": len(result["mechanically_eligible"])}, sort_keys=True))


if __name__ == "__main__":
    main()
