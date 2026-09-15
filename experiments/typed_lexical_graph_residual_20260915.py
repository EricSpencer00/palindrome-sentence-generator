"""Joint residual search with a small typed lexical/valency graph.

This is a successor to the isolated role-list search.  Words carry number,
tense, and semantic roles; clause templates carry subject/object positions;
and the residual decoder checks those features on both sides while matching
characters.  Brown contributes no sentence text: this experiment is fully
auditable from the inventory below.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters


FUNCTIONS = frozenset("a an the some my our his her this that each i we you he she they it".split())

# The lexical graph is deliberately small and hand-auditable.  Features are
# encoded in slot names rather than inferred after a tape has closed.
LEXICON: dict[str, tuple[str, ...]] = {
    "DET_S": ("a", "an", "the", "my", "our", "his", "her", "this", "that", "each"),
    "DET_P": ("the", "some", "our", "their", "these", "those"),
    "PRON": ("i", "we", "you", "he", "she", "they", "it"),
    "PERSON_S": "artist baker child clerk doctor farmer friend guard helper man parent poet pupil teacher worker writer reader singer dancer king queen nurse painter pilot woman".split(),
    "PERSON_P": "artists bakers children clerks doctors farmers friends guards helpers men parents poets pupils teachers workers writers singers dancers kings queens nurses painters pilots women".split(),
    "THING_S": "book boat bottle cake car cat chair coin cup dog door drum flower game gift hat house key lamp letter map meal memo mirror music note paper phone plant poem room rope ship shoe song stone story table ticket tool train tree vase watch water wheel".split(),
    "THING_P": "books boats bottles cakes cars cats chairs coins cups dogs doors drums flowers games gifts hats houses keys lamps letters maps meals memos mirrors notes papers phones plants poems rooms ropes ships shoes songs stones stories tables tickets tools trains trees vases watches waters wheels".split(),
    "V3": "asks buys calls cooks draws finds gives hears holds keeps likes makes needs opens reads sends shows takes tells uses wants writes".split(),
    "VBASE": "ask buy call cook draw find give hear hold keep like make need open read send show take tell use want write".split(),
    "VPAST": "asked bought called cooked drew found gave heard held kept liked made needed opened read sent showed took told used wanted wrote".split(),
    "ADV": "now then here there often again away home well early late".split(),
    "ADP": "at in on by for with from near after before under over into through around".split(),
}

# Complete clauses.  The two sides may choose different templates, and the
# residual parser solves their word boundaries independently.
SHAPES: tuple[tuple[str, ...], ...] = (
    ("DET_S", "PERSON_S", "V3", "DET_S", "THING_S"),
    ("DET_P", "PERSON_P", "VBASE", "DET_P", "THING_P"),
    ("DET_S", "PERSON_S", "VPAST", "DET_S", "THING_S"),
    ("PRON", "VPAST", "DET_S", "THING_S"),
    ("PRON", "V3", "DET_S", "THING_S"),
    ("PERSON_S", "V3", "THING_S"),
    ("PERSON_S", "VPAST", "THING_S"),
    ("VBASE", "DET_S", "THING_S"),
    ("DET_S", "THING_S", "V3", "DET_S", "THING_S"),
    ("DET_P", "THING_P", "VBASE", "DET_P", "THING_P"),
    ("DET_S", "PERSON_S", "V3", "ADV"),
    ("PRON", "V3", "ADV"),
)

VERB_OBJECTS = {
    # These are broad lexical valencies, not a language model.  A verb absent
    # from the map is never silently treated as transitive.
    verb: frozenset({"THING_S", "THING_P"})
    for verb in LEXICON["V3"] + LEXICON["VBASE"] + LEXICON["VPAST"]
}


def _tape(words: tuple[str, ...]) -> str:
    return "".join(words)


def _content(words: tuple[str, ...]) -> set[str]:
    return {word for word in words if word not in FUNCTIONS}


def _graph_valid(words: tuple[str, ...], slots: tuple[str, ...]) -> bool:
    """Check feature agreement and transitivity for one parsed clause."""
    if len(words) != len(slots):
        return False
    # Determiner/number agreement is carried by the slot itself.
    if any(word not in LEXICON.get(slot, ()) for word, slot in zip(words, slots)):
        return False
    for index, slot in enumerate(slots):
        if slot not in {"V3", "VBASE", "VPAST"}:
            continue
        if index + 1 < len(slots) and slots[index + 1] in {"THING_S", "THING_P"}:
            if words[index] not in VERB_OBJECTS:
                return False
        if index + 2 < len(slots) and slots[index + 1] in {"DET_S", "DET_P"} and slots[index + 2] in {"THING_S", "THING_P"}:
            if words[index] not in VERB_OBJECTS:
                return False
    return True


def _parse_right(tape: str, slots: tuple[str, ...], cap: int = 32):
    @lru_cache(maxsize=None)
    def rec(offset: int, index: int):
        if index == len(slots):
            return ((),) if offset == len(tape) else ()
        out = []
        slot = slots[index]
        for word in LEXICON[slot]:
            if tape.startswith(word, offset):
                for tail in rec(offset + len(word), index + 1):
                    candidate = (word,) + tail
                    if _graph_valid(candidate, slots):
                        out.append(candidate)
                        if len(out) >= cap:
                            return tuple(out)
        return tuple(out)
    return rec(0, 0)


def _iter_left(slots: tuple[str, ...], max_rows: int = 80_000):
    rows = [()]
    emitted = 0
    for slot in slots:
        rows = [prefix + (word,) for prefix in rows for word in LEXICON[slot]
                if len(_tape(prefix + (word,))) <= 48]
        if len(rows) > max_rows * 2:
            rows = rows[: max_rows * 2]
    for row in rows:
        if emitted >= max_rows:
            break
        if 18 <= len(_tape(row)) <= 48 and _graph_valid(row, slots):
            emitted += 1
            yield row


def _audit(left: tuple[str, ...], right: tuple[str, ...], operation: dict) -> dict:
    rendered = " ".join(left).capitalize() + "; " + " ".join(right) + "."
    tape = normalize_letters(rendered)
    checks = mechanical_admission_checks(rendered, min_letters=39, max_letters=120)
    return {
        "rendered": rendered,
        "letters": len(tape),
        "normalized_letters": tape,
        "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest(),
        "independent_ascii_exact": bool(tape) and tape == tape[::-1],
        "independent_two_pointer": all(tape[i] == tape[-1 - i] for i in range(len(tape) // 2)),
        "left": list(left), "right": list(right),
        "left_slots": list(operation["left_slots"]),
        "right_slots": list(operation["right_slots"]),
        "seed_content_overlap": sorted(_content(left + right) & {"aide", "rips", "nine", "memos", "some", "men", "inspire", "diana"}),
        "central_admission": checks,
        "mechanically_admitted": all(checks.values()),
        "reader_status": "not_run; exactness and diagnostics do not certify readability",
        "provenance": operation,
    }


def run() -> dict:
    rows: list[dict] = []
    stats = Counter()
    seen: set[str] = set()
    for left_slots in SHAPES:
        for left in _iter_left(left_slots):
            stats["left_clauses"] += 1
            reverse_tape = _tape(left)[::-1]
            for right_slots in SHAPES:
                rights = _parse_right(reverse_tape, right_slots)
                if not rights:
                    stats["residual_failures"] += 1
                    continue
                stats["residual_closures"] += len(rights)
                for right in rights:
                    if _content(left) & _content(right):
                        stats["content_disjoint_rejections"] += 1
                        continue
                    rendered_key = _tape(left) + _tape(right)
                    if rendered_key in seen:
                        continue
                    seen.add(rendered_key)
                    op = {
                        "operator": "typed_lexical_graph_residual",
                        "left_slots": left_slots,
                        "right_slots": right_slots,
                        "features": ["number", "tense", "determiner", "transitivity"],
                        "source_text_copied": False,
                    }
                    row = _audit(left, right, op)
                    rows.append(row)
                    stats["candidates"] += 1
                    if row["mechanically_admitted"]:
                        stats["mechanically_admitted"] += 1
    rows.sort(key=lambda row: (row["mechanically_admitted"], row["letters"]), reverse=True)
    return {
        "status": "complete_typed_lexical_graph_residual_no_reader_promotion",
        "config": {"shape_count": len(SHAPES), "minimum_letters": 39,
                    "independent_residual_parse": True, "content_disjoint": True,
                    "catalogue_text": False},
        "stats": dict(stats),
        "admitted": [row for row in rows if row["mechanically_admitted"]],
        "near_misses": rows[:100],
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "inventory": "hand-authored typed lexical graph", "reader_study": False},
        "next_operator": "Add attested subject–verb/object edges and tense/number variants to the residual-state frontier; do not widen isolated word lists without graph edges.",
        "reader_gate": "No output is human evidence; any future candidate requires randomized blinded intact-prose and shuffled controls.",
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
