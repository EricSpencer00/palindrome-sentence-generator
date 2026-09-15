"""Breadth-first search for independently authored typed clause closures.

The left and right clauses are generated from different role templates.  A
right-side residual parser consumes the reverse of the left clause's character
tape, so word boundaries are solved independently.  Whole-word reflections,
repeated units, catalogue material, and seed-derived text are rejected.  A
programmatic match is only a candidate for later human review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, deque
from functools import lru_cache
from pathlib import Path

from wordfreq import zipf_frequency

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

PERSONS = tuple("agent artist author child clerk dealer doctor father friend girl guard king man maker mother nurse painter parent person pilot poet queen reader singer son teacher woman worker writer".split())
THINGS = tuple("apple bag book boat bottle cake car cat chair coin cup dog door drawer drum dessert desserts diaper flower game gift hat house key lamp letter map meal mirror music note paper phone plant poem pot radio reward ring room rope ship shoe song stone story table ticket tool train tree vase watch water wheel".split())
VERBS = tuple("asks buys calls cooks draws drew finds gives hears holds keeps likes made makes needs opens paid reads repaid sees sends shows stressed takes tells uses wants writes".split())
DETS = ("a", "an", "the")
PRONOUNS = ("i", "he", "she", "we", "you", "they")
SEED_CONTENT = frozenset("aide rips nine memos some men inspire diana".split())

# Distinct clause roles on the two sides.  No template is the reverse of
# another at the token level, and the two independent parses must use at least
# two different content words.
LEFT_TEMPLATES = (
    ("det_person_verb_det_thing", ("DET", "PERSON", "VERB", "DET", "THING")),
    ("person_verb_det_thing", ("PERSON", "VERB", "DET", "THING")),
    ("det_thing_verb_person", ("DET", "THING", "VERB", "PERSON")),
)
RIGHT_TEMPLATES = (
    ("pron_verb_det_thing", ("PRON", "VERB", "DET", "THING")),
    ("det_person_verb_thing", ("DET", "PERSON", "VERB", "THING")),
    ("thing_verb_det_person", ("THING", "VERB", "DET", "PERSON")),
    ("person_verb_thing", ("PERSON", "VERB", "THING")),
)

ROLE_WORDS = {"PERSON": PERSONS, "THING": THINGS, "VERB": VERBS, "DET": DETS, "PRON": PRONOUNS}


def _tape(words: tuple[str, ...]) -> str:
    return "".join(words)


def _is_token_mirror(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    return bool(left) and tuple(word[::-1] for word in reversed(left)) == right


def _iter_left(slots: tuple[str, ...], max_rows: int = 60_000):
    """Deterministic breadth-first lexical expansion with a length bound."""
    queue = deque([()])
    emitted = 0
    while queue and emitted < max_rows:
        prefix = queue.popleft()
        if len(prefix) == len(slots):
            if 18 <= len(_tape(prefix)) <= 90:
                emitted += 1
                yield prefix
            continue
        role = slots[len(prefix)]
        for word in ROLE_WORDS[role]:
            child = prefix + (word,)
            if len(_tape(child)) <= 45:
                queue.append(child)


def _parse_right(tape: str, slots: tuple[str, ...], cap: int = 50):
    """Parse a fixed residual tape into a different typed clause template."""
    @lru_cache(maxsize=None)
    def rec(offset: int, slot: int):
        if slot == len(slots):
            return ((),) if offset == len(tape) else ()
        out = []
        for word in ROLE_WORDS[slots[slot]]:
            if tape.startswith(word, offset):
                for tail in rec(offset + len(word), slot + 1):
                    out.append((word,) + tail)
                    if len(out) >= cap:
                        return tuple(out)
        return tuple(out)
    return rec(0, 0)


def _audit(left: tuple[str, ...], right: tuple[str, ...], operation: dict) -> dict:
    text = " ".join(left).capitalize() + "; " + " ".join(right) + "."
    tape = normalize_letters(text)
    checks = mechanical_admission_checks(text, min_letters=39, max_letters=100)
    return {
        "rendered": text,
        "letters": len(tape),
        "normalized_letters": tape,
        "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest(),
        "independent_ascii_exact": bool(tape) and tape == tape[::-1],
        "independent_two_pointer": all(tape[i] == tape[-1 - i] for i in range(len(tape) // 2)),
        "left_clause": list(left),
        "right_clause": list(right),
        "left_role_trace": list(operation["left_slots"]),
        "right_role_trace": list(operation["right_slots"]),
        "word_order_shortcut": _is_token_mirror(left, right),
        "seed_content_overlap": sorted(set(left + right) & SEED_CONTENT),
        "central_admission": checks,
        "mechanically_admitted": all(checks.values()) and not _is_token_mirror(left, right),
        "reader_status": "not_run; exactness and diagnostics do not certify readability",
        "provenance": operation,
    }


def run() -> dict:
    rows = []
    seen = set()
    stats = Counter()
    failures = []
    for left_name, left_slots in LEFT_TEMPLATES:
        for left in _iter_left(left_slots):
            stats["left_clauses"] += 1
            ltape = _tape(left)
            if len(ltape) < 18:
                continue
            reverse_tape = ltape[::-1]
            for right_name, right_slots in RIGHT_TEMPLATES:
                rights = _parse_right(reverse_tape, right_slots)
                if not rights:
                    stats["residual_failures"] += 1
                    continue
                stats["residual_closures"] += len(rights)
                for right in rights:
                    if len(ltape) + len(_tape(right)) < 39:
                        continue
                    if set(left + right) & SEED_CONTENT:
                        stats["seed_overlap_rejections"] += 1
                        continue
                    if _is_token_mirror(left, right):
                        stats["word_order_rejections"] += 1
                        continue
                    text_key = _tape(left) + _tape(right)
                    if text_key in seen:
                        continue
                    seen.add(text_key)
                    operation = {
                        "operator": "breadth_first_independent_typed_semordnilap",
                        "left_template": left_name,
                        "right_template": right_name,
                        "left_slots": left_slots,
                        "right_slots": right_slots,
                        "residual": reverse_tape,
                        "source_text_copied": False,
                    }
                    row = _audit(left, right, operation)
                    rows.append(row)
                    stats["candidates"] += 1
                    if row["mechanically_admitted"]:
                        stats["mechanically_admitted"] += 1
    rows.sort(key=lambda row: (row["mechanically_admitted"], row["letters"]), reverse=True)
    return {
        "status": "complete_breadth_first_typed_semordnilap_no_reader_promotion",
        "config": {
            "role_inventory": "explicit PERSON/THING/transitive VERB pools",
            "left_templates": [name for name, _ in LEFT_TEMPLATES],
            "right_templates": [name for name, _ in RIGHT_TEMPLATES],
            "max_left_clauses_per_template": 60_000,
            "minimum_letters": 39,
            "independent_residual_parse": True,
            "word_order_or_seed_shortcuts": "rejected",
        },
        "stats": dict(stats),
        "admitted": [row for row in rows if row["mechanically_admitted"]],
        "near_misses": rows[:100],
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "roles_are_hand_authored": True,
            "catalogue_text": False,
        },
        "next_operator": (
            "Keep the residual-state BFS but replace isolated role lists with a typed lexical graph: add tense, "
            "number, determiner, and transitivity features, then expand both clauses jointly while scoring only "
            "attested subject-verb/object relations. Preserve independent residual parsing and exact audit."
        ),
        "reader_gate": "No output is human evidence; any future candidate requires blinded intact-prose and shuffled controls.",
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
