"""Object-level scene-program admission diagnostic.

Unlike clause banks, this enumerates a tiny executable scene: typed entities
and two events linked by an object relation.  It renders each complete program
on both sides and then performs an independent character walk over the two
finished tapes.  That is useful as a bounded object-level closure diagnostic,
but it is *not* an online character generator and must not be described as
live pruning.  No finished string is reversed or repaired.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from itertools import permutations, product
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "runs" / "object-scene-program-tape-20260921.json"


def norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.casefold())


def audit(s: str) -> dict:
    tape = norm(s)
    mismatch = next(((i, tape[i], tape[-i - 1]) for i in range(len(tape) // 2)
                     if tape[i] != tape[-i - 1]), None)
    return {"letters": len(tape), "exact": bool(tape) and mismatch is None,
            "first_mismatch": mismatch,
            "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest()}


@dataclass(frozen=True)
class Entity:
    ident: str
    kind: str
    forms: tuple[str, ...]


@dataclass(frozen=True)
class Event:
    ident: str
    subject: str
    verb: tuple[str, ...]
    object: str


ENTITIES = (
    Entity("scout", "person", ("the scout", "a scout")),
    Entity("guard", "person", ("the guard", "a guard")),
    Entity("map", "thing", ("the map", "a map")),
    Entity("gate", "thing", ("the gate", "a gate")),
)
EVENTS = (
    Event("mark", "scout", ("marks", "notes"), "gate"),
    Event("open", "guard", ("opens", "checks"), "map"),
)
ENTITY = {e.ident: e for e in ENTITIES}


def render(order: tuple[Event, ...], forms: tuple[str, ...]) -> str:
    chunks = []
    for event, subject in zip(order, forms):
        chunks.append(f"{subject} {event.verb[0]} {ENTITY[event.object].forms[0]}")
    return "; ".join(chunks) + "."


def post_render_pair(left: str, right: str) -> tuple[bool, int]:
    """Walk two already-rendered tapes and report their common prefix."""
    return post_render_tape_walk(norm(left), norm(right)[::-1])


def post_render_tape_walk(left: str, right_reversed: str) -> tuple[bool, int]:
    """Compare an exposed pair of complete tapes without post-hoc repair."""
    matched = 0
    for x, y in zip(left, right_reversed):
        if x != y:
            return False, matched
        matched += 1
    return len(left) == len(right_reversed), matched


def main() -> None:
    rows, exact = [], []
    states = 0
    # A program is valid only if event objects are typed and distinct; order
    # is a topological choice, not a mirror pairing.
    orders = list(permutations(EVENTS))
    subject_forms = tuple(form for e in ENTITIES if e.kind == "person" for form in e.forms)
    for left_order, right_order in product(orders, repeat=2):
        for left_forms in product(subject_forms, repeat=2):
            for right_forms in product(subject_forms, repeat=2):
                states += 1
                left = render(left_order, left_forms)
                right = render(right_order, right_forms)
                ok, matched = post_render_pair(left, right)
                rendered = f"{left} {right}"
                row = {"rendered": rendered, "left_program": [e.ident for e in left_order],
                       "right_program": [e.ident for e in right_order],
                       "matched_characters": matched, "post_render_exact": ok,
                       "audit": audit(rendered),
                       "provenance": {"source": "fresh hand-authored executable scene program",
                                      "entities": [e.ident for e in ENTITIES],
                                      "events": [e.ident for e in EVENTS],
                                      "topological_orders": True}}
                if ok and row["audit"]["exact"]:
                    exact.append(row)
                elif len(rows) < 12:
                    rows.append(row)
    payload = {"method": "object-scene-program-post-render-admission",
               "search_kind": "complete-program enumeration followed by a post-render character walk",
               "states": states, "controls": rows, "exact": exact,
               "independent_audit": "audit() normalizes letters and compares every mirror position; post_render_tape_walk() is diagnostic only",
               "next_repair": "compile the typed event alternatives into a true forward/reverse NFA before expansion; do not call this complete-product diagnostic online",
               "reader_evidence": {"status": "not_run", "reason": "no exact candidate passed the mechanical gate"}}
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"states": states, "controls": len(rows), "exact": len(exact),
                      "best": max((r["matched_characters"] for r in rows), default=0),
                      "output": str(OUT)}))


if __name__ == "__main__":
    main()
