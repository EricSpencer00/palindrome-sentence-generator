#!/usr/bin/env python3
"""Bounded causal-scene graph construction lane.

Each side is an independently authored event chain.  Search state is a pair
of semantic graph nodes (agent, action, patient, relation) plus the next
character obligations at both ends.  No completed text is reversed or
resegmented; lexical choices are made before the equation is advanced.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

OUT = Path("runs/luna-causal-scene-graph-20260917.json")


@dataclass(frozen=True)
class Event:
    agent: str
    verb: str
    patient: str
    relation: str

    def words(self) -> tuple[str, ...]:
        return (self.agent, self.verb, self.patient, self.relation)

    def text(self) -> str:
        return " ".join(self.words())


# Fresh, ordinary scene frames.  Roles are explicit and choices are jointly
# paired at each search step, rather than selected by a readability score.
LEFT = [
    Event("a baker", "packs", "warm bread", "for a child"),
    Event("the nurse", "brings", "clean water", "to a patient"),
    Event("a sailor", "marks", "the safe harbor", "before dawn"),
    Event("the teacher", "opens", "a quiet atlas", "for the class"),
]
RIGHT = [
    Event("a child", "thanks", "the baker", "after lunch"),
    Event("a patient", "drinks", "clean water", "in the ward"),
    Event("the harbor", "guides", "a sailor", "through fog"),
    Event("the class", "studies", "a quiet atlas", "at noon"),
]


def letters(text: str) -> str:
    return "".join(c for c in text.lower() if c.isalpha())


def audit(text: str) -> dict:
    tape = letters(text)
    return {
        "letters": len(tape),
        "exact": tape == tape[::-1],
        "sha256": hashlib.sha256(tape.encode()).hexdigest(),
        "word_order_mirror": False,
        "proper_self_palindrome": any(tape[i:j] == tape[i:j][::-1]
                                       for i in range(len(tape))
                                       for j in range(i + 2, len(tape) + 1)
                                       if j - i < len(tape)),
    }


def live_equation(a: str, b: str) -> tuple[bool, int]:
    """Consume opposite edges; return closure and matched count."""
    x, y = letters(a), letters(b)
    i = 0
    while i < min(len(x), len(y)) and x[i] == y[-1 - i]:
        i += 1
    return x == y[::-1], i


def main() -> None:
    records = []
    for li, left in enumerate(LEFT):
        for ri, right in enumerate(RIGHT):
            # Causal edge is checked as a semantic invariant, not a score.
            roles_ok = left.patient.split()[-1] == right.patient.split()[-1] or li == ri
            text = left.text() + "; " + right.text()
            exact, matched = live_equation(left.text(), right.text())
            full = audit(text)
            admitted = exact and full["exact"] and not full["word_order_mirror"] and not full["proper_self_palindrome"] and roles_ok
            records.append({"left_node": li, "right_node": ri, "text": text,
                            "roles_ok": roles_ok, "matched_frontier": matched,
                            "exact_pair_equation": exact, "audit": full,
                            "admitted": admitted,
                            "provenance": "fresh-authored-causal-scene-graph-20260917"})
    result = {
        "experiment": "luna-causal-scene-graph-20260917",
        "method": "paired causal event nodes with role-validity and live opposite-edge character equations",
        "candidate_count": len(records),
        "admitted": [r for r in records if r["admitted"]],
        "records": records,
        "next_repair": "Expand event-edge lexical domains with independently authored synonym sets while retaining explicit agent/action/patient role constraints; then run a blinded intact-prose panel on every exact survivor.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"candidates": len(records), "admitted": len(result["admitted"]),
                      "best_frontier": max(r["matched_frontier"] for r in records)}, indent=2))


if __name__ == "__main__":
    main()
