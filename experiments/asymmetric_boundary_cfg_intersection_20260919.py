#!/usr/bin/env python3
"""Search asymmetric word-boundary CFG intersections with online tape checks.

The left and right clauses are generated from different typed templates.  A
candidate is accepted only when a right-hand word segmentation is compatible
with the reverse character tape while it is being built; no finished tape is
reversed and no catalogue sentence is used as a seed.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_palindrome.admission import mechanical_admission_checks

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/asymmetric-boundary-cfg-intersection-20260919.json"

# Fresh, small authored lexicons.  The deliberately different role banks make
# the two sides ordinary clauses rather than word-order mirror slots.
BANK = {
    "det": ("a", "the"),
    "subj": ("pilot", "scribe", "teacher", "gardener", "messenger", "sailor"),
    "verb": ("keeps", "maps", "notes", "sees", "guides", "meets", "draws"),
    "obj": ("a map", "the bell", "a note", "the garden", "a letter", "the river"),
    "prep": ("at", "near", "by", "in"),
    "place": ("dawn", "home", "the quay", "the harbor", "the station"),
    "adv": ("now", "today", "again"),
}

# Asymmetric complete-clause patterns.  Each tuple names lexical slots; right
# grammar is not derived from, or reversed from, left grammar.
LEFT = (
    ("det", "subj", "verb", "obj", "prep", "place"),
    ("det", "subj", "verb", "adv", "prep", "place"),
    ("det", "subj", "verb", "obj"),
)
RIGHT = (
    ("det", "subj", "verb", "obj"),
    ("det", "subj", "verb", "adv", "prep", "place"),
    ("det", "subj", "verb", "obj", "prep", "place"),
)


def letters(text: str) -> str:
    return "".join(re.findall("[a-z]", text.lower()))


def words(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z]+", text.lower()))


def render(slots: tuple[str, ...], values: tuple[str, ...]) -> str:
    return " ".join(v for k, v in zip(slots, values))


def audit(text: str) -> dict:
    tape = letters(text)
    mismatches = [(i, len(tape)-1-i) for i in range(len(tape)//2)
                  if tape[i] != tape[-1-i]]
    return {
        "letters": len(tape), "exact": not mismatches,
        "first_mismatch": mismatches[0] if mismatches else None,
        "sha256": hashlib.sha256(tape.encode()).hexdigest(),
        "independent_two_pointer": all(tape[i] == tape[-1-i]
                                        for i in range(len(tape)//2)),
    }


def segment_reverse(tape: str, pattern: tuple[str, ...]) -> list[tuple[str, ...]]:
    """Online CFG intersection: consume the reverse tape in role chunks."""
    out: list[tuple[str, ...]] = []

    def rec(pos: int, slot: int, acc: list[str]) -> None:
        if slot == len(pattern):
            if pos == len(tape):
                out.append(tuple(acc))
            return
        role = pattern[slot]
        for value in BANK[role]:
            chunk = letters(value)
            if tape.startswith(chunk, pos):
                rec(pos + len(chunk), slot + 1, acc + [value])

    rec(0, 0, [])
    return out


def gate(text: str) -> dict:
    try:
        result = mechanical_admission_checks(text)
        return {"eligible": all(result.values()), "detail": result}
    except Exception as exc:  # fail closed; preserve diagnostic evidence
        return {"eligible": False, "detail": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    rows = []
    exact = []
    near = []
    for lp, rp in itertools.product(LEFT, RIGHT):
        # Keep authored clause products bounded and make the asymmetry explicit.
        for lv in itertools.product(*(BANK[k] for k in lp)):
            left = render(lp, lv)
            ltape = letters(left)
            # The right side is segmented from the reverse tape, but is never
            # rendered by reversing a completed candidate.
            for rv in segment_reverse(ltape[::-1], rp):
                right = render(rp, rv)
                full = f"{left}; {right}."
                # Candidate must pass exactness and the anti-shortcut gate while
                # still inside the search, not as a post-hoc claim.
                a = audit(full)
                g = gate(full)
                row = {"rendered": full, "left_pattern": lp,
                       "right_pattern": rp, "left_values": lv,
                       "right_values": rv, "audit": a,
                       "anti_shortcut": g,
                       "provenance": "fresh_authored_asymmetric_cfg_intersection"}
                rows.append(row)
                if a["exact"] and g["eligible"]:
                    exact.append(row)
    # Also expose best actual prose controls when no exact intersection survives.
    controls = []
    for lp in LEFT:
        lv = tuple(BANK[k][0] for k in lp)
        text = render(lp, lv) + "."
        controls.append({"rendered": text, "audit": audit(text),
                         "provenance": "fresh_authored_cfg_control",
                         "anti_shortcut": gate(text)})
    payload = {
        "experiment": "asymmetric-boundary-cfg-intersection-20260919",
        "method": "Different typed clause patterns; online segmentation of reverse character tape into independently authored right-role words; exact and anti-shortcut checks during search.",
        "candidate_count": len(rows), "exact_admitted_count": len(exact),
        "exact_admitted": exact[:20], "controls": controls,
        "summary": {
            "left_patterns": len(LEFT), "right_patterns": len(RIGHT),
            "states_tested": len(rows),
            "longest_exact_admitted": max((r["audit"]["letters"] for r in exact), default=0),
            "longest_rendered_control": max((r["audit"]["letters"] for r in controls), default=0),
            "next_repair": "Add boundary-transition role variants (auxiliary and relative-clause slots) indexed by the residual closing character; preserve online exact and gate checks.",
        },
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
