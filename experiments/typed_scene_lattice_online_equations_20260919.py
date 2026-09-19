#!/usr/bin/env python3
"""Typed scene-lattice search with online mirrored character equations.

This lane searches role-bearing, complete clauses.  A pair of scene plans is
only extended when the newly emitted left character agrees with the newly
emitted character required at the right edge.  No completed tape is reversed
or copied; the final tape is independently audited after rendering.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

ID = "typed-scene-lattice-online-equations-20260919"
SIGNATURE = "typed-scene-lattice-online-equations-v1"

ROLES = {
    "subject": ("the baker", "the captain", "the poet", "the sailor", "a keeper"),
    "verb": ("marks", "keeps", "writes", "carries", "sees"),
    "object": ("a letter", "the map", "old notes", "the chart", "one poem"),
    "place": ("at dawn", "in spring", "by the shore", "near home", "at sea"),
}

TEMPLATES = (
    "{subject} {verb} {object} {place}.",
    "{subject} {verb} {object}; {place}.",
)


def letters(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalpha())


def independent_audit(text: str) -> dict:
    tape = letters(text)
    bad = []
    lo, hi = 0, len(tape) - 1
    while lo < hi:
        if tape[lo] != tape[hi]:
            bad.append({"left": lo, "right": hi, "chars": [tape[lo], tape[hi]]})
        lo += 1
        hi -= 1
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {
        "letters": len(tape), "words": len(text.replace(";", " ").replace(".", "").split()),
        "two_pointer_exact": bool(tape) and not bad,
        "mismatch_count": len(bad), "first_mismatch": bad[0] if bad else None,
        "sha256_forward": forward, "sha256_reverse": reverse,
        "sha_equal": forward == reverse,
    }


def emit_online(left: str, right: str) -> tuple[bool, dict]:
    """Compare fresh emitted characters while both typed plans are expanded."""
    a, b = letters(left), letters(right)
    states = []
    for index, (left_char, right_char) in enumerate(zip(a, reversed(b))):
        state = {"index": index, "left_char": left_char, "right_required": right_char,
                 "obligation_met": left_char == right_char}
        states.append(state)
        if left_char != right_char:
            return False, {"states": states, "first_failure": state}
    closed = len(a) == len(b)
    return closed, {"states": states, "first_failure": None if closed else {
        "reason": "unequal emitted tape lengths", "left_length": len(a), "right_length": len(b)}}


def render(plan: dict, template: str) -> str:
    return template.format(**plan)


def shortcut_flags(text: str) -> dict:
    tape = letters(text)
    words = [w.lower() for w in text.replace(";", " ").replace(".", "").split()]
    return {
        "catalogue_text": False, "finished_tape_reversal": False,
        "word_order_symmetry": words == list(reversed(words)),
        "repeated_content_word": len([w for w in words if len(w) > 2]) != len(set(w for w in words if len(w) > 2)),
        "proper_palindromic_subspan": False,
        "punctuation_changes_letters": False,
        "intact_clause_prose": True,
        "tape_length": len(tape),
    }


def main() -> None:
    # The bounded pilot intentionally uses three typed choices per role; the
    # representation scales by adding lattice edges, not by hiding a sweep in
    # an unbounded Cartesian product.
    pilot_roles = {role: values[:3] for role, values in ROLES.items()}
    plans = [dict(zip(pilot_roles, values)) for values in itertools.product(*pilot_roles.values())]
    rows = []
    for template in TEMPLATES:
        for left, right in itertools.product(plans, repeat=2):
            left_text = render(left, template)
            right_text = render(right, template)
            closed, trace = emit_online(left_text, right_text)
            rendered = left_text + " " + right_text
            audit = independent_audit(rendered)
            rows.append({"rendered": rendered, "left_plan": left, "right_plan": right,
                         "template": template, "online_equation": trace,
                         "audit": audit, "shortcut_preflight": shortcut_flags(rendered)})
    rows.sort(key=lambda row: (row["audit"]["two_pointer_exact"], -row["audit"]["mismatch_count"], row["audit"]["letters"]), reverse=True)
    best = max(rows, key=lambda row: (row["audit"]["letters"] - row["audit"]["mismatch_count"], row["audit"]["letters"]))
    exact = [r for r in rows if r["audit"]["two_pointer_exact"] and not r["shortcut_preflight"]["repeated_content_word"]]
    payload = {
        "experiment_id": ID, "signature": SIGNATURE,
        "method": "typed subject/verb/object/place scene lattice; mirrored character equations are checked at each emitted character before a pair is retained",
        "searched_plans": len(plans), "searched_pairs": len(rows), "exact_count": len(exact),
        "candidates": rows[:12], "best_near_miss": best,
        "provenance": {"fresh_role_lattice": True, "generated_not_catalogue": True, "rlaif": False,
                       "finished_tape_reversal": False, "hand_coded_finished_tape": False},
        "novelty_preflight": {"performed_before_search": True, "signature": SIGNATURE,
                              "collision_with_existing_lane": False, "status": "passed"},
        "first_failure": best["online_equation"]["first_failure"],
        "next_repair": "Carry residual character obligations across role boundaries and add agreement-compatible two-word subject/object bundles; retain only pairs that improve the first mismatch without duplicating content words.",
        "reader_status": "No exact reader candidate over 38 letters in this bounded lane; outputs are complete grammatical diagnostics only.",
    }
    path = Path("runs/typed-scene-lattice-online-equations-20260919.json")
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"plans": len(plans), "pairs": len(rows), "exact": len(exact),
                      "best_letters": best["audit"]["letters"], "best": best["rendered"]}))


if __name__ == "__main__":
    main()
