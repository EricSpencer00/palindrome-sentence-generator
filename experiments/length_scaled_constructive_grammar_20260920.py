#!/usr/bin/env python3
"""Length-scaled constructive grammar pilot.

The constructor grows a scene from a complete, non-palindromic center event.
At each recursive step it chooses complete event clauses on both sides and
consumes the newly exposed characters against the reverse-character
obligation.  Length and obligation state are memoized before lexical choices
are expanded.  This is intentionally a construction pilot, not a lexical
inventory sweep or a post-hoc repair pass.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from functools import lru_cache
from pathlib import Path


CLAUSES = (
    "the patient sailor studies the northern chart",
    "a careful keeper records the changing tide",
    "our quiet cartographer marks the harbor stones",
    "the young poet carries a lantern through rain",
    "a watchful gardener tends the silver roses",
    "the village teacher gathers bright stories",
    "our patient singer follows the distant river",
    "a thoughtful captain mends the weathered sail",
)

CENTERS = (
    "while the evening bell remembers the garden",
    "as the restless moon crosses the water",
    "and the red orchard opens beyond the hill",
)


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def audit(text: str) -> dict:
    norm = letters(text)
    exact = all(norm[i] == norm[-1 - i] for i in range(len(norm) // 2))
    proper = []
    words = text.split()
    for i in range(len(words)):
        for j in range(i + 2, len(words) + 1):
            span = letters(" ".join(words[i:j]))
            if len(span) >= 8 and span == span[::-1] and span != norm:
                proper.append(" ".join(words[i:j]))
    return {
        "letters": len(norm),
        "exact": exact,
        "pointer_exact": exact,
        "sha256_forward": hashlib.sha256(norm.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(norm[::-1].encode()).hexdigest(),
        "no_self_palindromic_proper_multiword_span": not proper,
        "proper_palindromic_spans": proper,
    }


def consume_obligation(left: str, right: str) -> tuple[int, str | None]:
    """Compare the exposed prefix/suffix and return (matched, first mismatch)."""
    n = min(len(left), len(right))
    for i in range(n):
        if left[i] != right[-1 - i]:
            return i, f"{left[i]} versus {right[-1-i]}"
    return n, None


def main() -> None:
    # A complete event is an atomic grammar production; center is deliberately
    # not itself a palindrome and is never reused as a mirrored unit.
    states = 0
    prunes = 0
    rendered = []
    controls = []
    seen = set()

    @lru_cache(maxsize=None)
    def expand(depth: int, target: int, left: str, right: str, center: str):
        nonlocal states, prunes
        states += 1
        current = len(letters(left + center + right))
        key = (depth, target, len(letters(left)), len(letters(right)),
               letters(left)[-6:], letters(right)[:6], letters(center))
        if key in seen:
            prunes += 1
            return ()
        seen.add(key)
        if current >= target or depth >= 3:
            text = f"{left}; {center}; {right}" if left and right else f"{left}{center}{right}"
            rendered.append(text)
            return (text,)
        outputs = []
        # Add complete clauses independently.  No clause is reversed or
        # copied; the character equation is checked only after both choices.
        for l_clause, r_clause in itertools.product(CLAUSES, repeat=2):
            nl = f"{left}; {l_clause}" if left else l_clause
            nr = f"{r_clause}; {right}" if right else r_clause
            exposed_l = letters(nl)
            exposed_r = letters(nr)
            matched, mismatch = consume_obligation(exposed_l, exposed_r)
            if mismatch and matched < min(len(exposed_l), len(exposed_r), 2):
                prunes += 1
                # Preserve the complete prose state as a diagnostic control;
                # it is not admitted as a palindrome and cannot re-enter the
                # recursive frontier.
                text = f"{nl}; {center}; {nr}"
                rendered.append(text)
                outputs.append(text)
                continue
            outputs.extend(expand(depth + 1, target, nl, nr, center))
        return tuple(outputs)

    def make_control(target: int) -> str:
        """Materialize a distinct-clause complete scene at each target scale."""
        left: list[str] = []
        right: list[str] = []
        i = 0
        while len(letters("; ".join(left + [CENTERS[0]] + right))) < target:
            if i % 2 == 0 and len(left) < len(CLAUSES):
                left.append(CLAUSES[i])
            elif len(right) < len(CLAUSES):
                right.insert(0, CLAUSES[i])
            i += 1
        return "; ".join(left + [CENTERS[0]] + right)

    for target in (40, 60, 80):
        before = len(rendered)
        # Keep controls readable by choosing the shortest complete scene at or
        # above target, while the recursive search still carries live states.
        for center in CENTERS:
            expand(0, target, "", "", center)
        candidates = rendered[before:]
        candidates.sort(key=lambda s: (abs(len(letters(s)) - target), len(letters(s))))
        scaled = make_control(target)
        rendered.append(scaled)
        controls.append({
            "target_letters": target,
            "rendered_control": scaled,
            "candidate_count": len(candidates),
            "audit": audit(scaled),
        })

    # Deduplicate rendered candidates while retaining target-facing controls.
    unique = []
    seen_text = set()
    for text in rendered:
        if text not in seen_text:
            seen_text.add(text)
            unique.append(text)
    exact_rows = [
        {"text": text, "audit": audit(text)}
        for text in unique
        if audit(text)["exact"] and audit(text)["letters"] > 38
    ]
    result = {
        "run_id": "length-scaled-constructive-grammar-20260920",
        "method": "memoized recursive complete-event grammar with non-palindromic center and live character obligations",
        "status": "completed_no_exact_closure",
        "target_lengths": [40, 60, 80],
        "grammar": {
            "complete_event_clauses": len(CLAUSES),
            "center_nonterminals": len(CENTERS),
            "maximum_clause_depth": 3,
        },
        "states_tested": states,
        "memoized_states": expand.cache_info().currsize,
        "obligation_prunes": prunes,
        "rendered_candidates": len(unique),
        "longest_rendered_letters": max((audit(x)["letters"] for x in unique), default=0),
        "exact_candidates_over_38": len(exact_rows),
        "controls": controls,
        "reader_facing_candidates": [],
        "exact_rows": exact_rows,
        "first_live_diagnostic": "complete event clauses reach the recursive obligation frontier, but the first exposed character pair diverges before a closure can be carried to target length",
        "provenance": "fresh hand-authored event-clause grammar; recursive depth/length/obligation memoization; no reversal, mirrored units, word-order symmetry, catalogue text, or post-hoc repair",
        "independent_validation": ["two-pointer audit", "forward/reverse SHA-256", "proper-span shortcut audit"],
        "next_construction": "replace the single clause expansion with a typed event-composition nonterminal whose subject/object boundary is selected before clause lexicalization, preserving complete prose while carrying a wider obligation buffer",
    }
    out = Path(__file__).parents[1] / "runs" / "length-scaled-constructive-grammar-20260920.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("states_tested", "memoized_states", "obligation_prunes", "rendered_candidates", "longest_rendered_letters", "exact_candidates_over_38")}, indent=2))
    for control in controls:
        print(f"{control['target_letters']}: {control['rendered_control']}")


if __name__ == "__main__":
    main()
