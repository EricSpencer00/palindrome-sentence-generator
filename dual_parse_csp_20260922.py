"""Exact two-grammar CSP with admission constraints carried during search.

The operator treats the two slot plans as variables in one character-level
constraint problem.  A choice is committed only when its exposed characters
match the live residual; lexical uniqueness and central admissibility are
checked on every prefix, so inadmissible completed prose is never enumerated.
"""
from __future__ import annotations

import json
from pathlib import Path
from llm_palindrome.admission import mechanical_admission_checks, tokenize
from llm_palindrome.dual_parse import word_residual_search, letter_tape

ID = "dual-parse-csp-operator-20260922"


def _prefix_ok(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    """Fail closed on irreversible construction shortcuts while growing."""
    words = tuple(tokenize(" ".join(left + right)))
    # Content-word repetition and copied adjacent units cannot be repaired by
    # adding characters later; reject before the state enters the frontier.
    content = [letter_tape(w) for w in words if len(letter_tape(w)) > 2]
    if len(content) != len(set(content)):
        return False
    for i in range(len(words) - 1):
        for j in range(i + 2, len(words) - 1):
            if words[i:i + 2] == words[j:j + 2]:
                return False
    return True


def solve(left_slots, right_slots, *, max_states=20_000, max_results=20):
    """Run the simultaneous character/grammar CSP and admit only centers."""
    result = word_residual_search(
        tuple(left_slots), tuple(right_slots), max_states=max_states,
        max_results=max_results, allow_partial=_prefix_ok,
        reject_intermediate_closure=True,
    )
    admitted, rejected = [], 0
    for row in result["results"]:
        checks = mechanical_admission_checks(row["rendered"], min_letters=39, max_letters=200)
        if all(checks.values()):
            admitted.append({**row, "mechanical_admission": checks})
        else:
            rejected += 1
    return {**result, "results": admitted, "admission_rejections": rejected,
            "operator": ID, "frontier_bounded": bool(result["cap_reached"] or result["dead_frontiers"])}


LEFT = (("A:determiner", ("an",)), ("A:agent", ("aide",)),
        ("B:verb", ("rips",)), ("B:quantity", ("nine",)),
        ("B:object", ("memos",)))
RIGHT = (("B-prime:response", ("some",)), ("B-prime:agent", ("men",)),
         ("B-prime:verb", ("inspire",)), ("A-prime:patient", ("Diana",)))

if __name__ == "__main__":
    out = solve(LEFT, RIGHT)
    path = Path(__file__).resolve().parent / "runs" / f"{ID}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
