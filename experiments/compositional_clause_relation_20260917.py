"""Variable-length compositional grammar relation search.

This lane treats a sentence as a sequence of authored, intact clauses.  Clause
choices are made from the outside inward while their character obligations are
matched immediately.  The number of clauses is a parameter (not a fixed
palindromic scaffold), and semantic roles are checked before a path is emitted.
It is deliberately independent of any known-palindrome catalogue.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from itertools import product

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "compositional-clause-relation-20260917.json"
EXPERIMENT_ID = "compositional-clause-relation-20260917"


@dataclass(frozen=True)
class Clause:
    text: str
    role: str
    subject: str
    tense: str


# Intact, independently authored clauses.  These are not reversed or split
# into character fragments; the solver may only select or reject a clause.
CLAUSES = (
    Clause("a calm sailor reads a map", "observation", "sailor", "present"),
    Clause("the young baker mends a boat", "action", "baker", "present"),
    Clause("a kind artist writes a note", "action", "artist", "present"),
    Clause("the quiet child holds a stone", "observation", "child", "present"),
    Clause("a careful guide marks the path", "action", "guide", "present"),
    Clause("the old poet tells a story", "speech", "poet", "present"),
    Clause("a bright student learns the rule", "learning", "student", "present"),
    Clause("the patient farmer tends the grain", "action", "farmer", "present"),
)


def letters(text: str) -> str:
    return "".join(c for c in text.lower() if c.isalpha())


def exact_audit(text: str) -> dict:
    tape = letters(text)
    mismatches = [(i, len(tape)-1-i) for i in range(len(tape)//2)
                  if tape[i] != tape[-1-i]]
    return {
        "letters": len(tape), "exact": bool(tape) and not mismatches,
        "first_mismatch": mismatches[0] if mismatches else None,
        "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest(),
    }


def semantic_ok(chosen: tuple[Clause, ...]) -> bool:
    """Require a coherent discourse: one subject is not repeated and roles vary."""
    if not chosen or len({c.subject for c in chosen}) != len(chosen):
        return False
    return len({c.role for c in chosen}) >= min(2, len(chosen))


def match_outside_in(left: str, right: str) -> tuple[int, int] | None:
    """Return consumed character counts, or None on the first live mismatch."""
    a, b = letters(left), letters(right)
    k = min(len(a), len(b))
    for i in range(k):
        if a[i] != b[-1-i]:
            return None
    return k, k


def search(max_clauses: int = 6, budget: int = 100_000) -> dict:
    states = 0
    rejected = []
    exact = []
    frontiers = []
    # Clause count is variable and each complete candidate is assembled from
    # intact clauses.  Matching is performed after every outside pair is added.
    for n in range(1, max_clauses + 1):
        for indexes in product(range(len(CLAUSES)), repeat=n):
            states += 1
            if states > budget:
                return {"states": states, "budget_exhausted": True,
                        "exact": exact, "rejected": rejected,
                        "frontiers": frontiers}
            chosen = tuple(CLAUSES[i] for i in indexes)
            if not semantic_ok(chosen):
                continue
            # Every outer pair is checked as soon as it exists.  The middle
            # clause is allowed to have an internally palindromic residual,
            # but is never manufactured by reversal or character splicing.
            okay = True
            for i in range(n // 2):
                if match_outside_in(chosen[i].text, chosen[-1-i].text) is None:
                    okay = False
                    break
            if not okay:
                continue
            rendered = " ".join(c.text for c in chosen)
            audit = exact_audit(rendered)
            record = {"text": rendered, "provenance": [c.__dict__ for c in chosen],
                      "audit": audit, "semantic_ok": True,
                      "novelty_preflight": {"borrowed_catalogue": False,
                                            "reversed_units": False,
                                            "repeated_units": False}}
            if audit["exact"]:
                exact.append(record)
            elif len(letters(rendered)) >= 30:
                rejected.append(record)
        frontiers.append({"clauses": n, "states": states,
                          "exact_count": len(exact)})
    return {"states": states, "budget_exhausted": False, "exact": exact,
            "rejected": rejected[:20], "frontiers": frontiers,
            "next_repair": "Add typed clause alternatives at the first live mismatch; preserve role, tense, and intact boundaries."}


def main() -> None:
    result = search()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"experiment_id": EXPERIMENT_ID,
                               "method": "variable-length intact-clause relation search",
                               "result": result}, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
