"""Character-seam indexed search over a small authored Shakespearean lexicon.

The search does not score candidates.  It renders ordinary clause templates,
indexes their normalized tapes, and joins a left clause to the unique residual
right tape required by the palindrome equation.  This keeps readability in
the construction space rather than applying a reward after generation.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

OUT = Path(__file__).parents[1] / "runs" / "seam-lexical-index-20260919.json"

SUBJECTS = ("the king", "the queen", "a poet", "a player", "the bard", "a sailor", "the raven", "a child")
VERBS = ("guards", "praises", "remembers", "follows", "hears", "seeks", "answers", "watches")
OBJECTS = ("the dawn", "a rose", "the moon", "a song", "the truth", "a bell", "the shore", "a dream")
ADJUNCTS = ("at dusk", "by the sea", "in the hall", "under stars", "before dawn", "with calm", "in silence", "at noon")

def letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())

def independent_audit(text: str) -> dict:
    tape = "".join(ch for ch in text.lower() if "a" <= ch <= "z")
    two = all(tape[i] == tape[-1-i] for i in range(len(tape)//2))
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"letters": len(tape), "two_pointer": two, "sha256_forward": forward,
            "sha256_reverse": reverse, "sha256_equal": forward == reverse,
            "exact": two and forward == reverse, "normalized_tape": tape}

def clauses() -> list[str]:
    # These are intact, independently readable clause surfaces, not fragments.
    out = []
    for subject in SUBJECTS:
        for verb in VERBS:
            for obj in OBJECTS:
                for adjunct in ADJUNCTS:
                    out.append(f"{subject} {verb} {obj} {adjunct}")
    return out

@dataclass
class Result:
    rendered: str
    provenance: dict
    audit: dict

def search(min_letters: int = 39, max_letters: int = 90) -> dict:
    bank = clauses()
    # Index complete right-hand clauses by exact tape.  The residual obligation
    # for a left clause is simply reverse(left); no candidate-level LM call.
    index: dict[str, list[str]] = {}
    for phrase in bank:
        index.setdefault(letters(phrase), []).append(phrase)
    hits: list[Result] = []
    checked = 0
    for left in bank:
        residual = letters(left)[::-1]
        for right in index.get(residual, []):
            rendered = left + ". " + right + "."
            audit = independent_audit(rendered)
            checked += 1
            if min_letters <= audit["letters"] <= max_letters and audit["exact"]:
                hits.append(Result(rendered, {"method": "authored_clause_residual_index",
                    "left_clause": left, "right_clause": right,
                    "lexicon_sizes": {"subjects": len(SUBJECTS), "verbs": len(VERBS),
                                      "objects": len(OBJECTS), "adjuncts": len(ADJUNCTS)},
                    "residual_tape": residual}, audit))
    return {"method": "authored_clause_residual_index", "search_space": len(bank),
            "indexed_tapes": len(index), "joined_pairs_checked": checked,
            "length_range": [min_letters, max_letters], "exact_hits": [asdict(x) for x in hits],
            "independent_audit": "ASCII letters, two-pointer comparison, and forward/reverse SHA-256",
            "next_repair": "Add a third clause family and index residual prefixes at word boundaries; retain the same exact join equation."}

if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    result = search()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("search_space", "indexed_tapes", "joined_pairs_checked", "exact_hits")}, indent=2))
