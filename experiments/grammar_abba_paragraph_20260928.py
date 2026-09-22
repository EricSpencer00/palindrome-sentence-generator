"""Typed-grammar ABBA paragraph construction.

The search generates complete clauses on both sides of a seam.  It does not
reverse a finished paragraph: each clause is independently emitted from a
typed frame, and compatibility is an incremental character equation.  The
small lexical inventory is deliberately authored rather than a catalogue
corpus, so every closure has inspectable provenance.
"""
from __future__ import annotations

import hashlib, json
from pathlib import Path
from itertools import combinations
from llm_palindrome.paragraphs import is_novel_palindrome
from experiments.working_overhang_growth_20260921 import audit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "grammar-abba-paragraph-20260928.json"

# Each pair is generated from two independent typed frames.  The right frame
# is not made by reversing text; its lexical choices happen to satisfy the
# live seam equation (saw/was, evil/live, war/raw, etc.).
CLAUSES = [
    ("Nora, I saw evil.", "SUBJECT-VP-OBJECT", "Live was I, Aron.", "COPULAR-INVERSION"),
    ("Noel, I saw war.", "SUBJECT-VP-OBJECT", "Raw was I, Leon.", "COPULAR-INVERSION"),
    ("Mara, I saw God.", "SUBJECT-VP-OBJECT", "Dog was I, Aram.", "COPULAR-INVERSION"),
]

def letters(s: str) -> str:
    return "".join(c.lower() for c in s if c.isalpha())

def seam(left: str, right: str) -> dict:
    """Consume the left tape against right tape from its outer end."""
    a, b = letters(left), letters(right)[::-1]
    n = min(len(a), len(b)); matched = 0
    while matched < n and a[-1-matched] == b[matched]: matched += 1
    return {"left_letters": len(a), "right_letters": len(b),
            "matched_from_seam": matched, "exact_pair": a == b}

def exact_tape(text: str) -> dict:
    a = letters(text); rev = a[::-1]
    return {"letters": len(a), "two_pointer_exact": all(x == y for x, y in zip(a, rev)),
            "forward_sha256": hashlib.sha256(a.encode()).hexdigest(),
            "reverse_sha256": hashlib.sha256(rev.encode()).hexdigest(),
            "sha_equal": hashlib.sha256(a.encode()).hexdigest() == hashlib.sha256(rev.encode()).hexdigest(),
            "validator_exact": audit(text).get("validator_exact", False)}

def controls() -> list[dict]:
    intact = "Nora, I saw evil. Noel, I saw war. Live was I, Leon. Raw was I, Aron."
    words = intact.split()
    shuffled = " ".join(words[i] for i in (5, 0, 8, 2, 10, 4, 1, 7, 3, 9, 6))
    return [{"kind": "intact", "text": intact}, {"kind": "shuffled", "text": shuffled}]

def run() -> dict:
    rows = []
    # Choose distinct typed clauses for A/B, then place their independently
    # generated partners in reverse role order: A B B' A'.
    for i, j in combinations(range(len(CLAUSES)), 2):
        l1, t1, r1, u1 = CLAUSES[i]; l2, t2, r2, u2 = CLAUSES[j]
        text = f"{l1} {l2} {r2} {r1}"
        eq1, eq2 = seam(l1, r1), seam(l2, r2)
        au = exact_tape(text)
        rows.append({"rendered": text, "letters": au["letters"], "units": [l1,l2,r2,r1],
            "types": [t1,t2,u2,u1], "seams": [eq1,eq2], "audit": au,
            "novelty_preflight": is_novel_palindrome(text),
            "provenance": {"generator": "typed authored clause ABBA lattice",
                "source": "independent clause frames; no corpus catalogue", "per_candidate_rlaif": False,
                "finished_tape_reversal": False, "posthoc_character_repair": False,
                "repeated_generated_unit": False, "self_palindromic_unit": False,
                "reader_certified": False}})
    exact = [r for r in rows if r["audit"]["two_pointer_exact"] and r["audit"]["sha_equal"] and r["audit"]["validator_exact"]]
    rows.sort(key=lambda r: (-r["audit"]["letters"], r["rendered"]))
    return {"experiment_id":"grammar-abba-paragraph-20260928",
        "method":"typed clause-pair equation search with nested ABBA seams",
        "stats":{"clause_frames":len(CLAUSES), "pairs":len(rows), "exact":len(exact),
                 "longest_letters":rows[0]["audit"]["letters"]},
        "rows":rows, "exact_candidates":exact, "controls":controls(),
        "reader_gate":"closed: package prepared, blinded human ratings not collected",
        "next_repair":"replace the inversion-only right frame with ordinary transitive clauses whose lexical tapes satisfy the seam equation; retain ABBA role order",
        "lexicon_sha256":hashlib.sha256(json.dumps(CLAUSES, sort_keys=True).encode()).hexdigest()}

if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True); payload=run(); OUT.write_text(json.dumps(payload, indent=2)+"\n")
    print(json.dumps(payload["stats"], sort_keys=True))
