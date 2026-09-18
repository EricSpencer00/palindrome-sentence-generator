#!/usr/bin/env python3
"""Dream-RSI lane: compose complete clause shells around a named center.

Unlike the phrase-trie lane, every side is required to parse as a complete
authored clause (determiner + subject + verb + object).  The reverse tape is
segmented only through the same clause grammar, so a lucky fragment is not
counted as a construction.  This is an exploratory lane: exactness is checked
by an independent two-pointer replay after generation.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "syntax-constrained-bridge-composition-20260918.json"

NAMES = ("Ada", "Eve", "Iris", "Mara", "Nora", "Rhea")
DETS = ("a", "an", "the")
SUBJECTS = ("baker", "captain", "clerk", "gardener", "pilot", "writer")
VERBS = ("marks", "opens", "reads", "guards", "charts", "finds")
OBJECTS = ("maps", "doors", "gates", "notes", "plans", "books")

def norm(s: str) -> str:
    return "".join(c.lower() for c in s if c.isalpha())

def audit(s: str) -> dict[str, object]:
    t = norm(s); i, j, mm = 0, len(t)-1, []
    while i < j:
        if t[i] != t[j]: mm.append([i, j])
        i += 1; j -= 1
    h = hashlib.sha256(t.encode()).hexdigest()
    return {"letters": len(t), "exact": not mm, "mismatch_count": len(mm),
            "first_mismatch": mm[0][0] if mm else None,
            "independent_two_pointer": not mm,
            "forward_reverse_sha256": [h, hashlib.sha256(t[::-1].encode()).hexdigest()]}

def clauses() -> list[str]:
    # Authored, ordinary complete clauses.  No source corpus or catalogue text.
    return [f"{d} {s} {v} {o}" for d in DETS for s in SUBJECTS
            for v in VERBS for o in OBJECTS]

def parse_clause(s: str) -> bool:
    p = s.lower().split()
    return len(p) == 4 and p[0] in DETS and p[1] in SUBJECTS and p[2] in VERBS and p[3] in OBJECTS

def main() -> None:
    rows = []
    # A named center is semantically explicit and never used as a hidden
    # letter carrier.  Search both sides as complete clauses around it.
    for left in clauses():
        for center in NAMES:
            target = (norm(left) + norm(center))[::-1]
            # Segment the target only as a complete clause tape.  The grammar
            # is deliberately exact: no arbitrary word-boundary fragments.
            for right in clauses():
                if norm(right) != target:
                    continue
                rendered = f"{left}; {center} {right}."
                a = audit(rendered)
                rows.append({"left_clause": left, "center": center,
                             "right_clause": right, "rendered": rendered,
                             "audit": a,
                             "provenance": {"lexicon": "authored-role-inventory-v1",
                                            "catalogue_used": False, "borrowed_text": False,
                                            "generator": Path(__file__).name},
                             "novelty_preflight": {"complete_left_clause": parse_clause(left),
                                                   "complete_right_clause": parse_clause(right),
                                                   "center_is_named": True,
                                                   "repeated_unit": left == right,
                                                   "self_palindromic_unit": norm(left) == norm(left)[::-1],
                                                   "punctuation_carries_letters": False},
                             "reader_status": "unreviewed; programmatic exactness is not readability"})
    # Always preserve representative near misses so the next repair has an
    # actual rendered object, even when the exact grammar intersection is empty.
    controls = []
    for left, center, right in (("the baker marks maps", "Ada", "a pilot reads notes"),
                                ("a writer charts gates", "Iris", "the clerk opens doors"),
                                ("the captain guards plans", "Nora", "an editor finds books")):
        rendered = f"{left}; {center} {right}."
        controls.append({"rendered": rendered, "audit": audit(rendered),
                         "provenance": {"lexicon": "authored-role-inventory-v1", "catalogue_used": False,
                                        "borrowed_text": False, "generator": Path(__file__).name},
                         "reader_status": "control; not an exact candidate"})
    payload = {"experiment": "syntax-constrained-bridge-composition-20260918",
               "method": "reverse a complete authored clause plus named center, then require the reversed tape to equal another complete clause; replay exactness independently",
               "candidate_count": len(rows), "candidates": rows, "controls": controls,
               "summary": {"exact_count": sum(r["audit"]["exact"] for r in rows),
                           "longest_letters": max((r["audit"]["letters"] for r in rows), default=0),
                           "next_repair": "add agreement-carrying verb forms and a live character-seam index while retaining complete-clause parsing; then run a blinded reader screen only if a fresh exact row survives"}}
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], sort_keys=True))

if __name__ == "__main__": main()
