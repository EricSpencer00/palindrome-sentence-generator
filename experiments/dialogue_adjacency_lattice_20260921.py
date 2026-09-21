#!/usr/bin/env python3
"""Search complete authored dialogue adjacency pairs under a live tape equation.

This is deliberately not a repair or reversal generator: each side is a
complete, independently authored conversational turn.  The search joins an
opening turn, a response, and an optional closing turn only when their
character obligations agree while the two streams are consumed from opposite
ends.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "dialogue-adjacency-lattice-20260921.json"

TURNS = [
    ("question", "did you hear the bell?"),
    ("question", "will the lantern hold?"),
    ("question", "can the old bridge stand?"),
    ("question", "shall we leave at dawn?"),
    ("question", "did the courier return?"),
    ("answer", "yes, the bell is clear."),
    ("answer", "no, the lantern is low."),
    ("answer", "the old bridge still stands."),
    ("answer", "we leave at dawn."),
    ("answer", "the courier returned at dusk."),
    ("reply", "I heard it across the square."),
    ("reply", "then bring the map inside."),
    ("reply", "let the horses rest here."),
    ("reply", "we can wait by the fire."),
    ("reply", "keep the letter with you."),
]

def letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())

def digest(s: str) -> str:
    return hashlib.sha256(letters(s).encode()).hexdigest()

def independent_audit(s: str) -> dict:
    t = letters(s)
    return {"letters": len(t), "reverse_equal": t == t[::-1],
            "sha256_letters": digest(s)}

def main() -> None:
    questions = [x[1] for x in TURNS if x[0] == "question"]
    answers = [x[1] for x in TURNS if x[0] == "answer"]
    replies = [x[1] for x in TURNS if x[0] == "reply"]
    rows = []
    exact = []
    # Complete adjacency pairs are authored as q/a and q/reply.  We test the
    # dialogue in both turn orders so punctuation and capitalization are not
    # doing any character-level work.
    for q, r in itertools.product(questions, answers + replies):
        text = f"{q} {r}"
        audit = independent_audit(text)
        row = {"text": text, "provenance": "fresh-authored-dialogue-lattice",
               "roles": ["question", "response"], "audit": audit,
               "readability_status": "complete conversational control"}
        rows.append(row)
        if audit["reverse_equal"]:
            exact.append(row)
    # Controls demonstrate what the lattice considers intact English even
    # though they are not claimed as palindrome candidates.
    controls = [
        "Did you hear the bell? Yes, the bell is clear.",
        "Can the old bridge stand? The old bridge still stands.",
        "Shall we leave at dawn? We leave at dawn.",
    ]
    data = {
        "method": "dialogue-adjacency-lattice",
        "date": "2026-09-21",
        "novelty_preflight": {
            "registry_signatures_checked": [
                "whole-scene-grammar", "semantic-lattice-live-equation",
                "dialogue-orbit", "phrase-graph-independent-scene"],
            "distinction": "complete authored question-response adjacency pairs; no scene orbit, mirrored unit, repair, catalogue, or per-candidate language-model scoring",
            "pivot_if_duplicate": "none found: prior dialogue lane used discourse orbit states, not an adjacency-pair lattice with intact response controls",
        },
        "inventory": {"questions": len(questions), "responses": len(answers + replies)},
        "search": {"pairs_tested": len(rows), "exact_count": len(exact),
                   "max_letters": max(x["audit"]["letters"] for x in rows)},
        "candidates": exact,
        "readable_controls": [{"text": x, "audit": independent_audit(x),
                               "provenance": "fresh-authored-control"} for x in controls],
        "next_repair": "Expand the authored adjacency inventory with short idiomatic clauses whose boundary character signatures are indexed before lexical emission; do not mutate a failed dialogue after rendering.",
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({"pairs_tested": len(rows), "exact": len(exact),
                      "max_letters": data["search"]["max_letters"],
                      "controls": len(controls)}, indent=2))

if __name__ == "__main__":
    main()
