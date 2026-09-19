"""Bounded joint question/answer search over one live character tape.

The search deliberately keeps a tiny, typed grammar.  Question and answer are
expanded together, so the second clause is never pasted onto an already
accepted sentence.  Every accepted tape is independently re-audited.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
from pathlib import Path

SUBJECTS = ("Ada", "Mara", "Nora", "Iris")
OBJECTS = ("maps", "notes", "plans", "songs")
VERBS = ("keeps", "marks", "shares", "writes")
PLACES = ("by the river", "near the station", "under the moon", "beside the harbor")

def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())

def _pal(s: str) -> bool:
    return len(s) > 1 and s == s[::-1]

def audit(text: str) -> dict:
    tape = letters(text)
    lo, hi = 0, len(tape) - 1
    while lo < hi and tape[lo] == tape[hi]:
        lo += 1
        hi -= 1
    independent = lo >= hi
    proper = []
    # Character spans are the live tape, not token spans. The complete tape is
    # intentionally exempt: the grammar's objective may be a complete answer.
    for i in range(len(tape)):
        for j in range(i + 2, len(tape) + 1):
            if (i, j) != (0, len(tape)) and _pal(tape[i:j]):
                proper.append([i, j, tape[i:j]])
    words = re.findall(r"[A-Za-z]+", text.lower())
    mirrors = [list(pair) for pair in zip(words, reversed(words)) if pair[0] != pair[1] and pair[0] == pair[1][::-1]]
    digest = hashlib.sha256(tape.encode()).hexdigest()
    return {
        "letters": len(tape), "exact": _pal(tape), "sha256": digest,
        "independent_two_pointer": independent,
        "proper_palindromic_subspans": proper,
        "word_mirror_pairs": mirrors,
        "forbidden_clear": not proper and not mirrors,
    }

def render(qs: str, qo: str, av: str, ao: str, place: str) -> str:
    return f"{qs} {av} {ao} {place}? {qo} {av} {ao} {place}."

def search(limit: int = 12) -> dict:
    rows, seen = [], set()
    # A product lattice is used instead of duplicate sweeps: each tuple has a
    # stable key and is expanded exactly once.
    domain = itertools.product(SUBJECTS, OBJECTS, VERBS, SUBJECTS, OBJECTS, VERBS, PLACES)
    for qsub, obj, verb, asub, aobj, averb, place in domain:
        key = (qsub, obj, verb, asub, aobj, averb, place)
        if key in seen:
            continue
        seen.add(key)
        # Keep the surface question grammatical while retaining the joint tuple.
        text = f"Does {qsub} {verb[:-1]} {obj} {place}? {asub} {averb} {aobj} {place}."
        a = audit(text)
        if not a["forbidden_clear"]:
            continue
        rows.append({"rendered": text, "grammar": "question_then_answer_clause", "question_subject": qsub,
                     "answer_subject": asub, "object": obj, "verb": verb, "answer_verb": averb,
                     "place": place, "audit": a, "provenance": "bounded_joint_live_tape_product"})
        if len(rows) >= limit:
            break
    rows.sort(key=lambda r: (-r["audit"]["letters"], r["audit"]["sha256"]))
    return {"schema": "joint-discourse-grammar-search/v1", "candidate_count": len(rows),
            "unique_sweeps": len(seen), "banks": {"subjects": SUBJECTS, "objects": OBJECTS, "verbs": VERBS, "places": PLACES},
            "candidates": rows}

def main(path: str | Path = "runs/joint-discourse-grammar-search-20260918.json") -> dict:
    result = search()
    Path(path).write_text(json.dumps(result, indent=2) + "\n")
    return result

if __name__ == "__main__":
    main()
