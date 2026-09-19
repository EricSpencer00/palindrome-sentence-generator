"""Bounded discourse-frame search with character equations at frame seams.

The constructor combines two independently complete clauses through a typed
attribution frame.  It does not wrap a pre-existing palindrome: every word is
selected before the character tape is checked, and proper interior spans are
rejected by the shared admission gate.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks
from llm_palindrome.validator import is_palindrome, normalize


SUBJECTS = ("the patient scribe", "the quiet poet", "a young sailor", "the careful keeper")
VERBS = ("copies", "writes", "keeps", "marks")
OBJECTS = ("the letter", "old notes", "a small map", "the red seal")
TIMES = ("at dawn", "by the shore", "after rain", "before night")
NAMES = ("Mara", "Jon", "Eli", "Nora")
FRAMES = (
    "{left}, said {name}; {right}.",
    "{left}, {name} said; {right}.",
    "{left}; so {name} told us, {right}.",
)


def clause_rows() -> list[dict]:
    rows = []
    for subject, verb, obj, time in itertools.product(SUBJECTS, VERBS, OBJECTS, TIMES):
        text = f"{subject} {verb} {obj} {time}"
        rows.append({"text": text, "source": "independent typed clause lattice",
                     "slots": {"subject": subject, "verb": verb, "object": obj, "time": time}})
    return rows


def audit(text: str) -> dict:
    normalized = normalize(text)
    return {"exact": is_palindrome(text), "length": len(normalized),
            "normalized": normalized, "admission": mechanical_admission_checks(text)}


def run(max_rows: int = 200, max_candidates: int = 50) -> dict:
    clauses = clause_rows()[:max_rows]
    candidates = []
    controls = []
    checked = 0
    for left, right, name, frame in itertools.product(clauses, clauses, NAMES, FRAMES):
        text = frame.format(left=left["text"], right=right["text"], name=name)
        exact = is_palindrome(text)
        result = audit(text) if exact or len(controls) < 3 else {"exact": False, "length": len(normalize(text))}
        checked += 1
        row = {"text": text, "audit": result,
               "provenance": {"left": left, "right": right, "name": name,
                              "frame": frame}}
        if result["exact"]:
            candidates.append(row)
            if len(candidates) >= max_candidates:
                break
        elif len(controls) < 3:
            row["audit"] = audit(text)
            controls.append(row)
    return {"method": "typed discourse-frame character equation",
            "clauses": len(clauses), "checked": checked,
            "exact": len(candidates), "admitted": sum(
                r["audit"]["admission"].get("eligible", False) for r in candidates),
            "candidates": candidates, "controls": controls,
            "next_repair": "Replace the attribution connective with a typed relative-clause seam and carry the first mirrored character obligation across the clause boundary; reject any repair that creates a proper palindrome span."}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-rows", type=int, default=200)
    args = p.parse_args()
    result = run(args.max_rows)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("clauses", "checked", "exact", "admitted")}))


if __name__ == "__main__":
    main()
