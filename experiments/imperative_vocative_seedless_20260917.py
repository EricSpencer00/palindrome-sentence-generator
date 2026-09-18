"""Seedless exact search over imperative/vocative speech-act templates.

The left and right utterances are independently authored.  A non-palindromic
center token is inserted between them; exactness is checked only on the fully
rendered tape, followed by hard anti-shortcut checks.
"""
from __future__ import annotations
import argparse, itertools, json
from pathlib import Path
from llm_palindrome.admission import (normalize_letters, tokenize,
    has_distinct_content_words, has_repeated_nontrivial_unit,
    has_self_palindromic_proper_multiword_span, has_only_ordinary_short_words)

LEFT = [
    ("please help", "polite imperative"), ("open the door", "imperative"),
    ("close the gate", "imperative"), ("call the nurse", "imperative"),
    ("bring the map", "imperative"), ("read the note", "imperative"),
    ("watch the road", "imperative"), ("mind the child", "imperative"),
    ("carry a lamp", "imperative"), ("send a letter", "imperative"),
    ("tell the truth", "imperative"), ("save our town", "imperative"),
    ("please wait", "polite imperative"), ("look up", "imperative"),
]
RIGHT = [
    ("doctor", "vocative"), ("friend", "vocative"), ("nurse", "vocative"),
    ("my guide", "vocative"), ("dear captain", "vocative"),
    ("kind teacher", "vocative"), ("brave sailor", "vocative"),
    ("please help", "polite imperative"), ("open the door", "imperative"),
    ("call the nurse", "imperative"), ("read the note", "imperative"),
    ("watch the road", "imperative"), ("save our town", "imperative"),
]
CENTERS = ["was", "said", "asked", "told"]

def mismatch(a: str) -> int:
    return sum(x != y for x, y in zip(a, a[::-1])) // 2

def run() -> dict:
    rows = []
    for (left, lkind), (right, rkind), center in itertools.product(LEFT, RIGHT, CENTERS):
        rendered = f"{left}, {center}, {right}."
        letters = normalize_letters(rendered)
        units = tokenize(rendered)
        checks = {
            "exact": letters == letters[::-1],
            "distinct_content": has_distinct_content_words(units),
            "repeated_unit": has_repeated_nontrivial_unit(units),
            "proper_self_palindromic_span": has_self_palindromic_proper_multiword_span(units),
            "ordinary_short_words": has_only_ordinary_short_words(units),
        }
        rows.append({"rendered": rendered, "letters": letters, "length": len(letters),
                     "mismatch_pairs": mismatch(letters), "left_provenance": lkind,
                     "right_provenance": rkind, "center": center, "checks": checks,
                     "admitted": checks["exact"] and checks["distinct_content"] and not checks["repeated_unit"] and not checks["proper_self_palindromic_span"],
                     "provenance": "independent hand-authored imperative/vocative templates; seedless"})
    rows.sort(key=lambda x: (x["mismatch_pairs"], -x["length"]))
    exact = [x for x in rows if x["checks"]["exact"]]
    return {"signature": "seedless-imperative-vocative-utterance-lattice|typed-speech-act-templates|nonpalindromic-center-token|outside-in-character-equation|independent-full-tape-audit",
            "template_pairs": len(rows), "exact_candidates": len(exact),
            "mechanically_admitted": sum(x["admitted"] for x in rows), "reader_eligible": 0,
            "rendered_candidates_and_probes": rows[:40],
            "repair": "center-token sweep over four non-palindromic discourse edges; no seed-wrap and no repeated-content repair"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--output", required=True)
    args = ap.parse_args(); Path(args.output).write_text(json.dumps(run(), indent=2) + "\n")
