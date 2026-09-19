"""Targeted center extension of the best polar-question diagnostic.

This is one construction operator, not a vocabulary sweep: insert a typed
reversible predicate/answer pair (``raw``/``war``) at the live center of the
44-letter polar-question tape.  The run is useful because it produces a
longer exact tape, while the independent admission audit makes clear why the
surface is not yet a reader-facing result.  The diagnostic is never promoted
to an admitted candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize


BASE_QUESTION = "Was Noel an era, a gas, an item?"
BASE_ANSWER = "met in a, saga, arena, Leon saw."

# This is deliberately a tiny, authored repair set.  Every exact closure is
# retained as a diagnostic, but each is a direct reversible-word insertion and
# therefore fails the hidden-span gate.  None is imported corpus/catalogue
# text and none is promoted as readable prose.
PAIRS = (
    {"left": "raw", "right": "war", "typing": "predicate adjective / past-tense verb"},
    {"left": "smart", "right": "trams", "typing": "predicate adjective / noun"},
    {"left": "mad", "right": "dam", "typing": "predicate adjective / noun"},
    {"left": "live", "right": "evil", "typing": "predicate adjective / adjective"},
    {"left": "stop", "right": "pots", "typing": "verb / plural noun"},
)


def render(pair: dict[str, str]) -> str:
    # The left token extends the question predicate and the reversed right
    # token begins the answer.  Punctuation is outside the normalized tape.
    return f"Was Noel an era, a gas, an item {pair['left']}? {pair['right'].capitalize()} {BASE_ANSWER}"


def audit(text: str) -> dict[str, object]:
    tape = normalize_letters(text)
    reverse = tape[::-1]
    mismatches = [(i, len(tape) - 1 - i) for i in range(len(tape) // 2)
                  if tape[i] != tape[-1 - i]]
    forward = hashlib.sha256(tape.encode()).hexdigest()
    backward = hashlib.sha256(reverse.encode()).hexdigest()
    return {
        "letters": len(tape),
        "normalized": tape,
        "two_pointer_exact": bool(tape) and not mismatches,
        "mismatch_count": len(mismatches),
        "first_mismatch": mismatches[0] if mismatches else None,
        "sha256_forward": forward,
        "sha256_reverse": backward,
        "sha_equal_under_reversal": forward == backward,
    }


def hidden_palindromic_spans(text: str) -> list[dict[str, object]]:
    words = tokenize(text)
    spans = []
    for start in range(len(words)):
        for end in range(start + 2, len(words) + 1):
            tape = normalize_letters(" ".join(words[start:end]))
            if len(tape) > 1 and tape == tape[::-1] and end - start < len(words):
                spans.append({"start": start, "end": end, "words": list(words[start:end]),
                              "letters": len(tape)})
    return spans


def run() -> dict[str, object]:
    rows = []
    for pair in PAIRS:
        text = render(pair)
        checks = mechanical_admission_checks(text, min_letters=39, max_letters=260)
        row = {
            "rendered": text,
            "pair": pair,
            "provenance": {
                "operator": "center insertion into polar-question diagnostic",
                "base_question": BASE_QUESTION,
                "base_answer": BASE_ANSWER,
                "catalogue_text_imported": False,
                "word_order_mirror": False,
                "fresh_authored_pair": True,
            },
            "audit": audit(text),
            "hidden_palindromic_spans": hidden_palindromic_spans(text),
            "mechanical_checks": checks,
            "mechanically_admitted": all(checks.values()),
            "reader_status": "not_run; this diagnostic is not reader-eligible",
        }
        rows.append(row)
    exact = [row for row in rows if row["audit"]["two_pointer_exact"]]
    admitted = [row for row in rows if row["mechanically_admitted"]]
    return {
        "experiment_id": "reversible-predicate-center-extension-20260918",
        "signature": "targeted-center-insertion|typed-reversible-predicate-answer|independent-pointer-sha-audit",
        "config": {"base_letters": len(normalize_letters(BASE_QUESTION + " " + BASE_ANSWER)),
                   "pair_count": len(PAIRS)},
        "rendered_candidates": rows,
        "stats": {
            "probes": len(rows),
            "exact": len(exact),
            "mechanically_admitted": len(admitted),
            "reader_eligible": 0,
            "longest_exact_letters": max((row["audit"]["letters"] for row in exact), default=0),
        },
        "provenance": {
            "independent_validator": "two-pointer normalized tape plus forward/reverse SHA-256",
            "human_readability_certified": False,
            "admission_policy": "exact rows with hidden self-palindromic spans remain diagnostics",
        },
        "next_repair": "replace the reversible center pair with a complete, non-palindromic finite predicate complement and carry its semantic arguments across the same boundary",
        "reader_gate": "closed; no human study is justified until the exact row clears strict mechanical admission",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], indent=2))


if __name__ == "__main__":
    main()
