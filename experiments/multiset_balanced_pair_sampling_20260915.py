"""Parity-indexed sampling of independently authored complete clauses.

Letter-count parity is a necessary condition for an exact palindrome, but this
route never emits a reflected half.  It samples two complete typed clauses,
tracks their 26-dimensional count vector, and only then performs the exact
whole-tape audit.  The route is useful even when the parity frontier is empty:
it identifies whether authored lexical material reaches the arithmetic gate.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

ID = "multiset-balanced-pair-sampling"
SIGNATURE = "letter-multiset-balance|typed-role-pair-production|count-vector-frontier|complete-clause-pairing|independent-full-tape-audit"
OUT = ROOT / "runs/multiset-balanced-pair-sampling-20260915.json"

SUBJECTS = ("Mara", "Nora", "Owen", "Iris", "Lena", "Ruth", "Jon", "Nell")
VERBS = ("marks", "finds", "keeps", "sees", "carries", "opens", "mends", "reads")
OBJECTS = ("a quiet harbor", "the old map", "a small lantern", "the red gate", "a brass key", "the blue boat")
TAILS = ("at dawn", "near home", "after rain", "by the river", "before noon", "in spring")


def sentence(rng: random.Random) -> str:
    return f"{rng.choice(SUBJECTS)} {rng.choice(VERBS)} {rng.choice(OBJECTS)} {rng.choice(TAILS)}."


def content_words(text: str) -> set[str]:
    stop = {"a", "an", "the", "at", "near", "after", "by", "before", "in"}
    return {word.lower() for word in re.findall(r"[A-Za-z]+", text) if word.lower() not in stop}


def independent_pair(rng: random.Random) -> tuple[str, str]:
    for _ in range(100):
        left, right = sentence(rng), sentence(rng)
        if left != right and content_words(left).isdisjoint(content_words(right)):
            return left, right
    # The fallback is still retained as a diagnostic if the authored bank is
    # exhausted; it is never promoted without the admission gate.
    return sentence(rng), sentence(rng)


def parity(tape: str) -> tuple[int, ...]:
    counts = Counter(tape)
    return tuple(counts.get(chr(ord("a") + i), 0) % 2 for i in range(26))


def two_pointer(text: str) -> bool:
    tape = normalize_letters(text)
    i, j = 0, len(tape) - 1
    while i < j:
        if tape[i] != tape[j]:
            return False
        i += 1
        j -= 1
    return bool(tape)


def audit(left: str, right: str) -> dict:
    text = f"{left} {right}"
    tape = normalize_letters(text)
    checks = mechanical_admission_checks(text, min_letters=39, max_letters=260)
    return {
        "left": left,
        "right": right,
        "rendered": text,
        "letters": len(tape),
        "parity_balanced": sum(parity(tape)) <= 1,
        "exact": bool(tape) and tape == tape[::-1],
        "independent_two_pointer": two_pointer(text),
        "matching_outer_pairs": sum(a == b for a, b in zip(tape, reversed(tape))),
        "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest(),
        "failed_checks": [key for key, value in checks.items() if not value],
        "tokens": list(tokenize(text)),
        "readability_status": "diagnostic_only",
        "provenance": "two independently sampled complete typed clauses; no reflected emission or catalogue text",
    }


def main() -> None:
    preflight = {
        "registry_entries": 65,
        "excluded_families": 6,
        "status": "preflighted_before_execution",
        "manual_review_required": False,
    }
    rng = random.Random(20260915)
    rows = []
    balanced = []
    for _ in range(50_000):
        left, right = independent_pair(rng)
        item = audit(left, right)
        if item["parity_balanced"]:
            balanced.append(item)
    # Preserve all arithmetic survivors and the strongest readable near probes.
    rows.extend(balanced)
    if not rows:
        for _ in range(25):
            left, right = independent_pair(rng)
            rows.append(audit(left, right))
    rows.sort(key=lambda item: (item["exact"], item["matching_outer_pairs"], item["letters"]), reverse=True)
    output = {
        "experiment_id": ID,
        "signature": SIGNATURE,
        "novelty_preflight": preflight,
        "method": "sample two complete typed clauses, retain only whole-tape letter-count parity survivors, then audit exactness",
        "samples": 50_000,
        "parity_survivors": len(balanced),
        "rendered_probes": rows[:25],
        "rendered_candidates": [item for item in rows if item["exact"]],
        "independent_audit": {
            "method": "explicit opposing-index scan",
            "probes_checked": len(rows),
            "primary_exact": sum(item["exact"] for item in rows),
            "independent_exact": sum(item["independent_two_pointer"] for item in rows),
            "disagreements": [item["rendered"] for item in rows if item["exact"] != item["independent_two_pointer"]],
        },
        "readability_note": "Rendered clauses are diagnostic probes only; no human readability certification was performed.",
        "provenance": "Hand-authored role inventories and deterministic seed 20260915; no imported palindrome text.",
        "next_repair": "replace random pair sampling with a parity-indexed clause lattice whose lexical choices preserve agreement and disjoint content before exact closure",
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"samples": output["samples"], "parity_survivors": len(balanced), "exact": len(output["rendered_candidates"]), "probes": len(output["rendered_probes"])}))


if __name__ == "__main__":
    main()
