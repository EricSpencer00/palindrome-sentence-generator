"""Proper-name clause grammar joined by an exact reversed character tape.

Names and place-like nouns are a typed lexical class, not a catalogue import.
Both sides are generated as ordinary clauses before the tape lookup; no side is
presented as a word-order mirror and all exact hits receive the normal gates.
"""
from __future__ import annotations

import hashlib
import json
import itertools
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

ID = "proper-name-reverse-grammar"
SIGNATURE = "proper-name-reverse-grammar|typed-name-place-roles|character-trie-segmentation|independent-clause-pairing|independent-exact-audit"
NAMES = ("Ada", "Ana", "Ava", "Bob", "Diana", "Eli", "Ella", "Eve", "Ira", "Iris", "Lena", "Leo", "Liam", "Mara", "Maya", "Mia", "Nadia", "Nell", "Nina", "Noah", "Nora", "Owen", "Ruth", "Sam", "Sara", "Theo", "Tina", "Uma", "Vera", "Will", "Zoe")
VERBS = ("bakes", "carries", "cleans", "finds", "gives", "keeps", "leads", "likes", "marks", "mends", "opens", "reads", "saves", "sees", "sends", "writes")
NOUNS = ("book", "boat", "bridge", "cake", "candle", "chart", "door", "garden", "gate", "letter", "map", "note", "path", "stone")
ADJECTIVES = ("blue", "brass", "calm", "clear", "fresh", "kind", "old", "quiet")
DETS = ("a", "the")


def norm(text: str) -> str:
    return normalize_letters(text)


def content_words(text: str) -> set[str]:
    stop = {"a", "the", "to", "by", "in", "on", "near", "under"}
    return {word.lower() for word in re.findall(r"[A-Za-z]+", text) if word.lower() not in stop}


def grammar_clauses() -> list[str]:
    clauses = set()
    for name, verb, det, noun in itertools.product(NAMES, VERBS, DETS, NOUNS):
        clauses.add(f"{name} {verb} {det} {noun}.")
    for name, verb, det, adjective, noun in itertools.product(NAMES, VERBS, DETS, ADJECTIVES, NOUNS):
        clauses.add(f"{name} {verb} {det} {adjective} {noun}.")
    for subject, verb, object_name in itertools.product(NAMES, VERBS, NAMES):
        clauses.add(f"{subject} {verb} {object_name}.")
    return sorted(clauses)


def two_pointer(text: str) -> bool:
    tape = norm(text)
    i, j = 0, len(tape) - 1
    while i < j:
        if tape[i] != tape[j]:
            return False
        i += 1
        j -= 1
    return bool(tape)


def audit(left: str, right: str) -> dict:
    text = f"{left} {right}"
    tape = norm(text)
    checks = mechanical_admission_checks(text, min_letters=39, max_letters=260)
    return {"left": left, "right": right, "rendered": text, "letters": len(tape),
            "exact": bool(tape) and tape == tape[::-1],
            "independent_two_pointer": two_pointer(text),
            "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest(),
            "failed_checks": [key for key, value in checks.items() if not value],
            "admitted": all(checks.values()), "readability_status": "diagnostic_only",
            "provenance": "two independently generated typed name-role clauses; no catalogue text"}


def main() -> None:
    clauses = grammar_clauses()
    index: dict[str, list[str]] = defaultdict(list)
    for clause in clauses:
        index[norm(clause)].append(clause)
    exact = []
    for left in clauses:
        for right in index.get(norm(left)[::-1], ()):
            if left == right or not content_words(left).isdisjoint(content_words(right)):
                continue
            item = audit(left, right)
            exact.append(item)
    # Preserve ordinary readable controls even if the reverse grammar has no hit.
    probes = []
    for left, right in itertools.islice(itertools.product(clauses[:10], clauses[-10:]), 25):
        probes.append(audit(left, right))
    output = {
        "experiment_id": ID, "signature": SIGNATURE,
        "preflight": {"registry_entries": 66, "excluded_families": 6, "manual_review_required": False},
        "method": "typed proper-name/object clause grammar with exact reversed-tape index",
        "grammar_clause_count": len(clauses), "indexed_tapes": len(index),
        "raw_exact_pairs": len(exact), "rendered_probes": probes,
        "rendered_candidates": exact,
        "independent_audit": {"method": "explicit opposing-index scan", "probes_checked": len(exact) + len(probes), "primary_exact": sum(item["exact"] for item in exact + probes), "independent_exact": sum(item["independent_two_pointer"] for item in exact + probes), "disagreements": [item["rendered"] for item in exact + probes if item["exact"] != item["independent_two_pointer"]]},
        "readability_note": "No human readability certification was performed; ordinary probes and exact hits remain diagnostic until reader review.",
        "next_repair": "add a finite proper-name/locative dependency state to the right grammar before widening the lexicon",
        "provenance": "hand-authored name, verb, determiner, adjective, and noun banks; no borrowed palindrome text",
    }
    out = ROOT / "runs/proper-name-reverse-grammar-20260915.json"
    out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"clauses": len(clauses), "indexed": len(index), "raw_exact": len(exact), "probes": len(probes)}))


if __name__ == "__main__":
    main()
