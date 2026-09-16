"""Agreement-bearing parity lattice repair.

This is the next operator after an empty parity frontier: present/past verb
forms and a/an/the agreement are part of the clause state while complete
clauses are indexed by their letter-count parity.  Only content-disjoint
same-parity pairs reach the rendered exact audit.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

FAMILY = "multiset-balanced-pair-sampling"
SIGNATURE = "letter-multiset-balance|agreement-bearing-tense-determiner-lattice|parity-indexed-clause-join|content-disjointness|independent-full-tape-audit"
SUBJECTS = ("Mara", "Nora", "Owen", "Iris", "Lena", "Ruth", "Jon", "Nell")
VERBS = (("marks", "marked"), ("finds", "found"), ("keeps", "kept"), ("sees", "saw"),
         ("carries", "carried"), ("opens", "opened"), ("mends", "mended"), ("reads", "read"))
ADJECTIVES = ("quiet", "old", "small", "red", "brass", "blue", "young", "calm")
NOUNS = ("harbor", "map", "lantern", "gate", "key", "boat", "letter", "garden", "river", "stone", "path", "house")
TAILS = ("at dawn", "near home", "after rain", "by the river", "before noon", "in spring", "at dusk", "under glass")


def norm(text: str) -> str:
    return normalize_letters(text)


def parity(tape: str) -> tuple[int, ...]:
    counts = Counter(tape)
    return tuple(counts.get(chr(ord("a") + i), 0) % 2 for i in range(26))


def content_words(text: str) -> set[str]:
    stop = {"a", "an", "the", "at", "near", "after", "by", "before", "in", "spring", "dawn", "noon", "dusk"}
    return {word.lower() for word in re.findall(r"[A-Za-z]+", text) if word.lower() not in stop}


def clauses() -> list[dict]:
    rows = []
    for subject, verb_pair, tense, adjective, noun, tail in itertools.product(SUBJECTS, VERBS, range(2), ADJECTIVES, NOUNS, TAILS):
        verb = verb_pair[tense]
        determiner = "an" if adjective[0] in "aeiou" else "a"
        text = f"{subject} {verb} {determiner} {adjective} {noun} {tail}."
        rows.append({"text": text, "parity": parity(norm(text)), "content": content_words(text), "tense": "present" if tense == 0 else "past"})
    return rows


def two_pointer(text: str) -> bool:
    tape = norm(text)
    i, j = 0, len(tape) - 1
    while i < j:
        if tape[i] != tape[j]:
            return False
        i += 1
        j -= 1
    return bool(tape)


def audit(left: dict, right: dict, include_checks: bool = True) -> dict:
    text = f"{left['text']} {right['text']}"
    tape = norm(text)
    item = {
        "left": left["text"],
        "right": right["text"],
        "rendered": text,
        "letters": len(tape),
        "left_tense": left["tense"],
        "right_tense": right["tense"],
        "exact": bool(tape) and tape == tape[::-1],
        "independent_two_pointer": two_pointer(text),
        "matching_outer_pairs": sum(a == b for a, b in zip(tape, reversed(tape))),
        "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest(),
        "readability_status": "diagnostic_only",
        "provenance": "two complete content-disjoint clauses joined from an equal-parity bucket",
    }
    if include_checks:
        checks = mechanical_admission_checks(text, min_letters=39, max_letters=260)
        item["failed_checks"] = [key for key, value in checks.items() if not value]
        item["admitted"] = all(checks.values())
        item["tokens"] = list(tokenize(text))
    return item


def main() -> None:
    index: dict[tuple[int, ...], list[dict]] = defaultdict(list)
    all_clauses = clauses()
    for item in all_clauses:
        index[item["parity"]].append(item)
    top: list[dict] = []
    exact = []
    pair_count = 0
    for bucket in index.values():
        for left in bucket:
            for right in bucket:
                if left["text"] == right["text"] or not left["content"].isdisjoint(right["content"]):
                    continue
                pair_count += 1
                item = audit(left, right, include_checks=False)
                if item["exact"]:
                    item = audit(left, right, include_checks=True)
                    exact.append(item)
                top.append(item)
    top.sort(key=lambda item: (item["exact"], item["matching_outer_pairs"], item["letters"]), reverse=True)
    probes = [audit_item if "failed_checks" in audit_item else {**audit_item, **{}}
              for audit_item in top[:25]]
    for item in probes:
        if "failed_checks" not in item:
            # Reconstruct the admission diagnostics from rendered text.
            checks = mechanical_admission_checks(item["rendered"], min_letters=39, max_letters=260)
            item["failed_checks"] = [key for key, value in checks.items() if not value]
            item["admitted"] = all(checks.values())
            item["tokens"] = list(tokenize(item["rendered"]))
    output = {
        "status": "repair_of_registered_family",
        "family": FAMILY,
        "signature": SIGNATURE,
        "preflight": {"registry_entries": 66, "excluded_families": 6, "manual_review_required": True, "overlap": FAMILY, "disposition": "repair, not a retained family"},
        "method": "agreement-bearing present/past and determiner states indexed by letter parity before content-disjoint joins",
        "clause_count": len(all_clauses),
        "parity_buckets": len(index),
        "parity_joined_pairs": pair_count,
        "rendered_probes": probes,
        "rendered_candidates": exact,
        "independent_audit": {"method": "explicit opposing-index scan", "probes_checked": len(probes), "primary_exact": sum(item["exact"] for item in probes), "independent_exact": sum(item["independent_two_pointer"] for item in probes), "disagreements": [item["rendered"] for item in probes if item["exact"] != item["independent_two_pointer"]]},
        "readability_note": "Complete clauses are diagnostic probes only; no human readability certification was performed.",
        "next_repair": "add controlled clause-boundary connective states to the parity lattice, retaining agreement and content-disjointness",
        "provenance": "hand-authored tense, determiner, adjective, noun, and tail inventories; no reflected emission or catalogue text",
    }
    out = ROOT / "runs/multiset-agreement-lattice-repair-20260915.json"
    out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"clauses": output["clause_count"], "buckets": output["parity_buckets"], "pairs": output["parity_joined_pairs"], "exact": len(exact)}))


if __name__ == "__main__":
    main()
