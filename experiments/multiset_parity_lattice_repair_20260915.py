"""Indexed parity-lattice repair for the multiset construction family.

Unlike the preceding random sampler, this repair enumerates a finite typed
clause lattice, indexes clauses by their 26-letter parity vector, and joins
only complementary (equal-parity) clauses with disjoint content words.  The
join is still ordinary prose; exactness is checked only after rendering.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

FAMILY = "multiset-balanced-pair-sampling"
SIGNATURE = "letter-multiset-balance|indexed-parity-clause-lattice|content-disjoint-pair-join|typed-role-agreement|independent-full-tape-audit"
SUBJECTS = ("Mara", "Nora", "Owen", "Iris", "Lena", "Ruth", "Jon", "Nell")
VERBS = ("marks", "finds", "keeps", "sees", "carries", "opens", "mends", "reads")
OBJECTS = ("a quiet harbor", "the old map", "a small lantern", "the red gate", "a brass key", "the blue boat")
TAILS = ("at dawn", "near home", "after rain", "by the river", "before noon", "in spring")


def norm(text: str) -> str:
    return normalize_letters(text)


def content_words(text: str) -> set[str]:
    stop = {"a", "an", "the", "at", "near", "after", "by", "before", "in"}
    return {word.lower() for word in re.findall(r"[A-Za-z]+", text) if word.lower() not in stop}


def parity(tape: str) -> tuple[int, ...]:
    counts = Counter(tape)
    return tuple(counts.get(chr(ord("a") + i), 0) % 2 for i in range(26))


def clauses() -> list[dict]:
    rows = []
    for subject in SUBJECTS:
        for verb in VERBS:
            for obj in OBJECTS:
                for tail in TAILS:
                    text = f"{subject} {verb} {obj} {tail}."
                    rows.append({"text": text, "parity": parity(norm(text)), "content": content_words(text)})
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


def audit(left: dict, right: dict) -> dict:
    text = f"{left['text']} {right['text']}"
    tape = norm(text)
    checks = mechanical_admission_checks(text, min_letters=39, max_letters=260)
    return {
        "left": left["text"],
        "right": right["text"],
        "rendered": text,
        "letters": len(tape),
        "exact": bool(tape) and tape == tape[::-1],
        "independent_two_pointer": two_pointer(text),
        "matching_outer_pairs": sum(a == b for a, b in zip(tape, reversed(tape))),
        "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest(),
        "failed_checks": [key for key, value in checks.items() if not value],
        "readability_status": "diagnostic_only",
        "provenance": "two independently authored typed clauses joined from an equal-parity bucket",
    }


def main() -> None:
    indexed: dict[tuple[int, ...], list[dict]] = defaultdict(list)
    for item in clauses():
        indexed[item["parity"]].append(item)
    rows = []
    for bucket in indexed.values():
        for left in bucket:
            for right in bucket:
                if left["text"] == right["text"] or not left["content"].isdisjoint(right["content"]):
                    continue
                rows.append(audit(left, right))
    parity_joined_pairs = len(rows)
    if not rows:
        # Preserve ordinary controls even when the arithmetic frontier is
        # empty; these are not parity survivors and cannot be promoted.
        flat = [item for bucket in indexed.values() for item in bucket]
        for left in flat:
            for right in flat:
                if left["text"] != right["text"] and left["content"].isdisjoint(right["content"]):
                    rows.append(audit(left, right))
                    if len(rows) >= 25:
                        break
            if len(rows) >= 25:
                break
    rows.sort(key=lambda item: (item["exact"], item["matching_outer_pairs"], item["letters"]), reverse=True)
    output = {
        "status": "repair_of_registered_family",
        "family": FAMILY,
        "signature": SIGNATURE,
        "preflight": {
            "registry_entries": 66,
            "excluded_families": 6,
            "manual_review_required": True,
            "overlap": FAMILY,
            "disposition": "repair, not a retained family",
        },
        "method": "enumerate typed complete clauses, index by letter parity, and join only content-disjoint equal-parity clauses",
        "clause_count": sum(len(bucket) for bucket in indexed.values()),
        "parity_buckets": len(indexed),
        "parity_joined_pairs": parity_joined_pairs,
        "rendered_probe_count": len(rows),
        "rendered_probes": rows[:25],
        "rendered_candidates": [item for item in rows if item["exact"]],
        "independent_audit": {
            "method": "explicit opposing-index scan",
            "probes_checked": len(rows),
            "primary_exact": sum(item["exact"] for item in rows),
            "independent_exact": sum(item["independent_two_pointer"] for item in rows),
            "disagreements": [item["rendered"] for item in rows if item["exact"] != item["independent_two_pointer"]],
        },
        "readability_note": "Complete clauses are preserved as probes only; no human readability certification was performed.",
        "next_repair": "add agreement-bearing tense and determiner states to the parity-indexed lattice before any larger lexical inventory",
        "provenance": "hand-authored typed role banks; no reflected emission or catalogue text",
    }
    out = ROOT / "runs/multiset-parity-lattice-repair-20260915.json"
    out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"clauses": output["clause_count"], "buckets": output["parity_buckets"], "parity_pairs": output["parity_joined_pairs"], "probes": output["rendered_probe_count"], "exact": len(output["rendered_candidates"])}))


if __name__ == "__main__":
    main()
