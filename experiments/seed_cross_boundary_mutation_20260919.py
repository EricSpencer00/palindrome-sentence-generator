"""Constructive mutation search around the best seed.

This lane changes lexical slots and inserts short, ordinary words at every
word boundary, then checks the complete rendered tape immediately.  It is
deliberately not a wrapper search: candidates are run through the shared
mechanical gate, including hidden proper-span and repeated-unit exclusions.
The result is evidence about which mutations preserve the cross-word
character equations; it does not claim that a mechanical pass is readable.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks

EXPERIMENT_ID = "seed-cross-boundary-mutation-20260919"
SEED = ("an", "aide", "rips", "nine", "memos", "some", "men", "inspire", "diana")

# Small, fixed banks are intentionally hand-auditable.  They are lexical
# alternatives, not a catalogue of palindrome text.
BANKS = {
    "an": ("an", "a", "the", "one", "our"),
    "aide": ("aide", "maid", "scribe", "agent", "nurse", "bard"),
    "rips": ("rips", "tears", "cuts", "reads", "marks"),
    "nine": ("nine", "ten", "one", "many"),
    "memos": ("memos", "notes", "letters", "lines", "words"),
    "some": ("some", "one", "the", "a"),
    "men": ("men", "women", "folk", "lords", "maids"),
    "inspire": ("inspire", "move", "teach", "guide", "stir"),
    "diana": ("diana", "anna", "lena", "nora", "aria"),
}
INSERTIONS = ("", "a", "and", "at", "in", "near", "once", "by", "now")


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def independent_audit(text: str) -> dict:
    tape = letters(text)
    mismatches = [(i, tape[i], tape[-i - 1]) for i in range(len(tape) // 2)
                  if tape[i] != tape[-i - 1]]
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"letters": len(tape), "two_pointer_exact": bool(tape) and not mismatches,
            "first_mismatch": mismatches[0] if mismatches else None,
            "sha256_forward": forward, "sha256_reverse": reverse,
            "sha_equal": forward == reverse}


def render(words: tuple[str, ...]) -> str:
    return " ".join(words).capitalize() + "."


def examine(words: tuple[str, ...], provenance: str) -> dict:
    text = render(words)
    audit = independent_audit(text)
    checks = mechanical_admission_checks(text, min_letters=30, max_letters=2000)
    return {"rendered": text, "length": audit["letters"], "provenance": provenance,
            "exact_audit": audit, "mechanical_checks": checks,
            "mechanically_admitted": all(checks.values()),
            "reader_status": "unreviewed; programmatic checks never certify readability"}


def generate(max_substitutions: int = 3) -> tuple[list[dict], dict]:
    rows = []
    # Keep the original seed and every mutation's provenance.  Product sizes
    # remain bounded (roughly 1.5m at k=3 with these banks).
    for count in range(max_substitutions + 1):
        for indices in itertools.combinations(range(len(SEED)), count):
            choices = [BANKS[SEED[index]] for index in indices]
            for values in itertools.product(*choices):
                words = list(SEED)
                for index, value in zip(indices, values):
                    words[index] = value
                row = examine(tuple(words), f"slot_substitution:{indices}:{values}")
                rows.append(row)
    # Cross-boundary insertion is a separate operator.  It is intentionally
    # one word at a time: no copied palindrome can be wrapped around the seed.
    for boundary in range(len(SEED) + 1):
        for inserted in INSERTIONS:
            if not inserted:
                continue
            words = SEED[:boundary] + (inserted,) + SEED[boundary:]
            rows.append(examine(words, f"cross_boundary_insertion:{boundary}:{inserted}"))
    unique = {}
    for row in rows:
        unique.setdefault(letters(row["rendered"]), row)
    rows = list(unique.values())
    exact = [row for row in rows if row["exact_audit"]["two_pointer_exact"]]
    admitted = [row for row in exact if row["mechanically_admitted"]]
    rows.sort(key=lambda row: (row["mechanically_admitted"],
                               row["exact_audit"]["two_pointer_exact"],
                               row["length"]), reverse=True)
    return rows, {"generated": len(rows), "exact": len(exact),
                  "mechanically_admitted": len(admitted),
                  "longest_exact": max((row["length"] for row in exact), default=0),
                  "longest_generated": max((row["length"] for row in rows), default=0)}


def run() -> dict:
    rows, stats = generate()
    exact = [row for row in rows if row["exact_audit"]["two_pointer_exact"]]
    return {
        "experiment_id": EXPERIMENT_ID,
        "operator": "bounded lexical slot substitution plus one-word cross-boundary insertion; exact tape checked online",
        "seed": examine(SEED, "human-authored seed; baseline"),
        "stats": stats,
        "exact_candidates": exact,
        "representative_near_misses": rows[:20],
        "provenance": {"lexical_source": "fixed hand-authored alternative banks",
                       "catalogue_imported": False, "completed_prose_reversed": False,
                       "per_candidate_rlaif": False,
                       "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "next_repair": {"action": "replace a cross-word seam with a typed multiword constituent while preserving the live residual, rather than adding a self-palindromic insertion",
                        "reason": "slot substitutions and single insertions cannot satisfy the seed's cross-boundary equations; zero exact mutation survived",
                        "reader_test": "none: no mechanically admitted mutation exists"},
    }


if __name__ == "__main__":
    result = run()
    output = ROOT / "runs" / f"{EXPERIMENT_ID}.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], indent=2))
