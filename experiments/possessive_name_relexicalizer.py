"""Audit a catalogue-family relexicalization ablation.

The surface grammar is a single ordinary permission event:

    SUBJECT lets RECIPIENT see OWNER's telegram.

Exactness is not obtained by reversing the word sequence.  The possessive
``OWNER's`` and the final noun cross the reflected word boundaries.  Candidate
names are independently chosen from a small readable name inventory; a literal
character check decides which joint lexicalizations close.

The famous Norah/Sharon surface is not the only disallowed material: all
relexicalizations of its full syntactic frame are catalogue-family derivatives.
This script preserves the construction as a negative ablation and must never
promote one of its exact closures as a generated lead.
"""
from __future__ import annotations

import argparse
from itertools import product
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import (
    FORBIDDEN_CATALOGUE_TAPES,
    is_boundary_aligned_word_mirror as shared_word_mirror,
    is_catalogue_family_derivative as shared_catalogue_family_derivative,
    mechanical_admission_checks,
    normalize_letters,
    tokenize,
)

SUBJECTS = ("marge", "nora", "marie", "sarah")
RECIPIENTS = ("hara", "aino", "norah", "ari", "ane")
OWNERS = ("sarah", "sonia", "sharon", "sari", "sena")
OBJECTS = ("telegram",)

# This public surface is retained only to identify the entire borrowed family.
# The normalized form makes the exact-control exclusion punctuation-free.
def is_catalogue_family_derivative(text: str) -> bool:
    """Reject the known ``Marge lets ... see ... telegram`` construction frame.

    Substituting names into a famous palindrome does not create an independently
    generated candidate under the project's no-catalogue-shortcut policy.
    """
    return shared_catalogue_family_derivative(tokenize(text))


def letters(text: str) -> str:
    return normalize_letters(text)


def words(text: str) -> tuple[str, ...]:
    return tokenize(text)


def is_boundary_aligned_word_mirror(units: tuple[str, ...]) -> bool:
    """Detect a palindrome formed solely by reversed whole-word counterparts."""
    return shared_word_mirror(units)


def render(subject: str, recipient: str, owner: str, object_word: str) -> str:
    return f"{subject.capitalize()} lets {recipient.capitalize()} see {owner.capitalize()}'s {object_word}."


def audit(text: str, *, local_catalogue: set[str]) -> dict[str, bool]:
    return mechanical_admission_checks(text, local_catalogue=local_catalogue,
                                       min_letters=30, max_letters=80)


def character_crossing_witness(text: str) -> list[dict]:
    """Show why word order alone cannot account for exactness."""
    tape = letters(text)
    # Name spans are shown only as character positions; punctuation is excluded.
    spans, at = [], 0
    for unit in words(text):
        width = len(letters(unit))
        spans.append((unit, at, at + width))
        at += width
    rows = []
    for unit, start, end in spans:
        mirror_start, mirror_end = len(tape) - end, len(tape) - start
        overlapping = [other for other, left, right in spans
                       if max(left, mirror_start) < min(right, mirror_end)]
        rows.append({
            "source_word": unit,
            "letter_span": [start, end],
            "reflected_letter_span": [mirror_start, mirror_end],
            "reflected_words_touched": overlapping,
        })
    return rows


def run() -> dict:
    local_catalogue = set(json.loads((ROOT / "data" / "known_palindromes.json").read_text()))
    attempted, records = [], []
    for subject, recipient, owner, object_word in product(SUBJECTS, RECIPIENTS, OWNERS, OBJECTS):
        text = render(subject, recipient, owner, object_word)
        attempted.append(text)
        checks = audit(text, local_catalogue=local_catalogue)
        if checks["exact_letter_palindrome"]:
            records.append({
                "rendered": text,
                "letters": len(letters(text)),
                "slots": {
                    "subject": subject,
                    "finite_verb": "lets",
                    "recipient": recipient,
                    "infinitive": "see",
                    "possessor": owner,
                    "object": object_word,
                },
                "dependency_witness": {
                    "root": "lets",
                    "nsubj": subject,
                    "obj": recipient,
                    "xcomp": "see",
                    "xcomp_obj": object_word,
                    "possessor": owner,
                },
                "character_crossing_witness": character_crossing_witness(text),
                "checks": checks,
                "rejection_codes": [name for name, ok in checks.items() if not ok],
                "reader_status": "rejected_catalogue_family_ablation; never a reader-study candidate",
            })
    admitted = [record for record in records if not record["rejection_codes"]]
    vocabulary = {
        "subjects": SUBJECTS, "recipients": RECIPIENTS,
        "owners": OWNERS, "objects": OBJECTS,
    }
    return {
        "status": "complete_rejected_catalogue_family_ablation",
        "operator": "catalogue-family relexicalization audit, not a candidate generator",
        "surface_grammar": "SUBJECT lets RECIPIENT see OWNER's telegram.",
        "vocabulary": vocabulary,
        "vocabulary_sha256": hashlib.sha256(json.dumps(vocabulary, sort_keys=True).encode()).hexdigest(),
        "generator_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "catalogue_provenance_sha256": hashlib.sha256(
            (ROOT / "data" / "catalogue_provenance.json").read_bytes()
        ).hexdigest(),
        "attempted_derivations": len(attempted),
        "exact_closures": records,
        "mechanically_admitted": admitted,
        "rejection_reason": (
            "Every exact closure shares the famous Marge-lets-see-possessive-telegram frame. "
            "A changed surface name does not satisfy the no-catalogue-shortcut acceptance gate."
        ),
        "promotion": "forbidden; retain only as a rejected ablation",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "derivations": result["attempted_derivations"],
                      "exact_closures": len(result["exact_closures"]),
                      "mechanically_admitted": len(result["mechanically_admitted"])}, indent=2))


if __name__ == "__main__":
    main()
