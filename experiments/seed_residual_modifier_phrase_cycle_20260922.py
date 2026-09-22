"""Intersect attested modifier phrases at the seed's live ``m`` residual.

This is the concrete successor to the adjective-only cycle.  Rather than add
more adjectives, it changes the production to Brown-attested one-to-three
word prenominal modifier phrases (adjective and noun compounds).  Opposing
phrases are selected by the exact live-state equation before a sentence is
rendered.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks


ID = "seed-residual-modifier-phrase-cycle-20260922"
ALLOWED = {"ADJ", "NOUN"}


def tape(text: str) -> str:
    return "".join(re.findall(r"[a-z]", text.casefold()))


def pointer_audit(text: str) -> dict:
    letters = tape(text)
    mismatch = next((i for i in range(len(letters) // 2)
                     if letters[i] != letters[-1 - i]), None)
    forward = hashlib.sha256(letters.encode()).hexdigest()
    reverse = hashlib.sha256(letters[::-1].encode()).hexdigest()
    return {
        "letters": len(letters), "exact": mismatch is None and bool(letters),
        "first_mismatch": mismatch, "sha256_forward": forward,
        "sha256_reverse": reverse, "hashes_agree": forward == reverse,
    }


def modifier_inventory(*, minimum_count: int = 1) -> tuple[dict, ...]:
    """Extract contiguous modifiers immediately before Brown nouns."""
    from nltk.corpus import brown

    counts = Counter()
    heads: dict[tuple[str, ...], Counter] = defaultdict(Counter)
    for sentence in brown.tagged_sents(tagset="universal"):
        normalized = [(word.casefold(), tag) for word, tag in sentence]
        for noun_index, (head, tag) in enumerate(normalized):
            if tag != "NOUN" or not head.isascii() or not head.isalpha():
                continue
            for width in (1, 2, 3):
                start = noun_index - width
                if start < 0:
                    continue
                span = normalized[start:noun_index]
                if not all(tag_value in ALLOWED and word.isascii()
                           and word.isalpha() and len(word) >= 2
                           for word, tag_value in span):
                    continue
                words = tuple(word for word, _tag in span)
                counts[words] += 1
                heads[words][head] += 1
    return tuple({
        "words": words,
        "count": counts[words],
        "attested_heads": [head for head, _count in heads[words].most_common(5)],
    } for words in sorted(counts) if counts[words] >= minimum_count)


def run(*, minimum_count: int = 1, max_rows: int = 500) -> dict:
    inventory = modifier_inventory(minimum_count=minimum_count)
    by_tape: dict[str, list[dict]] = defaultdict(list)
    for phrase in inventory:
        by_tape["".join(phrase["words"])].append(phrase)

    rows = []
    equation_hits = 0
    for right in inventory:
        right_tape = "".join(right["words"])
        if not right_tape.startswith("m"):
            continue
        left_tape = "m" + right_tape[::-1][:-1]
        for left in by_tape.get(left_tape, ()):
            equation_hits += 1
            left_words = tuple(left["words"])
            right_words = tuple(right["words"])
            if set(left_words) & set(right_words):
                continue
            if len(set(left_words + right_words)) != len(left_words + right_words):
                continue
            left_phrase = " ".join(left_words)
            right_phrase = " ".join(right_words)
            rendered = (
                f"An aide rips nine {left_phrase} memos; "
                f"some {right_phrase} men inspire Diana."
            )
            audit = pointer_audit(rendered)
            if not audit["exact"]:
                raise AssertionError(rendered)
            checks = mechanical_admission_checks(
                rendered, min_letters=39, max_letters=400
            )
            rows.append({
                "rendered": rendered,
                "left_modifier": list(left_words),
                "right_modifier": list(right_words),
                "left_brown_heads": left["attested_heads"],
                "right_brown_heads": right["attested_heads"],
                "brown_counts": {"left": left["count"], "right": right["count"]},
                "cycle_equation": {
                    "residual": "m",
                    "left": left_tape + "m",
                    "right": "m" + right_tape[::-1],
                    "holds": left_tape + "m" == "m" + right_tape[::-1],
                    "intermediate_empty_residual": False,
                },
                "independent_exact_audit": audit,
                "mechanical_checks": checks,
                "mechanically_admitted": all(checks.values()),
                "reader_status": (
                    "not_run; Brown head attestation is provenance, not a "
                    "readability certificate for memos/men"
                ),
                "provenance": {
                    "method": "Brown-attested modifier phrase intersection at residual m",
                    "seed_use": "open typed frame; completed seed was not wrapped",
                    "finished_tape_reversal": False,
                    "catalogue_text": False,
                    "post_hoc_repair": False,
                    "per_candidate_rlaif": False,
                },
            })

    rows.sort(key=lambda row: (
        -min(row["brown_counts"].values()), -row["independent_exact_audit"]["letters"],
        row["rendered"],
    ))
    retained = rows[:max_rows]
    admitted = [row for row in retained if row["mechanically_admitted"]]
    return {
        "experiment_id": ID,
        "method": "attested prenominal phrase grammar intersected at a nonempty residual",
        "state_equation": "X + m = m + reverse(Y)",
        "stats": {
            "attested_modifier_phrases": len(inventory),
            "m_initial_right_phrases": sum(
                "".join(row["words"]).startswith("m") for row in inventory
            ),
            "equation_hits_before_distinctness": equation_hits,
            "distinct_exact_rows": len(rows),
            "retained_rows": len(retained),
            "mechanically_admitted_rows": len(admitted),
        },
        "rows": retained,
        "mechanically_admitted_candidates": admitted,
        "reader_packet": [],
        "status": (
            "distinct exact modifier cycles require direct prose review"
            if admitted else
            "no distinct modifier cycle; change the m-state grammar role"
        ),
        "next_discriminator": (
            "direct prose review, then blinded intact/shuffled reader packet"
            if admitted else
            "move from symmetric prenominal roles to an asymmetric modifier/relative-clause transition"
        ),
        "novelty_preflight": {
            "predecessor": "adjective-only residual cycle",
            "changed_production": "attested adjective/noun compound modifiers",
            "not_a_larger_adjective_sweep": True,
            "intermediate_closure_rejected": True,
        },
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "lexical_source": "Brown universal-tag prenominal spans",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--minimum-count", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=500)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    result = run(minimum_count=args.minimum_count, max_rows=args.max_rows)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))


if __name__ == "__main__":
    main()
