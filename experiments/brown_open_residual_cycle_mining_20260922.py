"""Mine attested phrase *shapes* for fresh open-residual cycle schemas.

For a live left-owned debt ``r``, a two-edge cycle exists when the right edge
exposes ``r+s`` and the left edge exposes ``s+r``.  Since a right surface is
consumed backwards, its ordinary tape is ``reverse(s)+reverse(r)``.  This is a
cyclic-reversal equation, not an exact mirror-pair closure.

Brown spans are diagnostics and provenance-bearing shape proposals only.  They
are never emitted as generated candidates and cannot enter a reader packet.
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

from llm_palindrome.admission import ORDINARY_TWO_LETTER_WORDS


def letters(text: str) -> str:
    return "".join(re.findall(r"[a-z]", text.casefold()))


def constituent_class(tags: tuple[str, ...]) -> str | None:
    """Return a conservative reusable grammar role for a universal-POS span."""
    if not tags:
        return None
    if "VERB" in tags and tags[0] in {"DET", "NOUN", "PRON", "ADJ"}:
        return "clause_or_vp"
    if tags[0] == "ADP" and any(tag in {"NOUN", "PRON"} for tag in tags[1:]):
        return "pp"
    if tags[-1] in {"NOUN", "PRON"} and "VERB" not in tags:
        return "np"
    if tags[0] in {"VERB", "ADV"} and "VERB" in tags:
        return "vp_or_adjunct"
    return None


def ordinary(words: tuple[str, ...]) -> bool:
    return all(
        word.isascii() and word.isalpha()
        and (len(word) != 1 or word in {"a", "i"})
        and (len(word) != 2 or word in ORDINARY_TWO_LETTER_WORDS)
        for word in words
    )


def iter_brown_spans(*, min_words: int, max_words: int,
                     min_letters: int, max_letters: int):
    """Stream typed Brown spans without retaining the corpus in memory."""
    from nltk.corpus import brown

    for tagged in brown.tagged_sents(tagset="universal"):
        row = tuple(
            (word.casefold(), tag)
            for word, tag in tagged
            if word.isascii() and word.isalpha()
        )
        if not row:
            continue
        for start in range(len(row)):
            for width in range(min_words, max_words + 1):
                part = row[start:start + width]
                if len(part) != width:
                    break
                words = tuple(word for word, _tag in part)
                if not ordinary(words):
                    continue
                tags = tuple(tag for _word, tag in part)
                role = constituent_class(tags)
                if role is None:
                    continue
                tape = "".join(words)
                if not min_letters <= len(tape) <= max_letters:
                    continue
                yield tape, {
                    "text": " ".join(words),
                    "tags": list(tags),
                    "role": role,
                    "count": 1,
                    "source": "NLTK Brown attested span; diagnostic only",
                }


def brown_spans(*, min_words: int, max_words: int,
                min_letters: int, max_letters: int,
                max_surfaces_per_tape: int) -> tuple[dict[str, list[dict]], dict]:
    index: dict[str, list[dict]] = defaultdict(list)
    accepted = 0
    for tape, record in iter_brown_spans(
            min_words=min_words, max_words=max_words,
            min_letters=min_letters, max_letters=max_letters):
                accepted += 1
                rows = index[tape]
                existing = next((row for row in rows
                                 if row["text"] == record["text"]
                                 and row["tags"] == record["tags"]),
                                None)
                if existing is not None:
                    existing["count"] += 1
                elif len(rows) < max_surfaces_per_tape:
                    rows.append(record)
    retained_surfaces = sum(len(rows) for rows in index.values())
    return dict(index), {
        "accepted_span_occurrences": accepted,
        "retained_surfaces": retained_surfaces,
        "distinct_tapes": len(index),
    }


def targeted_brown_cycles(*, target_residuals: tuple[str, ...],
                          min_words: int, max_words: int,
                          min_letters: int, max_letters: int,
                          max_surfaces_per_tape: int,
                          max_rows: int) -> tuple[list[dict], dict, dict]:
    """Two-pass low-memory intersection for already measured live debts."""
    desired: dict[str, list[dict]] = defaultdict(list)
    accepted = rotations = 0
    for right_tape, right in iter_brown_spans(
            min_words=min_words, max_words=max_words,
            min_letters=min_letters, max_letters=max_letters):
        accepted += 1
        exposed = right_tape[::-1]
        for residual in target_residuals:
            rotations += 1
            if not exposed.startswith(residual) or len(residual) >= len(exposed):
                continue
            remainder = exposed[len(residual):]
            left_tape = remainder + residual
            rows = desired[left_tape]
            existing = next((row for row in rows
                             if row["residual"] == residual
                             and row["right"]["text"] == right["text"]), None)
            if existing is not None:
                existing["right"]["count"] += 1
            elif len(rows) < max_surfaces_per_tape:
                rows.append({"residual": residual, "remainder": remainder,
                             "right": right, "right_tape": right_tape})

    found: dict[tuple[str, str, str, str], dict] = {}
    left_occurrences = 0
    for left_tape, left in iter_brown_spans(
            min_words=min_words, max_words=max_words,
            min_letters=min_letters, max_letters=max_letters):
        if left_tape not in desired:
            continue
        left_occurrences += 1
        for proposal in desired[left_tape]:
            right = proposal["right"]
            if left["role"] != right["role"] or left["text"] == right["text"]:
                continue
            if set(left["text"].split()) & set(right["text"].split()):
                continue
            key = (proposal["residual"], left["text"], right["text"], left["role"])
            if key in found:
                found[key]["left_surface"]["count"] += 1
                continue
            residual, remainder = proposal["residual"], proposal["remainder"]
            assert proposal["right_tape"][::-1] == residual + remainder
            assert left_tape == remainder + residual
            found[key] = {
                "residual": residual,
                "remainder": remainder,
                "role": left["role"],
                "left_surface": dict(left),
                "right_surface": dict(right),
                "equation": {
                    "right_exposed": proposal["right_tape"][::-1],
                    "left_exposed": left_tape,
                    "start_debt": residual,
                    "mid_debt": remainder,
                    "end_debt": residual,
                    "exact_open_cycle": True,
                },
                "candidate_status": "diagnostic_shape_only_not_generated_text",
            }
    rows = sorted(found.values(), key=lambda row: (
        -(row["left_surface"]["count"] + row["right_surface"]["count"]),
        row["residual"], row["left_surface"]["text"],
    ))
    by_residual = Counter(row["residual"] for row in rows)
    stats = {
        "rotations_tested": rotations,
        "compatible_tapes_before_role_gates": len(desired),
        "role_compatible_disjoint_cycles": len(rows),
        "residuals_with_two_or_more_fresh_cycles": {
            residual: count for residual, count in by_residual.items() if count >= 2
        },
        "target_residuals": list(target_residuals),
    }
    inventory = {
        "accepted_span_occurrences_first_pass": accepted,
        "desired_left_tapes": len(desired),
        "matching_left_occurrences_second_pass": left_occurrences,
        "streaming_two_pass": True,
    }
    return rows[:max_rows], inventory, stats


def mine_cycles(index: dict[str, list[dict]], *, max_residual: int,
                max_rows: int,
                target_residuals: tuple[str, ...] = ()) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    rotations_tested = compatible_tapes = 0
    for right_tape in sorted(index):
        exposed = right_tape[::-1]
        residual_widths = (
            tuple(len(residual) for residual in target_residuals
                  if exposed.startswith(residual) and len(residual) < len(exposed))
            if target_residuals else
            tuple(range(1, min(max_residual, len(exposed) - 1) + 1))
        )
        for width in residual_widths:
            rotations_tested += 1
            residual, remainder = exposed[:width], exposed[width:]
            left_tape = remainder + residual
            if left_tape not in index:
                continue
            compatible_tapes += 1
            for left in index[left_tape]:
                for right in index[right_tape]:
                    if left["role"] != right["role"]:
                        continue
                    if left["text"] == right["text"]:
                        continue
                    if set(left["text"].split()) & set(right["text"].split()):
                        continue
                    # Independent replay of r + s -> s + r.
                    assert right_tape[::-1] == residual + remainder
                    assert left_tape == remainder + residual
                    rows.append({
                        "residual": residual,
                        "remainder": remainder,
                        "role": left["role"],
                        "left_surface": left,
                        "right_surface": right,
                        "equation": {
                            "right_exposed": right_tape[::-1],
                            "left_exposed": left_tape,
                            "start_debt": residual,
                            "mid_debt": remainder,
                            "end_debt": residual,
                            "exact_open_cycle": True,
                        },
                        "candidate_status": "diagnostic_shape_only_not_generated_text",
                    })
    rows.sort(key=lambda row: (
        -(row["left_surface"]["count"] + row["right_surface"]["count"]),
        -len(row["residual"]), row["left_surface"]["text"],
        row["right_surface"]["text"],
    ))
    by_residual = Counter(row["residual"] for row in rows)
    fresh_families = {
        residual: count for residual, count in by_residual.items() if count >= 2
    }
    return rows[:max_rows], {
        "rotations_tested": rotations_tested,
        "compatible_tapes_before_role_gates": compatible_tapes,
        "role_compatible_disjoint_cycles": len(rows),
        "residuals_with_two_or_more_fresh_cycles": fresh_families,
        "target_residuals": list(target_residuals),
    }


def run(*, min_words: int = 2, max_words: int = 6,
        min_letters: int = 6, max_letters: int = 28,
        max_residual: int = 8, max_surfaces_per_tape: int = 3,
        max_rows: int = 200,
        target_residuals: tuple[str, ...] = ()) -> dict:
    if target_residuals:
        cycles, inventory, stats = targeted_brown_cycles(
            target_residuals=target_residuals,
            min_words=min_words, max_words=max_words,
            min_letters=min_letters, max_letters=max_letters,
            max_surfaces_per_tape=max_surfaces_per_tape,
            max_rows=max_rows,
        )
    else:
        index, inventory = brown_spans(
            min_words=min_words, max_words=max_words,
            min_letters=min_letters, max_letters=max_letters,
            max_surfaces_per_tape=max_surfaces_per_tape,
        )
        cycles, stats = mine_cycles(index, max_residual=max_residual,
                                    max_rows=max_rows)
    return {
        "experiment_id": "brown-open-residual-cycle-mining-20260922",
        "method": "cyclic-reversal intersection of typed attested phrase shapes at a preserved nonempty residual",
        "config": {
            "min_words": min_words, "max_words": max_words,
            "min_letters": min_letters, "max_letters": max_letters,
            "max_residual": max_residual,
            "max_surfaces_per_tape": max_surfaces_per_tape,
            "target_residuals": list(target_residuals),
        },
        "inventory": inventory,
        "stats": stats,
        "cycle_shape_proposals": cycles,
        "reader_candidates": [],
        "provenance": {
            "source": "NLTK Brown corpus",
            "borrowed_spans_are_diagnostic_only": True,
            "generated_text_claim": False,
            "catalogue_palindromes": False,
            "per_candidate_rlaif": False,
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "status": (
            "fresh residual families available for authored grammar realization"
            if stats["residuals_with_two_or_more_fresh_cycles"] else
            "no multi-variant residual family in this bounded attested-span inventory"
        ),
        "next_gate": "Author independent semantic lexicalizations of one multi-variant residual family, then require central mechanical admission and blinded readers; never present Brown spans as generated prose.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-words", type=int, default=2)
    parser.add_argument("--max-words", type=int, default=6)
    parser.add_argument("--min-letters", type=int, default=6)
    parser.add_argument("--max-letters", type=int, default=28)
    parser.add_argument("--max-residual", type=int, default=8)
    parser.add_argument("--max-surfaces-per-tape", type=int, default=3)
    parser.add_argument("--max-rows", type=int, default=200)
    parser.add_argument("--target-residuals", default="",
                        help="comma-separated exact live debts; empty tests all widths")
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    result = run(
        min_words=args.min_words, max_words=args.max_words,
        min_letters=args.min_letters, max_letters=args.max_letters,
        max_residual=args.max_residual,
        max_surfaces_per_tape=args.max_surfaces_per_tape,
        max_rows=args.max_rows,
        target_residuals=tuple(filter(None, args.target_residuals.split(","))),
    )
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"inventory": result["inventory"],
                      "stats": result["stats"],
                      "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
