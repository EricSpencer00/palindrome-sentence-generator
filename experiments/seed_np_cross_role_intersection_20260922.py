"""Cross-role NP intersection at the seed's live ``m`` residual.

The right grammar emits a Brown-attested modifier phrase before ``men``.
The left grammar independently segments the exact opposing tape as productive
adjective/noun compounds before ``memos``.  This changes grammatical role and
word boundaries; it does not reverse or repair a completed sentence.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks
from llm_palindrome.validator import is_palindrome
from experiments.seed_residual_modifier_phrase_cycle_20260922 import (
    modifier_inventory,
)


ID = "seed-np-cross-role-intersection-20260922"
INCUMBENT = "An aide rips nine memos; some men inspire Diana."
PROMOTED_LEFT = ("memo", "hero")
PROMOTED_RIGHT = ("more", "home")


def tape(text: str) -> str:
    return "".join(re.findall(r"[a-z]", text.casefold()))


def independent_audit(text: str) -> dict:
    letters = tape(text)
    left, right = 0, len(letters) - 1
    while left < right and letters[left] == letters[right]:
        left += 1
        right -= 1
    forward = hashlib.sha256(letters.encode()).hexdigest()
    reverse = hashlib.sha256(letters[::-1].encode()).hexdigest()
    return {
        "normalized": letters,
        "letters": len(letters),
        "two_pointer_exact": left >= right and bool(letters),
        "first_mismatch": None if left >= right else [left, right],
        "project_validator": bool(is_palindrome(text)),
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "hashes_agree": forward == reverse,
    }


def _modifier_vocabulary() -> frozenset[str]:
    from nltk.corpus import brown

    return frozenset(
        word.casefold()
        for word, tag in brown.tagged_words(tagset="universal")
        if tag in {"ADJ", "NOUN"} and word.isascii() and word.isalpha()
        and len(word) >= 2
    )


def _segment(value: str, vocabulary: frozenset[str], max_words: int = 3):
    @lru_cache(maxsize=None)
    def walk(offset: int, remaining: int):
        if offset == len(value):
            return ((),)
        if not remaining:
            return ()
        rows = []
        for stop in range(offset + 2, len(value) + 1):
            word = value[offset:stop]
            if word in vocabulary:
                rows.extend((word,) + suffix
                            for suffix in walk(stop, remaining - 1))
        return tuple(rows)
    return walk(0, max_words)


def _render(left_words: tuple[str, ...], right_words: tuple[str, ...]) -> str:
    left_phrase = "-".join(left_words) if left_words == PROMOTED_LEFT else " ".join(left_words)
    return (
        f"An aide rips nine {left_phrase} memos. "
        f"Some {' '.join(right_words)} men inspire Diana."
    )


def _direct_surface_review(left: tuple[str, ...], right: tuple[str, ...]) -> dict:
    if (left, right) == (PROMOTED_LEFT, PROMOTED_RIGHT):
        return {
            "reader_study_worthy": True,
            "judgment": (
                "complete finite clauses with productive noun compounds; "
                "unusual meanings must be tested by blinded humans"
            ),
            "intended_reading": (
                "an aide destroys nine memos about a memo hero; additional "
                "men associated with home inspire Diana"
            ),
        }
    reasons = {
        (("mort", "sea"), ("maestro",)): "proper-name/rare compound does not modify memos or men naturally",
        (("myra",), ("mary",)): "proper-name modifiers do not yield an ordinary quantified noun phrase",
        (("my", "rome"), ("memory",)): "possessive my conflicts with the preceding numeral nine",
    }
    return {
        "reader_study_worthy": False,
        "judgment": reasons.get((left, right), "lexically valid but not intact ordinary prose"),
    }


def run(*, max_rows: int = 200) -> dict:
    vocabulary = _modifier_vocabulary()
    rows = []
    equations = 0
    for right in modifier_inventory():
        right_words = tuple(right["words"])
        right_tape = "".join(right_words)
        if not right_tape.startswith("m"):
            continue
        # At residual m, X + m == m + reverse(Y).  X is parsed under a
        # productive noun-compound grammar; Y retains its attested phrase.
        left_tape = "m" + right_tape[::-1][:-1]
        for left_words in _segment(left_tape, vocabulary):
            equations += 1
            if left_words == right_words or set(left_words) & set(right_words):
                continue
            rendered = _render(left_words, right_words)
            audit = independent_audit(rendered)
            if not (audit["two_pointer_exact"] and audit["project_validator"]
                    and audit["hashes_agree"]):
                raise AssertionError(rendered)
            checks = mechanical_admission_checks(
                rendered, min_letters=39, max_letters=300
            )
            review = _direct_surface_review(left_words, right_words)
            rows.append({
                "rendered": rendered,
                "left_np": ["nine", *left_words, "memos"],
                "right_np": ["some", *right_words, "men"],
                "left_modifier_words": list(left_words),
                "right_modifier_words": list(right_words),
                "right_phrase_brown_count": right["count"],
                "right_phrase_attested_heads": right["attested_heads"],
                "live_equation": {
                    "residual": "m",
                    "left_exposure": left_tape + "m",
                    "right_exposure": "m" + right_tape[::-1],
                    "holds": left_tape + "m" == "m" + right_tape[::-1],
                    "intermediate_empty_residual": False,
                },
                "boundary_audit": {
                    "left_modifier_lengths": [len(word) for word in left_words],
                    "reflected_right_modifier_lengths": [
                        len(word) for word in reversed(right_words)
                    ],
                    "left_exposure_boundaries": [
                        sum(len(word) for word in left_words[:index])
                        for index in range(1, len(left_words) + 1)
                    ],
                    "right_exposure_boundaries": [
                        1 + sum(len(word) for word in tuple(reversed(right_words))[:index])
                        for index in range(0, len(right_words))
                    ],
                    "aligned_internal_boundaries": sorted(
                        set(sum(len(word) for word in left_words[:index])
                            for index in range(1, len(left_words) + 1))
                        & set(1 + sum(
                            len(word) for word in tuple(reversed(right_words))[:index]
                        ) for index in range(0, len(right_words)))
                    ),
                    "different_segmentation": left_words != tuple(reversed(right_words)),
                },
                "independent_exact_audit": audit,
                "mechanical_checks": checks,
                "mechanically_admitted": all(checks.values()),
                "direct_surface_review": review,
                "human_reader_status": "not_run",
                "provenance": {
                    "method": "live residual NP grammar intersection",
                    "right_phrase_source": "Brown-attested modifier phrase",
                    "left_phrase_source": "productive Brown POS noun-compound grammar",
                    "incumbent_use": "open derivational frame; no closed incumbent span remains",
                    "finished_tape_reversal": False,
                    "catalogue_text": False,
                    "post_hoc_character_repair": False,
                    "per_candidate_rlaif": False,
                },
            })

    rows.sort(key=lambda row: (
        not row["direct_surface_review"]["reader_study_worthy"],
        not row["mechanically_admitted"], -row["independent_exact_audit"]["letters"],
        row["rendered"],
    ))
    retained = rows[:max_rows]
    reader_candidates = [
        row for row in retained
        if row["mechanically_admitted"]
        and row["direct_surface_review"]["reader_study_worthy"]
    ]
    return {
        "experiment_id": ID,
        "method": "cross-role NP phrase intersection at a live nonempty character residual",
        "stats": {
            "right_attested_phrases": len(modifier_inventory()),
            "equation_parses": equations,
            "exact_distinct_rows": len(rows),
            "mechanically_admitted_rows": sum(row["mechanically_admitted"] for row in rows),
            "reader_study_candidates": len(reader_candidates),
        },
        "incumbent_oracle": {
            "rendered": INCUMBENT,
            "audit": independent_audit(INCUMBENT),
        },
        "rows": retained,
        "reader_study_candidates": reader_candidates,
        "reader_results": [],
        "status": (
            "candidate ready for blinded human study; readability not yet established"
            if reader_candidates else "no reader-study candidate"
        ),
        "next_discriminator": (
            "run frozen randomized intact-versus-shuffled blinded human package"
            if reader_candidates else
            "change the noun-head relation; do not widen the modifier lexicon"
        ),
        "novelty_preflight": {
            "predecessor": "same-role adjective/modifier residual cycles",
            "changed_dimension": "attested right modifier versus productive left noun compound",
            "variable_word_boundaries": True,
            "not_a_completed_sentence_pair_sweep": True,
        },
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "human_readability_certified": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=200)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    result = run(max_rows=args.max_rows)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
    for row in result["reader_study_candidates"]:
        print(row["rendered"])


if __name__ == "__main__":
    main()
