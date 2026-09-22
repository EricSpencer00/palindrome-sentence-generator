"""Search distinct adjective cycles at the seed's live ``m`` residual.

After ``An aide rips nine`` is matched against the outside-in suffix
``men inspire Diana``, the right grammar owns residual ``m``.  Inserting a
left adjective tape ``X`` before ``memos`` and a right adjective tape ``Y``
before ``men`` returns to exactly that state iff ``X + m = m + reverse(Y)``.
The search intersects Brown-attested adjective sequences under this equation;
it never scores or repairs completed candidates.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks


ID = "seed-residual-adjective-cycle-20260922"
CONTROL = "An aide rips nine mere memos; some mere men inspire Diana."


def tape(text: str) -> str:
    return "".join(re.findall(r"[a-z]", text.casefold()))


def pointer_audit(text: str) -> dict:
    letters = tape(text)
    mismatch = next((i for i in range(len(letters) // 2)
                     if letters[i] != letters[-1 - i]), None)
    forward = hashlib.sha256(letters.encode()).hexdigest()
    reverse = hashlib.sha256(letters[::-1].encode()).hexdigest()
    return {
        "letters": len(letters),
        "exact": mismatch is None and bool(letters),
        "first_mismatch": mismatch,
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "hashes_agree": forward == reverse,
    }


def adjective_inventory(*, minimum_count: int = 2) -> tuple[str, ...]:
    """Return frequent Brown-attested attributive-compatible word forms."""
    from nltk.corpus import brown

    counts = Counter()
    for word, tag in brown.tagged_words(tagset="universal"):
        normalized = word.casefold()
        if (tag == "ADJ" and normalized.isascii() and normalized.isalpha()
                and len(normalized) >= 3):
            counts[normalized] += 1
    return tuple(sorted(word for word, count in counts.items()
                        if count >= minimum_count))


def _segment(tape_value: str, vocabulary: set[str], *, max_words: int = 2):
    """Yield all bounded adjective segmentations of one exact tape."""
    def walk(offset: int, chosen: tuple[str, ...]):
        if offset == len(tape_value):
            yield chosen
            return
        if len(chosen) >= max_words:
            return
        for stop in range(offset + 3, len(tape_value) + 1):
            word = tape_value[offset:stop]
            if word in vocabulary:
                yield from walk(stop, chosen + (word,))
    yield from walk(0, ())


def run(*, minimum_count: int = 2, max_rows: int = 200) -> dict:
    adjectives = adjective_inventory(minimum_count=minimum_count)
    vocabulary = set(adjectives)
    m_adjectives = tuple(word for word in adjectives if word.startswith("m"))
    rows = []
    equation_hits = 0

    # Y must begin with m for reverse(Y) to end in m.  That makes
    # X = m + reverse(Y)[:-1] a direct, indexed construction rather than a
    # Cartesian completed-sentence comparison.
    for width in (1, 2):
        tails = ((),) if width == 1 else ((word,) for word in adjectives)
        for first, tail in itertools.product(m_adjectives, tails):
            y_words = (first,) + tuple(tail)
            y_tape = "".join(y_words)
            x_tape = "m" + y_tape[::-1][:-1]
            for x_words in _segment(x_tape, vocabulary, max_words=2):
                equation_hits += 1
                if set(x_words) & set(y_words):
                    continue
                if len(set(x_words + y_words)) != len(x_words + y_words):
                    continue
                left_phrase = " ".join(x_words)
                right_phrase = " ".join(y_words)
                rendered = (
                    f"An aide rips nine {left_phrase} memos; "
                    f"some {right_phrase} men inspire Diana."
                )
                audit = pointer_audit(rendered)
                if not audit["exact"]:
                    raise AssertionError(rendered)
                checks = mechanical_admission_checks(
                    rendered, min_letters=39, max_letters=300
                )
                rows.append({
                    "rendered": rendered,
                    "left_adjectives": list(x_words),
                    "right_adjectives": list(y_words),
                    "letters": audit["letters"],
                    "cycle_equation": {
                        "residual": "m",
                        "left": x_tape + "m",
                        "right": "m" + y_tape[::-1],
                        "holds": x_tape + "m" == "m" + y_tape[::-1],
                        "intermediate_empty_residual": False,
                    },
                    "independent_exact_audit": audit,
                    "mechanical_checks": checks,
                    "mechanically_admitted": all(checks.values()),
                    "reader_status": (
                        "not_run; attributive syntax and exactness do not "
                        "certify semantic readability"
                    ),
                    "provenance": {
                        "method": "Brown adjective grammar intersected at live residual m",
                        "seed_use": "open grammar state and surrounding typed frame",
                        "seed_present_as_proper_span": False,
                        "finished_tape_reversal": False,
                        "catalogue_text": False,
                        "post_hoc_repair": False,
                        "per_candidate_rlaif": False,
                    },
                })

    rows.sort(key=lambda row: (-row["letters"], row["rendered"]))
    retained = rows[:max_rows]
    admitted = [row for row in retained if row["mechanically_admitted"]]
    control_checks = mechanical_admission_checks(
        CONTROL, min_letters=39, max_letters=300
    )
    return {
        "experiment_id": ID,
        "method": "residual-preserving adjective grammar cycle inside a typed clause pair",
        "state_equation": "X + m = m + reverse(Y)",
        "stats": {
            "brown_adjectives": len(adjectives),
            "m_initial_adjectives": len(m_adjectives),
            "equation_hits_before_distinctness": equation_hits,
            "distinct_exact_rows": len(rows),
            "retained_rows": len(retained),
            "mechanically_admitted_rows": len(admitted),
        },
        "repeated_word_control": {
            "rendered": CONTROL,
            "audit": pointer_audit(CONTROL),
            "mechanical_checks": control_checks,
            "admitted": all(control_checks.values()),
            "disposition": "exact 46-letter clue rejected because mere repeats",
        },
        "rows": retained,
        "mechanically_admitted_candidates": admitted,
        "reader_packet": [],
        "status": (
            "exact distinct-word adjective cycles require direct prose review"
            if admitted else
            "no distinct-word adjective cycle; retain m state and change the phrase grammar"
        ),
        "next_discriminator": (
            "direct prose review, then blinded intact/shuffled reader packet"
            if admitted else
            "replace adjective-only X/Y with asymmetric adjective-plus-relative phrases"
        ),
        "novelty_preflight": {
            "new_dimension": "productive nonempty-residual cycle inside the seed derivation",
            "not_a_larger_completed_sentence_sweep": True,
            "rejects_repeated_cycle_surface": True,
            "rejects_intermediate_closure": True,
        },
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "lexical_source": "Brown universal-tag adjectives attested at least twice",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--minimum-count", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=200)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    result = run(minimum_count=args.minimum_count, max_rows=args.max_rows)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))


if __name__ == "__main__":
    main()
