"""Generate four-sentence ABBA paragraphs with nonaligned sentence seams.

The left half must parse as two complete sentences ``A B``.  The right half
must parse independently in reading order as ``B-prime A-prime``.  Exact
letters are enforced during lexical choice, and only closures whose sentence
boundaries are staggered survive.  Thus no whole sentence is installed as the
mirror mate of another sentence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import (
    REPEATABLE_FUNCTION_WORDS,
    mechanical_admission_checks,
)
from llm_palindrome.paragraph_product import (
    audit_staggered_abba,
    staggered_abba_search,
)


ID = "staggered-abba-paragraph-product-20260922"

DET = ("a", "an", "no", "one", "the")
QUANT = ("many", "no", "some", "the", "two")
AGENT = (
    "actor", "aide", "artist", "clerk", "editor", "elder", "keeper",
    "nurse", "pilot", "poet", "ranger", "sailor", "scholar", "writer",
)
AGENTS = (
    "actors", "aides", "artists", "clerks", "editors", "elders", "keepers",
    "men", "nurses", "pilots", "poets", "rangers", "sailors", "scholars",
    "writers",
)
PAST = (
    "carried", "checked", "filed", "found", "guarded", "kept", "marked",
    "opened", "read", "recorded", "ripped", "saved", "sent", "signed",
    "sorted", "studied", "traced", "wrote",
)
PRESENT = (
    "admire", "assist", "encourage", "guide", "help", "inspire", "praise",
    "support", "surprise", "thank",
)
DOCUMENT = (
    "article", "book", "chart", "essay", "journal", "ledger", "letter",
    "map", "memo", "memos", "message", "note", "page", "plan", "poem",
    "report", "route", "story",
)
SCENE_OBJECT = (
    "bridge", "door", "gate", "harbor", "inlet", "lantern", "path",
    "signal", "station", "trail", "tower",
)
NAMES = (
    "Ada", "Anna", "Diana", "Eva", "Helena", "Joanna", "Julia", "Lena",
    "Mara", "Marina", "Nina", "Nora", "Regina", "Rosa", "Sara", "Tara",
)


def left_plans() -> tuple:
    return (
        (
            ("A:determiner", DET),
            ("A:agent", AGENT),
            ("A:event-past", PAST),
            ("A:object-determiner", DET),
            ("A:document", DOCUMENT),
        ),
        (
            ("B:name", NAMES),
            ("B:event-past", PAST),
            ("B:object-determiner", DET),
            ("B:scene-object", SCENE_OBJECT),
        ),
    )


def right_plans() -> tuple:
    return (
        (
            ("B-prime:name", NAMES),
            ("B-prime:event-past", PAST),
            ("B-prime:object-determiner", DET),
            ("B-prime:document", DOCUMENT),
        ),
        (
            ("A-prime:quantifier", QUANT),
            ("A-prime:agents", AGENTS),
            ("A-prime:response-present", PRESENT),
            ("A-prime:patient", NAMES),
        ),
    )


def _allow_partial(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    content = [
        word.casefold() for word in left + right
        if word.casefold() not in REPEATABLE_FUNCTION_WORDS
    ]
    return (
        len(content) == len(set(content))
        and all(word != word[::-1] for word in content)
    )


def topology_control() -> dict:
    """Known formulaic tape, resegmented only to exercise the seam audit."""
    sentences = (
        "Nora, I saw evil Noel.",
        "I saw war.",
        "Raw was I, Leon.",
        "Live was I, Aron.",
    )
    audit = audit_staggered_abba(sentences[:2], sentences[2:])
    rendered = " ".join(sentences)
    admission = mechanical_admission_checks(rendered, min_letters=39,
                                             max_letters=200)
    return {
        "rendered": rendered,
        "audit": audit,
        "mechanical_admission": admission,
        "mechanically_admitted": all(admission.values()),
        "status": "topology control only; inherited formulaic tape and punctuation resegmentation",
    }


def run(*, max_states: int = 1_000_000, max_results: int = 200) -> dict:
    search = staggered_abba_search(
        left_plans(), right_plans(), max_states=max_states,
        max_results=max_results, allow_partial=_allow_partial,
    )
    rows = []
    for candidate in search["cross_sentence_candidates"]:
        admission = mechanical_admission_checks(
            candidate["rendered"], min_letters=39, max_letters=300
        )
        rows.append({
            **candidate,
            "mechanical_admission": admission,
            "mechanically_admitted": all(admission.values()),
        })
    admitted = [row for row in rows if row["mechanically_admitted"]]
    frontiers = search["dead_frontiers"]
    return {
        "experiment_id": ID,
        "method": "four-sentence dual grammar with exact online character product and staggered ABBA sentence seams",
        "stats": {
            "states": search["states"],
            "transitions": search["transitions"],
            "cap_reached": search["cap_reached"],
            "exact_closures": len(search["candidates"]),
            "staggered_cross_sentence_closures": len(rows),
            "mechanically_admitted_gt38": len(admitted),
            "deepest_matched_letters": max(
                (row["matched_letters"] for row in frontiers), default=0
            ),
        },
        "topology_control": topology_control(),
        "exact_candidates": rows,
        "mechanically_admitted_candidates": admitted,
        "deepest_frontiers": frontiers,
        "reader_packet": [],
        "novelty_preflight": {
            "status": "passed",
            "distinction": "four independently complete sentences with nonaligned reflected boundaries; not a connector-separated two-clause half or a bank of closed mirror pairs",
            "finished_tape_reversal": False,
            "preclosed_sentence_units": False,
            "catalogue_text_in_search": False,
            "per_candidate_rlaif": False,
        },
        "provenance": {
            "lexicon": "fresh finite typed person/document/scene domains",
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "exact_audits": ["two-pointer", "forward/reverse SHA-256", "half-tape equation"],
            "reader_gate": "randomized blinded intact-versus-shuffled ratings after an admitted closure",
        },
        "status": (
            "admitted staggered paragraph requires blinded readers" if admitted
            else "no admitted staggered paragraph in this bounded grammar"
        ),
        "next_discriminator": (
            "If empty, replace the fixed SVO slot order with a packed clause automaton that can move the sentence boundary at a live residual; do not widen these word lists."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-states", type=int, default=1_000_000)
    parser.add_argument("--max-results", type=int, default=200)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    result = run(max_states=args.max_states, max_results=args.max_results)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stats": result["stats"], "status": result["status"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
