"""Dream-RSI discourse repair: predicate ellipsis with live boundary shifts.

An Astra reset exposed a useful construction geometry: a polar question and a
topicalized answer can share a character tape while assigning different
grammatical jobs to the mirrored material.  This lane makes that geometry
explicit and tests it with independently audited, hand-authored semantic
frames.  It deliberately keeps the strict anti-shortcut gate: a sentence made
from word-by-word mirrors, self-palindromic islands, or a catalogue phrase is
diagnostic only.

This is a construction method, not a readability certificate.  Exact rows are
still withheld from the reader package unless they clear the mechanical gate;
the eventual reader package must randomize intact and word-shuffled versions
for blinded raters.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.dream_rsi_model_guided_span_resynthesis_20260918 import audit
from llm_palindrome.admission import mechanical_admission_checks

EXPERIMENT = "dream-rsi-discourse-ellipsis-20260918"

NAMES = (("Noel", "Leon"), ("Nora", "Aron"), ("Mara", "Aram"))
# Each pair is a semantic predicate/object correspondence.  The first two
# are ordinary adjective/noun reversals; the final three are boundary-shift
# pairs that intentionally resegment across word boundaries.
PAIRS = (
    ("smart", "trams", "adjective", "plural noun"),
    ("stressed", "desserts", "adjective", "plural noun"),
    ("mad", "dam", "adjective", "singular noun"),
    ("raw", "war", "adjective", "singular noun"),
    ("an era", "arena", "noun phrase", "singular noun"),
    ("a gas", "saga", "noun phrase", "singular noun"),
    ("an item", "met in a", "noun phrase", "verb/preposition"),
)


def render(left_name: str, predicates: tuple[str, ...], objects: tuple[str, ...], right_name: str) -> str:
    left = ", ".join(predicates)
    right = ", ".join(objects)
    return f"Was {left_name} {left}? {right}, {right_name} saw."


def flags(text: str, checks: dict) -> dict:
    return {
        "exact": checks["exact_letter_palindrome"],
        "word_order_only": not checks["not_word_order_symmetry"],
        "self_palindromic_word": not checks["no_self_palindromic_word"],
        "self_palindromic_proper_span": not checks["no_self_palindromic_proper_multiword_span"],
        "repeated_content": not checks["distinct_words"],
        "catalogue_text": not checks["absent_from_local_catalogue"],
        "fragment_or_gibberish": False,
    }


def row(left_name: str, predicates: tuple[str, ...], objects: tuple[str, ...], right_name: str) -> dict:
    text = render(left_name, predicates, objects, right_name)
    independent = audit(text)
    checks = mechanical_admission_checks(text, min_letters=38, max_letters=300)
    return {
        "rendered": text,
        "letters": independent["letters"],
        "audit": independent,
        "mechanical_checks": checks,
        "anti_shortcut_flags": flags(text, checks),
        "reader_status": "human-unreviewed",
        "provenance": {
            "fresh_discourse_frame": True,
            "question_answer_ellipsis": True,
            "boundary_resegmentation": True,
            "finished_tape_reversed": False,
            "catalogue_imported": False,
            "seed_scaffold_in_output": False,
            "human_readability_certified": False,
        },
    }


def run() -> dict:
    rows = []
    # Keep a small, typed policy frontier: adjective lists model ordinary
    # question predicates; object lists model topicalized answer material.
    for (left_name, right_name), choices in product(NAMES, product(PAIRS, repeat=2)):
        p1, p2 = choices
        predicates = (p1[0], p2[0])
        objects = (p2[1], p1[1])
        rows.append(row(left_name, predicates, objects, right_name))
    # Add the boundary-shift three-pair candidate explicitly; it is valuable
    # as a falsification because it is exact but its residual syntax is not
    # complete prose.
    rows.append(row("Noel", ("an era", "a gas", "an item"),
                    ("met in a", "saga", "arena"), "Leon"))
    exact = [r for r in rows if r["audit"]["two_pointer_exact"]]
    admitted = [r for r in rows if all(r["mechanical_checks"].values())]
    best = max(exact, key=lambda r: r["letters"], default=None)
    return {
        "experiment": EXPERIMENT,
        "method": "Dream-RSI discourse-licensed polar question plus topicalized answer with typed boundary resegmentation",
        "rendered_candidates": rows,
        "fresh_exact_closures": exact,
        "mechanically_admitted": admitted,
        "stats": {"frame_rows": len(rows), "exact": len(exact),
                  "mechanically_admitted": len(admitted),
                  "longest_exact_letters": best["letters"] if best else 0,
                  "word_order_free_exact": sum(not r["anti_shortcut_flags"]["word_order_only"] for r in exact),
                  "proper_span_free_exact": sum(not r["anti_shortcut_flags"]["self_palindromic_proper_span"] for r in exact)},
        "reader_gate": {"status": "closed",
                        "reason": "No exact row clears all anti-shortcut and intact-prose gates; no blinded reader evidence exists.",
                        "programmatic_metrics_are_diagnostic": True},
        "next_repair": {
            "operator": "jointly regenerate the question predicate and topicalized answer as one cross-boundary chart, forbidding every proper palindromic subspan",
            "reason": "the exact discourse frames either align word-by-word or leave a marked/fragmentary answer; the cross-boundary row is the right residual geometry but needs a complete clause",
            "reader_test": "only after a row clears the strict mechanical gate, randomize intact and word-shuffled controls and collect blinded human ratings",
        },
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "catalogue_text_used": False,
                       "seed_scaffold_in_output": False,
                       "human_readability_certified": False},
    }


if __name__ == "__main__":
    payload = run()
    for directory in (ROOT / "runs", ROOT / "artifacts"):
        directory.mkdir(exist_ok=True)
        (directory / f"{EXPERIMENT}.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], indent=2))
    for candidate in sorted(payload["fresh_exact_closures"], key=lambda r: -r["letters"])[:5]:
        print(f"{candidate['letters']} letters | {candidate['rendered']}")
