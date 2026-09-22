"""Packed four-sentence grammar with movable ABBA sentence boundaries.

Unlike a fixed slot product, each sentence may take one of several complete
clause shapes.  The left grammar emits ``A`` then ``B`` in reading order.  The
right grammar is traversed outside-in through ``A-prime`` then ``B-prime`` and
is rendered normally as ``B-prime`` then ``A-prime``.  Exact letters, grammar
state, and the live residual advance together before any paragraph is rendered.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import (
    REPEATABLE_FUNCTION_WORDS,
    mechanical_admission_checks,
)
from llm_palindrome.paragraph_product import audit_staggered_abba
from llm_palindrome.recursive_product import Edge, search


ID = "packed-staggered-paragraph-automaton-20260922"

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
INTRANSITIVE = ("arrived", "returned", "stood", "waited", "walked")
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
PLACE = (
    "arena", "bridge", "garden", "harbor", "inlet", "marina", "plaza",
    "river", "station", "tower",
)
PREP = ("at", "beside", "beyond", "near", "under")
NAMES = (
    "Ada", "Anna", "Diana", "Eva", "Helena", "Joanna", "Julia", "Lena",
    "Mara", "Marina", "Nina", "Nora", "Regina", "Rosa", "Sara", "Tara",
)
PRONOUN = ("he", "she", "they", "we")

Slot = tuple[str, tuple[str, ...]]
Pattern = tuple[Slot, ...]


def _intro_patterns() -> tuple[Pattern, ...]:
    return (
        (("determiner", DET), ("agent", AGENT), ("event-past", PAST),
         ("object-determiner", DET), ("document", DOCUMENT)),
        (("name", NAMES), ("event-past", PAST),
         ("object-determiner", DET), ("scene-object", SCENE_OBJECT)),
        (("name", NAMES), ("motion-past", INTRANSITIVE), ("preposition", PREP),
         ("place-determiner", DET), ("place", PLACE)),
        (("quantifier", QUANT), ("agents", AGENTS),
         ("response-present", PRESENT), ("patient", NAMES)),
    )


def _followup_patterns() -> tuple[Pattern, ...]:
    return (
        (("pronoun", PRONOUN), ("event-past", PAST),
         ("object-determiner", DET), ("document", DOCUMENT)),
        (("name", NAMES), ("event-past", PAST),
         ("object-determiner", DET), ("scene-object", SCENE_OBJECT)),
        (("name", NAMES), ("motion-past", INTRANSITIVE), ("preposition", PREP),
         ("place-determiner", DET), ("place", PLACE)),
        (("determiner", DET), ("agent", AGENT), ("event-past", PAST),
         ("object-determiner", DET), ("document", DOCUMENT)),
    )


def _add_patterns(graph: dict[str, list[Edge]], source: str, target: str,
                  phase: str, patterns: tuple[Pattern, ...], *,
                  reverse_slots: bool = False) -> None:
    for pattern_index, original in enumerate(patterns):
        pattern = tuple(reversed(original)) if reverse_slots else original
        current = source
        for slot_index, (role, words) in enumerate(pattern):
            following = (
                target if slot_index == len(pattern) - 1
                else f"{phase}:{pattern_index}:{slot_index + 1}"
            )
            for word in words:
                graph[current].append(
                    Edge(following, word, phase=phase, role=role)
                )
            current = following


def grammars() -> tuple[dict[str, tuple[Edge, ...]],
                        dict[str, tuple[Edge, ...]]]:
    left: dict[str, list[Edge]] = defaultdict(list)
    right: dict[str, list[Edge]] = defaultdict(list)
    _add_patterns(left, "S", "Q", "A", _intro_patterns())
    _add_patterns(left, "Q", "F", "B", _followup_patterns())
    # Outside-in traversal sees the last sentence and its last word first.
    _add_patterns(right, "S", "Q", "A-prime", _intro_patterns(),
                  reverse_slots=True)
    _add_patterns(right, "Q", "F", "B-prime", _followup_patterns(),
                  reverse_slots=True)
    return ({state: tuple(edges) for state, edges in left.items()},
            {state: tuple(edges) for state, edges in right.items()})


def _sentences(words: tuple[str, ...], phases: tuple[str, ...],
               expected: tuple[str, str]) -> tuple[str, str] | None:
    if len(words) != len(phases) or not words:
        return None
    groups: list[tuple[str, list[str]]] = []
    for word, phase in zip(words, phases):
        if not groups or groups[-1][0] != phase:
            groups.append((phase, [word]))
        else:
            groups[-1][1].append(word)
    if tuple(phase for phase, _ in groups) != expected:
        return None
    rendered = []
    for _phase, group in groups:
        sentence = " ".join(group)
        sentence = sentence[:1].upper() + sentence[1:]
        rendered.append(sentence + ".")
    return rendered[0], rendered[1]


def _unique_content(words: tuple[str, ...]) -> bool:
    content = [
        word.casefold() for word in words
        if word.casefold() not in REPEATABLE_FUNCTION_WORDS
    ]
    return (len(content) == len(set(content))
            and all(word != word[::-1] for word in content))


def run(*, max_states: int = 2_000_000,
        max_results: int = 200) -> dict:
    left, right = grammars()
    report = search(left, right, max_states=max_states,
                    max_results=max_results,
                    reject_intermediate_closure=False)
    rows = []
    for witness in report.witnesses:
        left_sentences = _sentences(witness.left_words, witness.left_phases,
                                    ("A", "B"))
        right_sentences = _sentences(witness.right_words, witness.right_phases,
                                     ("B-prime", "A-prime"))
        if left_sentences is None or right_sentences is None:
            continue
        rendered = " ".join(left_sentences + right_sentences)
        structural = audit_staggered_abba(left_sentences, right_sentences)
        admission = mechanical_admission_checks(rendered, min_letters=39,
                                                 max_letters=300)
        words = witness.left_words + witness.right_words
        rows.append({
            "rendered": rendered,
            "left_sentences": list(left_sentences),
            "right_sentences": list(right_sentences),
            "left_roles": list(witness.left_roles),
            "right_roles": list(witness.right_roles),
            "structural_audit": structural,
            "unique_content_words": _unique_content(words),
            "mechanical_admission": admission,
            "mechanically_admitted": (
                structural["cross_sentence_coupled"]
                and _unique_content(words) and all(admission.values())
            ),
        })
    admitted = [row for row in rows if row["mechanically_admitted"]]
    reachable_second_left = sum(state.left == "Q" or state.left.startswith("B:")
                                for state in report.reachable)
    reachable_second_right = sum(state.right == "Q" or state.right.startswith("B-prime:")
                                 for state in report.reachable)
    return {
        "experiment_id": ID,
        "method": "packed alternative-clause ABBA automata intersected online at character residuals",
        "stats": {
            "left_states": len(left),
            "right_states": len(right),
            "reachable_product_states": len(report.reachable),
            "coaccessible_product_states": len(report.coaccessible),
            "cap_reached": len(report.reachable) > max_states,
            "exact_witnesses": len(report.witnesses),
            "staggered_exact_witnesses": sum(
                row["structural_audit"]["cross_sentence_coupled"]
                for row in rows
            ),
            "mechanically_admitted_gt38": len(admitted),
            "states_reaching_left_B": reachable_second_left,
            "states_reaching_outside_in_right_B_prime": reachable_second_right,
            "maximum_live_residual": max(
                (len(state.residual) for state in report.reachable), default=0
            ),
        },
        "exact_candidates": rows,
        "mechanically_admitted_candidates": admitted,
        "reader_packet": [],
        "novelty_preflight": {
            "status": "passed",
            "representation_change": "packed alternative clause paths and movable sentence termination replace the failed fixed SVO slot order",
            "word_bank_widening": False,
            "finished_tape_reversal": False,
            "preclosed_sentence_pairs": False,
            "per_candidate_rlaif": False,
        },
        "provenance": {
            "lexical_domains": "finite authored typed domains",
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "reader_gate": "only exact, cross-sentence, centrally admitted outputs enter randomized blinded intact/shuffled ratings",
        },
        "status": (
            "admitted paragraph requires blinded readers" if admitted
            else "no admitted paragraph in the packed automaton"
        ),
        "next_discriminator": (
            "If no witness reaches both second-sentence states, learn entry paths from the measured residual graph; if both do, add semantic discourse binding before any lexical expansion."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-states", type=int, default=2_000_000)
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
