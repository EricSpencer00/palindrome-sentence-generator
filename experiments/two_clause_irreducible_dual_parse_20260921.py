"""Irreducible two-clause semantic ABBA dual-parse search.

Both halves carry two discourse beats, but the residual is forbidden to close
at any interior grammar boundary.  That prevents the search from composing a
long output out of independently palindromic sentence pairs.  Agent, tense,
valency, transition, and exact-character state advance together.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import (
    REPEATABLE_FUNCTION_WORDS,
    mechanical_admission_checks,
    normalize_letters,
)
from llm_palindrome.dual_parse import word_residual_search


ID = "two-clause-irreducible-dual-parse-20260921"
OUT = ROOT / "runs" / f"{ID}.json"

DET = ("a", "an", "no", "one", "the")
QUANT = ("all", "many", "no", "several", "some", "the", "three", "two")
AGENT = (
    "actor", "agent", "aide", "artist", "author", "clerk", "doctor",
    "editor", "elder", "farmer", "keeper", "nurse", "painter", "pilot",
    "poet", "porter", "ranger", "reporter", "sailor", "scholar", "singer",
    "soldier", "steward", "surveyor", "teacher", "writer",
)
AGENTS = (
    "actors", "agents", "aides", "artists", "authors", "clerks", "doctors",
    "editors", "elders", "farmers", "keepers", "men", "nurses", "painters",
    "pilots", "poets", "porters", "rangers", "reporters", "sailors",
    "scholars", "singers", "soldiers", "stewards", "surveyors", "teachers",
    "women", "writers",
)
DOCUMENT_VERB_PAST = (
    "archived", "carried", "checked", "copied", "edited", "filed", "kept",
    "marked", "opened", "read", "recorded", "reviewed", "ripped", "saved",
    "sent", "signed", "sorted", "studied", "traced", "wrote",
)
SCENE_VERB_PAST = (
    "carried", "checked", "closed", "found", "guarded", "kept", "left",
    "marked", "opened", "read", "recorded", "returned", "saved", "sent",
    "signed", "sorted", "studied", "waited", "wrote",
)
RESPONSE_VERB_PAST = (
    "admired", "assisted", "encouraged", "guided", "helped", "inspired",
    "praised", "supported", "surprised", "thanked",
)
DOCUMENT = (
    "article", "book", "chart", "essay", "journal", "ledger", "letter",
    "map", "memo", "message", "note", "page", "plan", "poem", "report",
    "route", "song", "story",
)
SCENE_OBJECT = (
    "answer", "bridge", "chart", "door", "gate", "harbor", "inlet",
    "journal", "lantern", "ledger", "letter", "map", "note", "page",
    "path", "plan", "report", "route", "signal", "trail",
)
NAMES = (
    "Ada", "Anna", "Diana", "Eva", "Helena", "Joanna", "Julia", "Lena",
    "Leon", "Mara", "Marina", "Nina", "Noel", "Nora", "Regina", "Rosa",
    "Sara", "Selena", "Tara",
)
CONNECTOR = ("and", "because", "so", "while")
PRONOUN = ("he", "she", "they", "we")


def _allow_partial(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    content = [
        word.casefold() for word in left + right
        if word.casefold() not in REPEATABLE_FUNCTION_WORDS
    ]
    return len(content) == len(set(content)) and all(word != word[::-1] for word in content)


def _left_slots() -> tuple:
    return (
        ("A:determiner", DET),
        ("A:human-agent", AGENT),
        ("A:document-action-past", DOCUMENT_VERB_PAST),
        ("A:object-determiner", DET),
        ("A:document-object", DOCUMENT),
        ("A-to-B:discourse-transition", CONNECTOR),
        ("B:anaphoric-subject", PRONOUN),
        ("B:scene-action-past", SCENE_VERB_PAST),
        ("B:object-determiner", DET),
        ("B:scene-object", SCENE_OBJECT),
    )


def _right_slots() -> tuple:
    return (
        ("B-prime:plural-quantifier", QUANT),
        ("B-prime:human-agent-plural", AGENTS),
        ("B-prime:scene-action-past", SCENE_VERB_PAST),
        ("B-prime:object-determiner", DET),
        ("B-prime:scene-object", SCENE_OBJECT),
        ("B-prime-to-A-prime:discourse-transition", CONNECTOR),
        ("A-prime:subject-determiner", DET),
        ("A-prime:human-agent", AGENT),
        ("A-prime:human-response-past", RESPONSE_VERB_PAST),
        ("A-prime:human-patient", NAMES),
    )


def _audit(text: str) -> dict:
    tape = normalize_letters(text)
    reverse = tape[::-1]
    mismatch = next(
        ((index, tape[index], tape[-1 - index])
         for index in range(len(tape) // 2)
         if tape[index] != tape[-1 - index]),
        None,
    )
    return {
        "letters": len(tape),
        "two_pointer_exact": bool(tape) and mismatch is None,
        "first_mismatch": mismatch,
        "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(reverse.encode()).hexdigest(),
    }


def run() -> dict:
    search = word_residual_search(
        _left_slots(),
        _right_slots(),
        max_states=1_000_000,
        max_results=200,
        allow_partial=_allow_partial,
        reject_intermediate_closure=True,
    )
    rows = []
    for closure in search["results"]:
        rendered = closure["rendered"][:1].upper() + closure["rendered"][1:] + "."
        admission = mechanical_admission_checks(rendered, min_letters=39, max_letters=300)
        rows.append({
            **closure,
            "rendered": rendered,
            "audit": _audit(rendered),
            "mechanical_admission": admission,
            "mechanically_admitted": all(admission.values()),
            "semantic_topology": ["A", "B", "B-prime", "A-prime"],
        })
    admitted = [row for row in rows if row["mechanically_admitted"]]
    return {
        "experiment_id": ID,
        "method": "irreducible two-clause semantic ABBA word-residual product",
        "stats": {
            "states": search["states"],
            "transitions": search["transitions"],
            "cap_reached": search["cap_reached"],
            "intermediate_closure_rejections": search["intermediate_closure_rejections"],
            "exact_closures": len(rows),
            "mechanically_admitted_gt38": len(admitted),
            "deepest_matched_letters": max(
                (row["matched_letters"] for row in search["dead_frontiers"]), default=0
            ),
        },
        "exact_candidates": rows,
        "mechanically_admitted_candidates": admitted,
        "deepest_frontiers": search["dead_frontiers"],
        "novelty_preflight": {
            "novel_algorithm_claim": False,
            "representation_change": "two semantic clauses per half plus hard rejection of every interior exact closure",
            "distinct_from": "mirrored sentence pairs, paragraph unit ABBA, seed wrapping, and post-render clause scoring",
        },
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "finished_tape_reversal": False,
            "post_hoc_repair": False,
            "per_candidate_rlaif": False,
            "catalogue_text": False,
            "reader_gate": "central mechanical admission followed by randomized blinded intact/shuffled controls",
        },
        "reader_packet": [],
        "status": "fresh exact requires blinded readers" if admitted else "no fresh exact irreducible two-clause closure",
        "next_operator": "Compile clause-boundary punctuation and anaphora into a packed character lattice, preserving the no-intermediate-closure invariant.",
    }


if __name__ == "__main__":
    payload = run()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
