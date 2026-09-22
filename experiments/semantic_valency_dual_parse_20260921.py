"""Semantic valency and exact residual search in one finite product.

The Brown POS product demonstrated that syntax-only states still close on
short repeated controls.  This successor narrows lexical choices by semantic
role before the character walk: human agents take agreement-bearing verbs,
document verbs take document objects, and response verbs take human objects.
The historical 38-letter item is a recovery control, never a new candidate.
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


ID = "semantic-valency-dual-parse-20260921"
OUT = ROOT / "runs" / f"{ID}.json"
SEED = "an aide rips nine memos some men inspire Diana"

DETERMINERS = ("a", "an", "no", "one", "the")
HUMAN_SINGULAR = (
    "actor", "agent", "aide", "artist", "author", "clerk", "doctor",
    "editor", "elder", "farmer", "keeper", "nurse", "painter", "pilot",
    "poet", "porter", "ranger", "reporter", "sailor", "scholar",
    "singer", "soldier", "steward", "surveyor", "teacher", "writer",
)
HUMAN_PLURAL = (
    "actors", "agents", "aides", "artists", "authors", "clerks",
    "doctors", "editors", "elders", "farmers", "keepers", "men",
    "nurses", "painters", "pilots", "poets", "porters", "rangers",
    "reporters", "sailors", "scholars", "singers", "soldiers",
    "stewards", "surveyors", "teachers", "women", "writers",
)
DOCUMENT_VERB_3SG = (
    "archives", "carries", "checks", "copies", "edits", "files", "keeps",
    "marks", "opens", "reads", "records", "reviews", "rips", "saves",
    "sends", "signs", "sorts", "studies", "traces", "writes",
)
DOCUMENT_OBJECT = (
    "article", "articles", "book", "books", "chart", "charts", "essay",
    "essays", "journal", "journals", "ledger", "ledgers", "letter",
    "letters", "map", "maps", "memo", "memos", "message", "messages",
    "note", "notes", "page", "pages", "plan", "plans", "poem", "poems",
    "report", "reports", "route", "routes", "song", "songs", "story",
    "stories",
)
QUANTITY = (
    "a", "an", "five", "four", "many", "nine", "one", "several", "some",
    "ten", "the", "three", "two",
)
PLURAL_QUANTIFIER = (
    "all", "many", "more", "most", "no", "several", "some", "the",
    "these", "those", "three", "two",
)
HUMAN_RESPONSE_VERB = (
    "admire", "assist", "encourage", "guide", "help", "inspire", "praise",
    "support", "surprise", "thank",
)
HUMAN_NAMES = (
    "Ada", "Anna", "Diana", "Eva", "Helena", "Joanna", "Julia", "Lena",
    "Leon", "Mara", "Marina", "Nina", "Noel", "Nora", "Regina", "Rosa",
    "Sara", "Selena", "Tara",
)


def _unique_content(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    content = [
        word.casefold() for word in left + right
        if word.casefold() not in REPEATABLE_FUNCTION_WORDS
    ]
    return len(content) == len(set(content)) and all(word != word[::-1] for word in content)


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


def _document_frame() -> tuple[tuple, tuple]:
    left = (
        ("A:determiner", DETERMINERS),
        ("A:human-agent-singular", HUMAN_SINGULAR),
        ("B:document-action-3sg", DOCUMENT_VERB_3SG),
        ("B:quantity", QUANTITY),
        ("B:document-object", DOCUMENT_OBJECT),
    )
    right = (
        ("B-prime:plural-quantifier", PLURAL_QUANTIFIER),
        ("B-prime:human-agent-plural", HUMAN_PLURAL),
        ("B-prime:human-response-base", HUMAN_RESPONSE_VERB),
        ("A-prime:human-patient", HUMAN_NAMES),
    )
    return left, right


def _fresh_endpoint_frames() -> list[tuple[str, tuple, tuple]]:
    """Small, grammatically distinct endpoint families; none wraps the seed."""
    objects = ("answer", "book", "chart", "evidence", "letter", "map", "note", "plan", "report", "story")
    base_verbs = ("check", "copy", "edit", "file", "find", "keep", "mark", "open", "read", "record", "save", "send", "sort", "study", "write")
    return [
        (
            "we/new",
            (("A:pronoun", ("we",)), ("B:base-verb", base_verbs),
             ("B:determiner", DETERMINERS), ("B:object", objects)),
            (("B-prime:determiner", DETERMINERS), ("B-prime:human-subject", HUMAN_SINGULAR),
             ("A-prime:copula", ("is", "was", "became", "looks", "seems")),
             ("A-prime:property", ("new",))),
        ),
        (
            "as-I/visa",
            (("A:subordinator", ("as",)), ("A:pronoun", ("i",)),
             ("B:base-verb", base_verbs + ("value", "verify", "view", "visit")),
             ("B:determiner", DETERMINERS), ("B:object", objects)),
            (("B-prime:determiner", DETERMINERS), ("B-prime:human-subject", HUMAN_SINGULAR),
             ("B-prime:possession-verb", ("carries", "checks", "finds", "keeps", "needs", "reviews", "uses")),
             ("A-prime:determiner", ("a", "the")), ("A-prime:document", ("visa",))),
        ),
    ]


def run() -> dict:
    searches = [("document-control", *_document_frame()), *_fresh_endpoint_frames()]
    rows: list[dict] = []
    frontiers: list[dict] = []
    stats = {"frame_pairs": len(searches), "states": 0, "transitions": 0}
    for frame, left, right in searches:
        result = word_residual_search(
            left,
            right,
            max_states=500_000,
            max_results=100,
            allow_partial=_unique_content,
        )
        stats["states"] += result["states"]
        stats["transitions"] += result["transitions"]
        for frontier in result["dead_frontiers"]:
            frontiers.append({**frontier, "frame": frame})
        for closure in result["results"]:
            rendered = closure["rendered"][:1].upper() + closure["rendered"][1:] + "."
            audit = _audit(rendered)
            admission = mechanical_admission_checks(rendered, min_letters=39, max_letters=220)
            rows.append({
                **closure,
                "frame": frame,
                "rendered": rendered,
                "audit": audit,
                "mechanical_admission": admission,
                "mechanically_admitted": all(admission.values()),
                "historical_recovery_control": normalize_letters(rendered) == normalize_letters(SEED),
            })

    frontiers.sort(key=lambda row: (-row["matched_letters"], len(row["residual"])))
    recovery = [row for row in rows if row["historical_recovery_control"]]
    fresh = [row for row in rows if row["mechanically_admitted"] and not row["historical_recovery_control"]]
    stats.update({
        "exact_closures": len(rows),
        "recovery_controls": len(recovery),
        "fresh_mechanically_admitted_gt38": len(fresh),
        "deepest_matched_letters": frontiers[0]["matched_letters"] if frontiers else 0,
    })
    return {
        "experiment_id": ID,
        "method": "semantic-valency slot product with live exact word residual and partial anti-shortcut gate",
        "stats": stats,
        "recovery_controls": recovery,
        "fresh_exact_candidates": fresh,
        "all_exact_closures": rows,
        "deepest_frontiers": frontiers[:32],
        "novelty_preflight": {
            "novel_algorithm_claim": False,
            "representation_change": "semantic agent/action/object types are compiled into the corrected dual-parse residual state",
            "overlaps_acknowledged": [
                "semantic-valency-attachment-solver-20260917",
                "semantic-valency-trie-nfa-multichar-20260917",
                "brown-pos-dual-parse-search-20260921",
            ],
        },
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "per_candidate_rlaif": False,
            "catalogue_text": False,
            "seed_policy": "recovery control only; never promoted as fresh output or wrapped",
            "reader_gate": "fresh central admission first, then randomized blinded intact/shuffled study",
        },
        "reader_packet": [],
        "status": "fresh exact requires blinded readers" if fresh else "no fresh exact closure; recovery invariant passed",
        "next_operator": "Compile two-clause discourse transitions into the same valency/residual state; do not widen these lexical banks.",
    }


if __name__ == "__main__":
    payload = run()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
