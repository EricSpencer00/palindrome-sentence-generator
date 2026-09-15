"""Let live palindrome residuals cross complete typed clause boundaries.

This is the follow-up to the exhausted single-clause endpoint search.  Each
side is assembled from one or two complete SVO clauses, but exact characters
are still matched outside-in before rendering.  Sentence boundaries carry no
letters and may therefore occur at different mirrored positions.
"""
from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.predicate_subject_bridge_search_20260914 import (
    DETERMINERS,
    NAMES,
    PEOPLE,
    build_pools,
    independent_audit,
    indexed_search,
)
from llm_palindrome.admission import mechanical_admission_checks


def render_side(words: tuple[str, ...], break_after: tuple[int, ...]) -> str:
    clauses = []
    start = 0
    for end in break_after:
        clause = " ".join(words[start:end])
        clauses.append(clause[:1].upper() + clause[1:] + ".")
        start = end
    return " ".join(clauses)


def run(*, state_budget: int = 2_000_000) -> dict:
    left_base, right_base, inventory = build_pools()
    _, _, singular_verbs, objects = left_base
    subjects, plural_verbs, names = right_base
    opening_people = tuple(f"{det} {person}" for det in DETERMINERS for person in PEOPLE)
    terminal_nps = names + tuple(
        f"{det} {noun}" for det in ("a", "an", "the", "my", "our")
        for noun in ("book", "canvas", "door", "gift", "letter", "map", "memo",
                     "note", "plan", "report", "room", "story", "tray")
    )
    one_left = (opening_people, singular_verbs, objects)
    one_right = (subjects, plural_verbs, names)
    two_left = one_left + (subjects, plural_verbs, terminal_nps)
    two_right = (subjects, plural_verbs, terminal_nps) + one_right
    shapes = (
        ("two_left_one_right", two_left, one_right, (3, 6), (3,)),
        ("one_left_two_right", one_left, two_right, (3,), (3, 6)),
        ("two_left_two_right", two_left, two_right, (3, 6), (3, 6)),
    )
    records = []
    searches = []
    for name, left, right, left_breaks, right_breaks in shapes:
        pairs, stats = indexed_search(left, right, state_budget=state_budget)
        searches.append({"shape": name, **stats, "exact": len(pairs)})
        for left_words, right_words in pairs:
            text = render_side(left_words, left_breaks) + " " + render_side(right_words, right_breaks)
            audit = independent_audit(text)
            checks = mechanical_admission_checks(text, min_letters=39, max_letters=240)
            records.append({"shape": name, "text": text, "audit": audit,
                            "mechanical_checks": checks,
                            "mechanically_eligible": all(checks.values()),
                            "reader_status": "human-unreviewed"})
    unique = {row["audit"]["normalized"]: row for row in records}
    exact = list(unique.values())
    return {"status": "multiclause_residual_bridge_complete",
            "config": {"shapes": len(shapes), "state_budget_per_shape": state_budget,
                       "staggered_sentence_boundaries": True,
                       "grammar_during_search": True},
            "inventory": {**inventory, "opening_people": len(opening_people),
                          "terminal_nps": len(terminal_nps)},
            "searches": searches, "exact_records": exact,
            "eligible_closures": [row for row in exact if row["mechanically_eligible"]],
            "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                           "material": "typed lexical products and Brown POS inventories; no complete corpus text"},
            "next_operator_if_empty": "Apply the partial anti-shortcut gate to frontier states, then split the best admissible nonzero residual inside a typed multiword constituent and synthesize both adjacent lexical boundaries jointly.",
            "reader_next": "Novel mechanically eligible outputs require blinded intact-prose versus shuffled-control ratings."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--state-budget", type=int, default=2_000_000)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing to overwrite output")
    result = run(state_budget=args.state_budget)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"states": sum(row["states"] for row in result["searches"]),
                      "exact": len(result["exact_records"]),
                      "eligible": len(result["eligible_closures"])}, indent=2))


if __name__ == "__main__":
    main()
