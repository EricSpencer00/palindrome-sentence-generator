"""Targeted outer-edge repair for the semantic-relay grammar.

The parent semantic-relay lane failed at its first edge: ``the`` was paired
with the reverse of a consequence object such as ``chart``.  This repair
changes exactly that boundary by authoring vowel-initial subjects with ``an``
and terminal ``arena`` objects, then carries the resulting residual through
the same complete two-clause grammar.  It is eight boundary probes, not a
lexical sweep, and preserves the hard reversed-token/hidden-span filters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import (  # noqa: E402
    has_self_palindromic_proper_multiword_span,
    mechanical_admission_checks,
    normalize_letters,
)
from experiments.semantic_relay_svo_svo_repair_20260918 import (
    direct_reversed_token_pairs,
    live_obligation_trace,
)


@dataclass(frozen=True)
class EdgeRepairScene:
    identifier: str
    event: str
    setting: str
    actor: str
    pronoun: str
    left_verb: str
    left_determiner: str
    left_object: str
    preposition: str
    setting_word: str
    right_verb: str
    right_determiner: str
    right_object: str
    subject_determiner: str = "an"

    def left_tokens(self) -> tuple[str, ...]:
        return (self.subject_determiner, self.actor, self.left_verb,
                self.left_determiner, self.left_object, self.preposition,
                self.setting_word)

    def right_tokens(self) -> tuple[str, ...]:
        return (self.pronoun, self.right_verb, self.right_determiner,
                self.right_object)


# Each probe changes only the outer determiner/terminal-object boundary from
# the parent scene.  The right verb makes the authored ``an arena`` phrase
# grammatical; the event and setting link remain explicit.
SCENES = (
    EdgeRepairScene("archive-edge", "archive", "reading_room", "archivist", "she",
                    "marks", "a", "ledger", "at", "dusk", "maps", "an", "arena"),
    EdgeRepairScene("artist-edge", "art", "studio", "artist", "she",
                    "paints", "a", "portrait", "at", "dawn", "sees", "an", "arena"),
    EdgeRepairScene("editor-edge", "archive", "office", "editor", "she",
                    "reads", "a", "report", "at", "noon", "maps", "an", "arena"),
    EdgeRepairScene("engineer-edge", "repair", "station", "engineer", "she",
                    "tests", "a", "signal", "at", "night", "sees", "an", "arena"),
    EdgeRepairScene("author-edge", "writing", "study", "author", "she",
                    "edits", "a", "draft", "at", "dawn", "maps", "an", "arena"),
    EdgeRepairScene("actor-edge", "theater", "stage", "actor", "she",
                    "reads", "a", "script", "at", "night", "sees", "an", "arena"),
    EdgeRepairScene("aide-edge", "office", "hall", "aide", "she",
                    "files", "a", "memo", "at", "dusk", "maps", "an", "arena"),
    EdgeRepairScene("analyst-edge", "research", "lab", "analyst", "she",
                    "checks", "a", "record", "at", "noon", "sees", "an", "arena"),
)


def render(scene: EdgeRepairScene) -> str:
    left = " ".join(scene.left_tokens()).capitalize()
    right = " ".join(scene.right_tokens())
    return f"{left}; {right}."


def audit(text: str) -> dict[str, object]:
    tape = normalize_letters(text)
    reverse = tape[::-1]
    mismatches = [i for i in range(len(tape) // 2) if tape[i] != tape[-1 - i]]
    return {
        "letters": len(tape),
        "two_pointer_exact": bool(tape) and not mismatches,
        "mismatch_count": len(mismatches),
        "first_mismatch": mismatches[0] if mismatches else None,
        "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(reverse.encode()).hexdigest(),
        "sha_equal_under_reversal": hashlib.sha256(tape.encode()).hexdigest()
        == hashlib.sha256(reverse.encode()).hexdigest(),
    }


def row(scene: EdgeRepairScene) -> dict[str, object]:
    text = render(scene)
    tokens = scene.left_tokens() + scene.right_tokens()
    trace = live_obligation_trace(scene)
    checks = mechanical_admission_checks(text, min_letters=39, max_letters=180)
    return {
        "id": scene.identifier,
        "rendered": text,
        "provenance": {
            "operator": "outer-edge determiner plus terminal consequence-object repair",
            "parent_lane": "semantic-relay-svo-svo-repair-20260918",
            "catalogue_text_imported": False,
            "finished_tape_reversed": False,
            "authored_boundary": {"subject_determiner": scene.subject_determiner,
                                  "terminal_object": scene.right_object},
        },
        "live_character_obligation": trace,
        "independent_audit": audit(text),
        "direct_reversed_token_pairs": direct_reversed_token_pairs(tokens),
        "proper_multiword_palindromic_span": has_self_palindromic_proper_multiword_span(tokens),
        "mechanical_checks": checks,
        "mechanically_admitted": bool(trace["closed"] and audit(text)["two_pointer_exact"]
                                       and all(checks.values())),
        "reader_status": "not_run; no exact survivor entered the reader gate",
    }


def run() -> dict[str, object]:
    rows = [row(scene) for scene in SCENES]
    exact = [r for r in rows if r["independent_audit"]["two_pointer_exact"]]
    admitted = [r for r in rows if r["mechanically_admitted"]]
    best = max(rows, key=lambda r: (
        r["live_character_obligation"]["matched_outer_characters"],
        -r["independent_audit"]["mismatch_count"],
        r["id"],
    ))
    return {
        "experiment_id": "semantic-relay-outer-edge-repair-20260918",
        "signature": "semantic-relay|outer-edge-determiner-object|live-residual|independent-audit",
        "config": {"probe_count": len(rows), "search": "eight targeted authored boundary probes"},
        "rendered_candidates": rows,
        "stats": {"probes": len(rows), "exact": len(exact),
                   "mechanically_admitted": len(admitted), "reader_eligible": 0,
                   "best_matched_outer_characters": max(
                       r["live_character_obligation"]["matched_outer_characters"] for r in rows)},
        "best_frontier": best,
        "provenance": {"independent_validator": "two-pointer normalized tape plus forward/reverse SHA-256",
                       "human_readability_certified": False},
        "next_repair": "co-design the first subject lexical token with the reverse terminal object suffix; the determiner-only repair advances two characters but dies at the actor boundary",
        "reader_gate": "closed until an exact row clears all strict mechanical checks",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], indent=2))


if __name__ == "__main__":
    main()
