"""Bounded semantic-relay repair with two complete SVO clauses.

This is deliberately not a paraphrase of the polar-question or reversible-word
lanes.  Each authored scene contains two independently complete finite clauses:
the second has an anaphoric subject and a consequence in the same event and
setting.  Character obligations are compared while the clauses are emitted at
opposite ends, token by token; the program never obtains an output by reversing
a completed string.

The inventory is intentionally small (eight authored scenes).  It is a
frontier experiment, not a corpus sweep.  Any direct reversible-token pair or
proper multiword palindromic island is rejected before an exact closure could
be admitted.  Programmatic checks still do not establish readability.
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


@dataclass(frozen=True)
class RelayScene:
    """One authored, internally linked pair of complete finite clauses."""

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

    def left_tokens(self) -> tuple[str, ...]:
        return ("the", self.actor, self.left_verb, self.left_determiner,
                self.left_object, self.preposition, self.setting_word)

    def right_tokens(self) -> tuple[str, ...]:
        return (self.pronoun, self.right_verb, self.right_determiner,
                self.right_object)


# Each row was authored as a coherent relay before any character comparison.
# The shared event/setting relation is explicit in the typed record rather than
# inferred from post-hoc word overlap.  Right-hand objects are consequences or
# records of the left-hand action, and the pronoun resolves to the left actor.
SCENES = (
    RelayScene("archive", "archive", "reading_room", "archivist", "she",
               "marks", "a", "ledger", "at", "dusk", "files", "one", "chart"),
    RelayScene("garden", "garden", "garden", "gardener", "she",
               "waters", "a", "cedar", "at", "dawn", "plants", "one", "root"),
    RelayScene("harbor", "repair", "harbor", "sailor", "he",
               "mends", "a", "sail", "at", "sea", "stows", "one", "chart"),
    RelayScene("clinic", "clinic", "ward", "nurse", "she",
               "checks", "a", "pulse", "at", "noon", "files", "one", "report"),
    RelayScene("school", "teaching", "school", "teacher", "she",
               "reads", "a", "poem", "at", "school", "marks", "one", "draft"),
    RelayScene("bakery", "baking", "bakery", "baker", "he",
               "cools", "a", "loaf", "at", "dawn", "serves", "one", "tart"),
    RelayScene("flight", "navigation", "airfield", "pilot", "he",
               "maps", "a", "route", "at", "night", "logs", "one", "chart"),
    RelayScene("theater", "rehearsal", "theater", "actor", "she",
               "reads", "a", "script", "at", "night", "marks", "one", "draft"),
)


def render(scene: RelayScene) -> str:
    """Render two grammatical clauses; punctuation does not affect the tape."""
    left = " ".join(scene.left_tokens()).capitalize()
    right = " ".join(scene.right_tokens())
    return f"{left}; {right}."


def direct_reversed_token_pairs(tokens: tuple[str, ...]) -> list[dict[str, object]]:
    """Find literal token pairs such as ``was``/``saw`` before admission."""
    pairs: list[dict[str, object]] = []
    normalized = tuple(normalize_letters(token) for token in tokens)
    for left, word in enumerate(normalized):
        for right in range(left + 1, len(normalized)):
            if word and word == normalized[right][::-1]:
                pairs.append({"left_index": left, "right_index": right,
                              "left": tokens[left], "right": tokens[right]})
    return pairs


def _consume(
    side: str,
    debt: str,
    left_token: str | None,
    right_build_token: str | None,
) -> tuple[str, str, dict[str, object]] | None:
    """Compare the next token pair without constructing a reversed output."""
    left = (debt if side == "left" else "") + (
        normalize_letters(left_token) if left_token else ""
    )
    right = (debt if side == "right" else "") + (
        normalize_letters(right_build_token)[::-1] if right_build_token else ""
    )
    shared = min(len(left), len(right))
    for index in range(shared):
        if left[index] != right[index]:
            return None
    if len(left) > len(right):
        next_side, next_debt = "left", left[shared:]
    elif len(right) > len(left):
        next_side, next_debt = "right", right[shared:]
    else:
        next_side, next_debt = "", ""
    return next_side, next_debt, {
        "left_stream": left,
        "right_stream": right,
        "matched_characters": shared,
        "next_side": next_side,
        "next_debt": next_debt,
    }


def live_obligation_trace(scene: RelayScene) -> dict[str, object]:
    """Carry exact character debt across word boundaries of complete clauses."""
    left = scene.left_tokens()
    # The right clause must be emitted from its outer edge, but is rendered in
    # ordinary grammar order by ``render``.  This is not a finished-tape flip.
    right_build = tuple(reversed(scene.right_tokens()))
    left_index = right_index = 0
    side = debt = ""
    trace: list[dict[str, object]] = []
    matched = 0
    while left_index < len(left) or right_index < len(right_build):
        left_token = None
        right_token = None
        if side in ("", "right") and left_index < len(left):
            left_token = left[left_index]
            left_index += 1
        if side in ("", "left") and right_index < len(right_build):
            right_token = right_build[right_index]
            right_index += 1
        compared = _consume(side, debt, left_token, right_token)
        if compared is None:
            left_stream = (debt if side == "left" else "") + (
                normalize_letters(left_token) if left_token else ""
            )
            right_stream = (debt if side == "right" else "") + (
                normalize_letters(right_token)[::-1] if right_token else ""
            )
            offset = next((index for index, pair in enumerate(zip(left_stream, right_stream))
                           if pair[0] != pair[1]), 0)
            return {
                "closed": False,
                "matched_outer_characters": matched + offset,
                "first_mismatch": {
                    "left": left_stream[offset] if offset < len(left_stream) else None,
                    "right": right_stream[offset] if offset < len(right_stream) else None,
                    "left_token": left_token,
                    "right_build_token": right_token,
                },
                "trace": trace,
            }
        side, debt, step = compared
        matched += int(step["matched_characters"])
        trace.append({"left_token": left_token, "right_build_token": right_token,
                      **step})
    return {"closed": not side and not debt, "matched_outer_characters": matched,
            "first_mismatch": None, "trace": trace}


def independent_audit(text: str) -> dict[str, object]:
    """Separate pointer and SHA checks; neither is a language-quality score."""
    tape = normalize_letters(text)
    mismatches = [index for index in range(len(tape) // 2)
                  if tape[index] != tape[-index - 1]]
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {
        "letters": len(tape),
        "two_pointer_exact": bool(tape) and not mismatches,
        "mismatch_count": len(mismatches),
        "first_mismatch": mismatches[0] if mismatches else None,
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "sha_equal_under_reversal": forward == reverse,
    }


def scene_row(scene: RelayScene) -> dict[str, object]:
    left = scene.left_tokens()
    right = scene.right_tokens()
    tokens = left + right
    text = render(scene)
    direct_pairs = direct_reversed_token_pairs(tokens)
    hidden_span = has_self_palindromic_proper_multiword_span(tokens)
    preflight = {
        "two_complete_finite_clauses": True,
        "anaphoric_actor_link": {"actor": scene.actor, "pronoun": scene.pronoun},
        "shared_event": scene.event,
        "shared_setting": scene.setting,
        "direct_reversed_token_pairs": direct_pairs,
        "proper_multiword_palindromic_span": hidden_span,
    }
    blocked = bool(direct_pairs or hidden_span)
    trace = live_obligation_trace(scene)
    audit = independent_audit(text)
    checks = mechanical_admission_checks(text, min_letters=39, max_letters=180)
    mechanically_admitted = (
        not blocked
        and trace["closed"]
        and audit["two_pointer_exact"]
        and all(checks.values())
    )
    return {
        "id": scene.identifier,
        "rendered": text,
        "left_clause": list(left),
        "right_clause": list(right),
        "semantic_relay": preflight,
        "live_character_obligation": trace,
        "independent_audit": audit,
        "mechanical_checks": checks,
        "blocked_before_admission": blocked,
        "mechanically_admitted": mechanically_admitted,
        "reader_status": "not_run; exactness and syntax constraints do not certify readability",
    }


def run() -> dict[str, object]:
    rows = [scene_row(scene) for scene in SCENES]
    exact = [row for row in rows if row["independent_audit"]["two_pointer_exact"]]
    admitted = [row for row in rows if row["mechanically_admitted"]]
    best = max(rows, key=lambda row: (
        row["live_character_obligation"]["matched_outer_characters"],
        -row["independent_audit"]["mismatch_count"],
        row["id"],
    ))
    return {
        "experiment_id": "semantic-relay-svo-svo-repair-20260918",
        "status": "completed_no_exact_closure",
        "method": "eight authored semantic-relay SVO/SVO scenes with live cross-token character debt",
        "config": {
            "scene_count": len(SCENES),
            "search": "fixed authored probes; no Cartesian lexical sweep",
            "forbidden": ["polar-question scaffold", "finished-tape reversal",
                          "direct reversed token pairs", "proper multiword palindrome"],
        },
        "stats": {
            "complete_clause_pairs": len(rows),
            "exact_closures": len(exact),
            "mechanically_admitted": len(admitted),
            "reader_eligible": 0,
        },
        "best_frontier": best,
        "rendered_candidates": rows,
        "provenance": {
            "material": "fresh authored typed event inventory",
            "catalogue_text_imported": False,
            "seed_or_polar_question_used": False,
            "finished_tape_reversed": False,
            "independent_validator": "two-pointer scan plus forward/reverse SHA-256",
            "human_readability_certified": False,
        },
        "next_repair": (
            "Preserve this two-clause relay grammar and add exactly one authored "
            "right-hand consequence-object alternative whose reverse suffix matches "
            "the best frontier's first mismatch; keep the direct-token and hidden-span "
            "rejections hard rather than relaxing them."
        ),
        "reader_gate": "closed until a fresh exact, mechanically admitted row exists",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    payload = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], indent=2))


if __name__ == "__main__":
    main()
