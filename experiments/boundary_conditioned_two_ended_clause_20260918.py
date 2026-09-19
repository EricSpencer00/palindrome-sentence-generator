#!/usr/bin/env python3
"""Boundary-conditioned two-ended clause construction.

The previous phrase products were invalid diagnostics: their outer edge
alphabets were incompatible before any center or semantic state was reached.
This lane first freezes 24 freshly authored opening/ending constituent pairs
with at least a four-character live match, then extends both grammatical sides
from that boundary with a small residual search.  Neither half is a complete
clause during construction; the midpoint may fall inside a word.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/boundary-conditioned-two-ended-clause-20260918.json"
EXPERIMENT = "boundary-conditioned-two-ended-clause-20260918"
MAX_EXPANSIONS_PER_PAIR = 200

# Fresh boundary constituents.  The right strings are ordinary clause endings;
# their normalized reverse is the live obligation exposed to the left opener.
LEFT_OPENINGS = [
    "Not a", "Not an", "No one", "A quiet", "A patient", "The old",
    "The kind", "One careful", "In a", "At a", "After the", "Before the",
    "All the", "Some of", "We can", "She will",
]
RIGHT_ENDINGS = [
    "weighs a ton", "costs a ton", "holds a ton", "writes a ton",
    "reads a ton", "needs a ton", "makes a ton", "moves a ton",
    "finds a ton", "keeps a ton", "the old prison", "the quiet garden",
    "a folded chart", "the patient guide", "a small stone", "the river bank",
    "a bright idea", "the village gate", "an open door", "the winter road",
    "a silver line", "the harbor wall", "an empty room", "the orchard path",
]

# Continuations are typed but intentionally small.  A state is only expanded
# when the next exposed characters can discharge the mirrored residual.
LEFT_SLOTS = (
    ("ADJ", ["quiet", "patient", "careful", "young", "kind", "bright"]),
    ("NOUN", ["keeper", "baker", "teacher", "pilot", "gardener", "mason", "scribe"]),
    ("VERB", ["marks", "opens", "carries", "copies", "guards", "writes", "keeps"]),
    ("DET", ["a", "the", "one"]),
    ("NOUN", ["map", "letter", "gate", "chart", "stone", "garden", "note"]),
)
RIGHT_REVERSE_SLOTS = (
    ("NOUN", ["keeper", "baker", "teacher", "pilot", "gardener", "mason", "scribe", "prison", "garden", "chart", "stone", "bank", "idea", "gate", "door", "road", "line", "wall", "room", "path"]),
    ("DET", ["a", "an", "the", "one"]),
    ("PREP", ["near", "under", "beside", "toward", "across", "in", "on", "by", "at"]),
    ("NOUN", ["keeper", "baker", "teacher", "pilot", "gardener", "mason", "scribe", "friend", "child", "captain"]),
    ("VERB", ["marks", "opens", "carries", "copies", "guards", "writes", "keeps", "reads"]),
    ("NOUN", ["keeper", "baker", "teacher", "pilot", "gardener", "mason", "scribe", "friend", "child", "captain"]),
    ("ADJ", ["quiet", "patient", "careful", "young", "kind", "bright"]),
    ("DET", ["a", "an", "the", "one"]),
)


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict:
    tape = letters(text)
    mismatches = [
        {"left": i, "right": len(tape) - 1 - i, "actual": tape[i], "expected": tape[-1 - i]}
        for i in range(len(tape) // 2)
        if tape[i] != tape[-1 - i]
    ]
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {
        "letters": len(tape),
        "exact": bool(tape) and not mismatches,
        "two_pointer_exact": bool(tape) and not mismatches,
        "mismatch_count": len(mismatches),
        "first_mismatch": mismatches[0] if mismatches else None,
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "sha256_equal": forward == reverse,
    }


def matched_prefix(left: str, right: str) -> int:
    n = min(len(left), len(right))
    i = 0
    while i < n and left[i] == right[i]:
        i += 1
    return i


def boundary_pairs() -> list[dict]:
    pairs = []
    for left, right in itertools.product(LEFT_OPENINGS, RIGHT_ENDINGS):
        left_tape = letters(left)
        right_obligation = letters(right)[::-1]
        matched = matched_prefix(left_tape, right_obligation)
        if matched < 4:
            continue
        pairs.append({
            "left_opening": left,
            "right_ending": right,
            "left_tape": left_tape,
            "right_obligation": right_obligation,
            "matched_characters": matched,
        })
    # Freeze a small, deterministic frontier; no later lexical expansion is
    # allowed to inflate this diagnostic into a sweep.
    return pairs[:24]


def advance(left_pending: str, right_pending: str) -> tuple[str, str, int] | None:
    matched = 0
    while left_pending and right_pending:
        if left_pending[0] != right_pending[0]:
            return None
        left_pending = left_pending[1:]
        right_pending = right_pending[1:]
        matched += 1
    return left_pending, right_pending, matched


def mismatch_detail(left: str, right: str) -> dict | None:
    for index, (actual, expected) in enumerate(zip(left, right)):
        if actual != expected:
            return {"offset": index, "actual": actual, "expected": expected}
    return None


def expand_pair(pair: dict) -> dict:
    # The fixed boundary consumes the same characters on both sides; the
    # continuation search starts at its first residual character.
    initial_left = pair["left_tape"]
    initial_right = pair["right_obligation"]
    consumed = pair["matched_characters"]
    lp = initial_left[consumed:]
    rp = initial_right[consumed:]
    # Preserve pending suffixes after the matched boundary; they are never
    # silently discarded when a boundary constituent is longer than its mate.
    states = [{
        "left_slots": 0,
        "right_slots": 0,
        "left_pending": lp,
        "right_pending": rp,
        "left_words": pair["left_opening"].split(),
        "right_reverse_words": list(reversed(pair["right_ending"].split())),
        "matched": consumed,
    }]
    visited = set()
    terminal = []
    expansions = 0
    first_extension_failure = None
    matched_lengths = []
    milestones = {4: 0, 8: 0, 12: 0, 16: 0}
    while states and expansions < MAX_EXPANSIONS_PER_PAIR:
        state = states.pop(0)
        key = (
            state["left_slots"], state["right_slots"], state["left_pending"],
            state["right_pending"], tuple(state["left_words"]), tuple(state["right_reverse_words"]),
        )
        if key in visited:
            continue
        visited.add(key)
        expansions += 1
        matched_lengths.append(state["matched"])
        for threshold in milestones:
            if state["matched"] >= threshold:
                milestones[threshold] += 1
        if state["left_slots"] == len(LEFT_SLOTS) and state["right_slots"] == len(RIGHT_REVERSE_SLOTS):
            if not state["left_pending"] and not state["right_pending"]:
                terminal.append(state)
            continue

        # If both sides are between tokens, choose the next token on each side;
        # otherwise extend only the side whose residual is empty.
        left_choices = [(state["left_slots"], "", state["left_slots"] + 1, [])]
        right_choices = [(state["right_slots"], "", state["right_slots"] + 1, [])]
        if not state["left_pending"] and state["left_slots"] < len(LEFT_SLOTS):
            typ, vocab = LEFT_SLOTS[state["left_slots"]]
            left_choices = [(state["left_slots"], word, state["left_slots"] + 1, [word]) for word in vocab]
        if not state["right_pending"] and state["right_slots"] < len(RIGHT_REVERSE_SLOTS):
            typ, vocab = RIGHT_REVERSE_SLOTS[state["right_slots"]]
            right_choices = [(state["right_slots"], word, state["right_slots"] + 1, [word]) for word in vocab]
        for _, left_word, next_left, left_add in left_choices:
            for _, right_word, next_right, right_add in right_choices:
                if not left_word and not right_word:
                    continue
                nl = state["left_pending"] + letters(left_word)
                nr = state["right_pending"] + letters(right_word)
                checked = advance(nl, nr)
                if checked is None:
                    if first_extension_failure is None:
                        first_extension_failure = {
                            "left_slot": LEFT_SLOTS[state["left_slots"]][0] if state["left_slots"] < len(LEFT_SLOTS) else None,
                            "right_slot": RIGHT_REVERSE_SLOTS[state["right_slots"]][0] if state["right_slots"] < len(RIGHT_REVERSE_SLOTS) else None,
                            "left_pending": nl,
                            "right_pending": nr,
                            "mismatch": mismatch_detail(nl, nr),
                        }
                    continue
                residual_left, residual_right, gained = checked
                states.append({
                    "left_slots": next_left if left_word else state["left_slots"],
                    "right_slots": next_right if right_word else state["right_slots"],
                    "left_pending": residual_left,
                    "right_pending": residual_right,
                    "left_words": state["left_words"] + left_add,
                    "right_reverse_words": state["right_reverse_words"] + right_add,
                    "matched": state["matched"] + gained,
                })

    rows = []
    for state in terminal:
        right_words = list(reversed(state["right_reverse_words"]))
        rendered = " ".join(state["left_words"] + right_words) + "."
        rows.append({"rendered": rendered, "audit": audit(rendered), "state": state})
    return {
        "boundary": pair,
        "expansions": expansions,
        "visited_states": len(visited),
        "milestones": milestones,
        "terminal_rows": rows,
        "first_extension_failure": first_extension_failure,
        "best_matched": max(matched_lengths, default=pair["matched_characters"]),
    }


def main() -> None:
    pairs = boundary_pairs()
    results = [expand_pair(pair) for pair in pairs]
    exact_rows = [row for result in results for row in result["terminal_rows"] if row["audit"]["exact"]]
    controls = [
        "Not a quiet keeper guards a small gate.",
        "The patient baker carries warm loaves home.",
        "No one marks the old chart near the river.",
    ]
    payload = {
        "experiment": EXPERIMENT,
        "method": "boundary-conditioned two-ended clause construction with live residual and typed continuation search",
        "boundary_inventory": {"left_openings": LEFT_OPENINGS, "right_endings": RIGHT_ENDINGS},
        "config": {"boundary_pairs": len(pairs), "max_expansions_per_pair": MAX_EXPANSIONS_PER_PAIR, "midpoint_free_inside_word": True},
        "novelty_preflight": {
            "status": "fresh_authored_boundaries",
            "catalogue_sentence_imported": False,
            "finished_tape_reversal": False,
            "existing_seed_used": False,
            "selection": "only boundary pairs with >=4 live matched characters; no catalogue phrase row is output",
        },
        "stats": {
            "boundary_pairs": len(pairs),
            "survive_4": sum(r["boundary"]["matched_characters"] >= 4 for r in results),
            "survive_8": sum(r["milestones"][8] > 0 for r in results),
            "survive_12": sum(r["milestones"][12] > 0 for r in results),
            "survive_16": sum(r["milestones"][16] > 0 for r in results),
            "expansions": sum(r["expansions"] for r in results),
            "terminal_rows": sum(len(r["terminal_rows"]) for r in results),
            "exact_count": len(exact_rows),
            "admissible_count": 0,
        },
        "boundary_results": results,
        "exact_rows": exact_rows,
        "controls": [{"rendered": text, "audit": audit(text), "provenance": "fresh intact prose control"} for text in controls],
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "independent_audits": ["two-pointer mismatch scan", "forward/reverse SHA-256"],
            "human_readability": "unreviewed",
        },
        "next_repair": "If interior feasibility survives 8 characters, add one typed continuation at the first residual; if it collapses at 4, replace only the boundary ending with a fresh semantic constituent and rerun the frozen 24-pair frontier.",
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))


if __name__ == "__main__":
    main()
