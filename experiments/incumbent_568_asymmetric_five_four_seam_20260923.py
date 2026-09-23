#!/usr/bin/env python3
"""Bounded, asymmetric 5-word/4-word seam probes against the 568 tape.

This is an obstruction experiment, not a new incumbent.  It fixes eight
authored scene pairs *before* looking at their character obligations.  The
left and right clauses have unequal word counts and are walked from opposite
outer character cursors.  A pair may be inserted at the two recorded partial
word cuts only if every live obligation closes; no output is made by reversing
a completed clause or by using a word-reversal catalogue.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import (  # noqa: E402
    has_repeated_nontrivial_unit,
    has_self_palindromic_proper_multiword_span,
    normalize_letters,
)
from llm_palindrome.validator import is_palindrome  # noqa: E402


PARENT_REL = "runs/incumbent-560-outer-causal-scene-20261002.json"
PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
OUTPUT_REL = "runs/incumbent-568-asymmetric-five-four-seam-20260923.json"
EXCLUDED_SELF_PATHS = frozenset({
    "experiments/incumbent_568_asymmetric_five_four_seam_20260923.py",
    "tests/test_incumbent_568_asymmetric_five_four_seam_20260923.py",
    "docs/INCUMBENT-568-ASYMMETRIC-FIVE-FOUR-SEAM-20260923.md",
    OUTPUT_REL,
})

# Zero-width normalized cuts inside the mutually mirrored source words.
LEFT_CUT = 99
RIGHT_CUT = 469


@dataclass(frozen=True)
class ScenePair:
    """A small authored semantic scene with non-isomorphic tokenizations."""

    identifier: str
    event: str
    left: str
    right: str

    def left_words(self) -> tuple[str, ...]:
        return tuple(re.findall(r"[a-z]+", self.left.lower()))

    def right_words(self) -> tuple[str, ...]:
        return tuple(re.findall(r"[a-z]+", self.right.lower()))


# These are source-authored ordinary clauses, not samples from the repository
# palindrome banks.  Each right clause reports, shares, or acts in the same
# everyday setting as the left clause.
PAIRS = (
    ScenePair("bakery", "baking", "A baker cools one tart.",
              "Some diners share data."),
    ScenePair("clinic", "record keeping", "A nurse files one chart.",
              "Some doctors review notes."),
    ScenePair("flight", "navigation", "A pilot maps one route.",
              "Some crews plan flights."),
    ScenePair("harbor", "repair", "The sailor mends one sail.",
              "Some dockers move cargo."),
    ScenePair("workshop", "writing", "An artist drafts one poem.",
              "Some writers edit records."),
    ScenePair("garden", "gardening", "A gardener waters one cedar.",
              "Some visitors praise gardens."),
    ScenePair("theater", "rehearsal", "One actor marks the stage.",
              "Some critics review scenes."),
    ScenePair("school", "teaching", "My teacher reads one poem.",
              "Some pupils share notes."),
)


def sha256(tape: str) -> str:
    return hashlib.sha256(tape.encode("ascii")).hexdigest()


def positions(text: str) -> list[int]:
    return [index for index, char in enumerate(text) if char.isascii() and char.isalpha()]


def token_owners(text: str, side: str) -> list[dict[str, object]]:
    """Return one owner record per normalized letter in normal render order."""
    owners: list[dict[str, object]] = []
    for token_index, match in enumerate(re.finditer(r"[A-Za-z]+", text)):
        token = match.group().lower()
        for char_index, char in enumerate(token):
            owners.append({
                "side": side,
                "token_index": token_index,
                "token": token,
                "token_char_index": char_index,
                "letter": char,
            })
    return owners


def direct_cross_token_reversals(pair: ScenePair) -> list[dict[str, str]]:
    """Hard-reject literal left/right reverse tokens; length-one words ignored."""
    rejected: list[dict[str, str]] = []
    for left in pair.left_words():
        for right in pair.right_words():
            if len(left) > 1 and left == right[::-1]:
                rejected.append({"left": left, "right": right})
    return rejected


def self_palindromic_tokens(words: tuple[str, ...]) -> list[str]:
    return [word for word in words if len(word) > 1 and word == word[::-1]]


def live_obligation_trace(pair: ScenePair) -> dict[str, object]:
    """Consume authored clauses with opposite, independent character cursors.

    ``right_cursor`` begins at the final rendered character and only moves
    inward.  The function never obtains ``right`` by flipping a completed
    string; its trace makes every source owner and residual obligation public.
    """
    left = token_owners(pair.left, "left")
    right = token_owners(pair.right, "right")
    left_cursor, right_cursor = 0, len(right) - 1
    matched: list[dict[str, object]] = []
    while left_cursor < len(left) and right_cursor >= 0:
        left_owner, right_owner = left[left_cursor], right[right_cursor]
        if left_owner["letter"] != right_owner["letter"]:
            return {
                "closed": False,
                "matched_outer_characters": len(matched),
                "left_cursor": left_cursor,
                "right_cursor": right_cursor,
                "first_mismatch": {"left": left_owner, "right": right_owner},
                "residual": {
                    "side": "left" if len(left) - left_cursor > right_cursor + 1 else "right",
                    "left_remaining": len(left) - left_cursor,
                    "right_remaining": right_cursor + 1,
                },
                "matched_owners": matched,
            }
        matched.append({"left": left_owner, "right": right_owner})
        left_cursor += 1
        right_cursor -= 1
    closed = left_cursor == len(left) and right_cursor == -1
    return {
        "closed": closed,
        "matched_outer_characters": len(matched),
        "left_cursor": left_cursor,
        "right_cursor": right_cursor,
        "first_mismatch": None,
        "residual": {
            "side": None if closed else ("left" if left_cursor < len(left) else "right"),
            "left_remaining": len(left) - left_cursor,
            "right_remaining": right_cursor + 1,
        },
        "matched_owners": matched,
    }


def independent_pointer_audit(left: str, right: str) -> dict[str, object]:
    """A second normalizer and pointer walk, separate from the live trace."""
    left_tape = "".join(char.lower() for char in left if char.isascii() and char.isalpha())
    right_tape = "".join(char.lower() for char in right if char.isascii() and char.isalpha())
    pairs = min(len(left_tape), len(right_tape))
    mismatch = next((
        index for index in range(pairs)
        if left_tape[index] != right_tape[-index - 1]
    ), None)
    equation_exact = len(left_tape) == len(right_tape) and mismatch is None
    return {
        "left_letters": len(left_tape),
        "right_letters": len(right_tape),
        "first_mismatch_offset": mismatch,
        "equation_exact": equation_exact,
        "left_sha256": sha256(left_tape),
        "right_reversed_sha256": sha256(right_tape[::-1]),
        "sha_equal_under_equation": sha256(left_tape) == sha256(right_tape[::-1]),
    }


def source_seam(parent_text: str) -> dict[str, object]:
    letter_positions = positions(parent_text)
    parent_tape = normalize_letters(parent_text)
    assert len(parent_tape) == 568 and parent_tape == parent_tape[::-1]
    # The letter immediately before a left cut is mirrored by the first
    # letter at the right cut, and vice versa.
    assert parent_tape[LEFT_CUT - 1] == parent_tape[RIGHT_CUT]
    assert parent_tape[LEFT_CUT] == parent_tape[RIGHT_CUT - 1]
    left_raw, right_raw = letter_positions[LEFT_CUT], letter_positions[RIGHT_CUT]
    assert parent_text[left_raw - 1:left_raw + 1].lower() == "li"
    assert parent_text[right_raw - 1:right_raw + 1].lower() == "il"
    return {
        "parent_normalized_cuts": [[LEFT_CUT, LEFT_CUT], [RIGHT_CUT, RIGHT_CUT]],
        "left_owner": {
            "surface_word": "delivers", "surface_cut": "del|ivers", "letter_cut": "3/8",
            "normalized_cursor": LEFT_CUT, "raw_cursor": left_raw,
        },
        "right_owner": {
            "surface_word": "reviled", "surface_cut": "revi|led", "letter_cut": "4/7",
            "normalized_cursor": RIGHT_CUT, "raw_cursor": right_raw,
        },
        "insertion_equation": (
            "T[:99] + L + T[99:469] + R + T[469:] is exact only when "
            "L = reverse(R)"
        ),
    }


def baseline_novelty_check() -> dict[str, object]:
    """Check fixed source phrases against pre-existing tracked text only."""
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, check=True,
                             text=True, capture_output=True).stdout.splitlines()
    corpus = "\n".join(
        (ROOT / path).read_text(errors="ignore")
        for path in tracked
        if path not in EXCLUDED_SELF_PATHS and (ROOT / path).is_file()
    ).casefold()
    phrases = [
        {"side": side, "phrase": surface.casefold().rstrip(".")}
        for pair in PAIRS
        for side, surface in (("left", pair.left), ("right", pair.right))
    ]
    hits = [item for item in phrases if item["phrase"] in corpus]
    return {
        "scope": "tracked repository text excluding this experiment's four paths",
        "checked_surface_phrases": phrases,
        "exact_surface_hits": hits,
        "passed": not hits,
    }


def row(pair: ScenePair) -> dict[str, object]:
    left_words, right_words = pair.left_words(), pair.right_words()
    all_words = left_words + right_words
    direct = direct_cross_token_reversals(pair)
    self_tokens = self_palindromic_tokens(all_words)
    repeated = has_repeated_nontrivial_unit(all_words)
    hidden = has_self_palindromic_proper_multiword_span(all_words)
    trace = live_obligation_trace(pair)
    audit = independent_pointer_audit(pair.left, pair.right)
    assert len(left_words) == 5 and len(right_words) == 4
    assert trace["matched_outer_characters"] == (
        audit["first_mismatch_offset"] if audit["first_mismatch_offset"] is not None
        else min(audit["left_letters"], audit["right_letters"])
    )
    return {
        "id": pair.identifier,
        "event_relation": pair.event,
        "left_rendered": pair.left,
        "right_rendered": pair.right,
        "word_segmentation": {"left": list(left_words), "right": list(right_words),
                              "shape": "5 words <-> 4 words"},
        "left_tape": normalize_letters(pair.left),
        "right_tape": normalize_letters(pair.right),
        "novelty_and_shortcut_gate": {
            "direct_cross_token_reversals": direct,
            "self_palindromic_tokens": self_tokens,
            "repeated_nontrivial_multiword_unit": repeated,
            "proper_multiword_palindromic_span": hidden,
            "passed": not (direct or self_tokens or repeated or hidden),
        },
        "live_character_obligation": trace,
        "independent_pointer_audit": audit,
        "exact_insertion_possible": bool(audit["equation_exact"] and not direct and not self_tokens
                                           and not repeated and not hidden),
        "reader_risk": (
            "The two finite clauses are individually ordinary, but their scene relation is "
            "only local; their immediate character mismatch means seam punctuation cannot repair them."
        ),
    }


def main() -> None:
    parent = json.loads((ROOT / PARENT_REL).read_text())
    parent_text = parent["rows"][0]["rendered"]
    parent_tape = normalize_letters(parent_text)
    assert sha256(parent_tape) == PARENT_SHA256 and is_palindrome(parent_text)
    seam = source_seam(parent_text)
    rows = [row(pair) for pair in PAIRS]
    novelty = baseline_novelty_check()
    assert novelty["passed"]
    assert not any(item["exact_insertion_possible"] for item in rows)
    best = max(rows, key=lambda item: (
        item["live_character_obligation"]["matched_outer_characters"],
        -abs(len(item["left_tape"]) - len(item["right_tape"])), item["id"],
    ))
    artifact = {
        "experiment_id": "incumbent-568-asymmetric-five-four-seam-20260923",
        "status": "completed_no_exact_asymmetric_closure",
        "claim_boundary": (
            "No 588-character child is rendered or admitted. The 568 parent supplies only "
            "a symmetric coordinate system and remains outside this experiment's readability claim."
        ),
        "parent": {"artifact": PARENT_REL, "normalized_letter_length": len(parent_tape),
                   "normalized_letter_sha256": PARENT_SHA256},
        "source_provenance": seam,
        "seam_viability": {
            "left_context": "Aidan del|ivers maps.",
            "right_context": "Spam's revi|led, Nadia.",
            "standalone_clause_insertion": False,
            "reason": (
                "A delimiter around an inserted clause leaves the source fragments 'del', 'ivers', "
                "'revi', and 'led' at the seam. Avoiding those fragments requires letters to fuse "
                "across inherited word owners, which is the already-archived boundary-shift family."
            ),
            "archive_collision": [
                "experiments/live_boundary_shift_grammar_20260920.py",
                "experiments/morphology_crossword_transducer_20260916.py",
            ],
        },
        "method": {
            "inventory": "eight authored typed scene pairs; no Cartesian sweep",
            "topology": "partial-word 568 lineage seam, 5-word left clause <-> 4-word right clause",
            "construction_rule": "fixed authored clauses, live opposing cursors, then independent pointer+SHA audit",
            "prohibited": ["archived 38-letter control", "literal reversed token pairs", "self-palindromic tokens", "repeated multiword units", "catalogue text"],
        },
        "novelty_preflight": novelty,
        "rows": rows,
        "obstruction": {
            "best_clean_probe": best["id"],
            "rendered_pair": [best["left_rendered"], best["right_rendered"]],
            "equation": f"{best['left_tape']} != reverse({best['right_tape']})",
            "matched_outer_characters": best["live_character_obligation"]["matched_outer_characters"],
            "first_mismatch": best["live_character_obligation"]["first_mismatch"],
            "lengths": [len(best["left_tape"]), len(best["right_tape"])],
            "conclusion": "No row supplies the L = reverse(R) obligation required by the selected 568 partial-word seam.",
        },
        "next_action": {
            "parent": PARENT_REL,
            "seam": [[148, 163], [405, 420]],
            "operator": (
                "replace the actual mirrored clause window with independently authored finite-event clauses "
                "under a 5-word/4-word live character equation; require different word-boundary signatures "
                "and reject every cross-side reversed-token pair before rendering"
            ),
            "reason": (
                "The [99,99]/[469,469] seam cannot accept complete clauses without splitting inherited "
                "words, while generic character-splice repair duplicates archived operators. The selected "
                "clean-boundary seam has a previously rejected tokenwise child, so the new test must change "
                "the segmentation and equation method, not replay that child."
            ),
        },
    }
    (ROOT / OUTPUT_REL).write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({"artifact": OUTPUT_REL, "status": artifact["status"],
                      "best": artifact["obstruction"], "parent_sha256": PARENT_SHA256}, indent=2))


if __name__ == "__main__":
    main()
