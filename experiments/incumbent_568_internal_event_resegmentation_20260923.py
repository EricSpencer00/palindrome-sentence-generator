#!/usr/bin/env python3
"""Record exact resegmentations of a previously untouched 568-letter seam.

The left and right clause surfaces are independently composed, then admitted
only if they close the pinned parent's live reflected-character obligation.
This is construction evidence, not a readability certification.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.validator import is_palindrome as project_is_palindrome


PARENT_PATH = ROOT / "runs" / "incumbent-560-outer-causal-scene-20261002.json"
OUT_PATH = ROOT / "runs" / "incumbent-568-internal-event-resegmentation-20260923.json"
PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
PREFLIGHT_REVISION = "961ee718"
OLD_LEFT = "Nadia stops, so Tara rewards Nadia."
OLD_RIGHT = "Aidan's drawer, Aratos, spots Aidan."
LEFT_NORMALIZED_SPAN = (20, 48)
RIGHT_NORMALIZED_SPAN = (520, 548)

PROPOSALS = (
    {
        "id": "internal-event-resegmentation-584",
        "left": "A rat saw Iris. Iris stops a yak. Aidan was a ram.",
        "right": "Mara saw Nadia. Kaya spots Siri. Siri was Tara.",
        "expected_sha256": "895cbb340d071220c99c5bec737bb871d6eb8456427afcd86ff4016cd1a3e819",
        "construction_note": (
            "Six locally complete clauses; the animal/name propositions are not yet tied into one discourse."
        ),
    },
    {
        "id": "internal-report-response-resegmentation-586",
        "left": "Iris notes a rat. Ari maps a yak. Sara spots a hen.",
        "right": "Neha stops Aras. Kaya, spam Ira. Tara, set on Siri.",
        "expected_sha256": "6fb55d90a48067d1bea8f61b24ad97868953803bc5c4966e2c8be1908e0c8db6",
        "construction_note": (
            "The left reads as three reports; the right has an imperative and an unusual phrasal-verb reading."
        ),
    },
)


def normalize(text: str) -> str:
    return "".join(char.lower() for char in text if char.isascii() and char.isalpha())


def outside_in(tape: str) -> dict[str, Any]:
    left, right = 0, len(tape) - 1
    comparisons = 0
    first_mismatch = None
    while left < right:
        comparisons += 1
        if tape[left] != tape[right]:
            first_mismatch = {
                "left_cursor": left,
                "right_cursor": right,
                "left_char": tape[left],
                "right_char": tape[right],
            }
            break
        left += 1
        right -= 1
    return {
        "exact": first_mismatch is None,
        "first_mismatch": first_mismatch,
        "comparisons": comparisons,
    }


@lru_cache(maxsize=8)
def _phrase_preflight_cached(phrases: tuple[str, ...]) -> dict[str, Any]:
    """Check literal authored-clause reuse against the recorded pre-run HEAD."""
    command = ["git", "grep", "-n", "-F", "-i"]
    for phrase in phrases:
        command.extend(("-e", phrase))
    command.extend((PREFLIGHT_REVISION, "--", "runs", "experiments", "docs", "data"))
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"git grep failed for novelty preflight: {result.stderr}")
    hits = [{"match": line} for line in result.stdout.splitlines()]
    return {
        "revision": PREFLIGHT_REVISION,
        "scope": "tracked runs/, experiments/, docs/, and data/ at the pre-experiment HEAD",
        "method": "case-insensitive literal search for every authored clause surface",
        "status": "passed" if not hits else "hits_found",
        "hits": hits,
    }


def phrase_preflight(phrases: list[str]) -> dict[str, Any]:
    return _phrase_preflight_cached(tuple(sorted(set(phrases))))


def _token_boundary_audit(left: str, right: str) -> dict[str, Any]:
    left_tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", left)
    right_tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", right)
    left_norm = [normalize(token) for token in left_tokens]
    right_norm = [normalize(token) for token in right_tokens]
    return {
        "left_tokens": left_tokens,
        "right_tokens": right_tokens,
        "whole_token_sequence_is_mirror": left_norm == list(reversed(right_norm)),
        "left_internal_boundaries": _word_boundaries(left_tokens),
        "reflected_right_internal_boundaries": sorted(
            len(normalize(right)) - boundary for boundary in _word_boundaries(right_tokens)
        ),
        "interpretation": "Character closure is not a word-order-only or whole-token-mirror construction.",
    }


def _word_boundaries(tokens: list[str]) -> list[int]:
    cursor = 0
    boundaries = []
    for token in tokens[:-1]:
        cursor += len(normalize(token))
        boundaries.append(cursor)
    return boundaries


def build_payload() -> dict[str, Any]:
    parent_payload = json.loads(PARENT_PATH.read_text())
    parent_row = parent_payload["rows"][0]
    parent = str(parent_row["rendered"])
    parent_tape = normalize(parent)
    parent_sha = hashlib.sha256(parent_tape.encode("ascii")).hexdigest()
    if len(parent_tape) != 568 or parent_sha != PARENT_SHA256:
        raise AssertionError("pinned 568-letter parent changed")
    if parent_row["audit"]["sha256_forward"] != PARENT_SHA256:
        raise AssertionError("pinned parent's source artifact hash changed")
    if parent.count(OLD_LEFT) != 1 or parent.count(OLD_RIGHT) != 1:
        raise AssertionError("preflighted raw seam no longer occurs exactly once")

    left_raw_start = parent.index(OLD_LEFT)
    left_raw_end = left_raw_start + len(OLD_LEFT)
    right_raw_start = parent.index(OLD_RIGHT)
    right_raw_end = right_raw_start + len(OLD_RIGHT)
    if normalize(parent[:left_raw_start]) != parent_tape[:LEFT_NORMALIZED_SPAN[0]]:
        raise AssertionError("left raw seam start does not align with the recorded normalized span")
    if normalize(OLD_LEFT) != parent_tape[slice(*LEFT_NORMALIZED_SPAN)]:
        raise AssertionError("left raw seam does not match the recorded normalized span")
    if normalize(parent[:right_raw_start]) != parent_tape[:RIGHT_NORMALIZED_SPAN[0]]:
        raise AssertionError("right raw seam start does not align with the recorded normalized span")
    if normalize(OLD_RIGHT) != parent_tape[slice(*RIGHT_NORMALIZED_SPAN)]:
        raise AssertionError("right raw seam does not match the recorded normalized span")
    retained = parent[left_raw_end:right_raw_start]

    rows = []
    for proposal in PROPOSALS:
        left = proposal["left"]
        right = proposal["right"]
        left_tape = normalize(left)
        right_tape = normalize(right)
        if left_tape != right_tape[::-1]:
            raise AssertionError(f"{proposal['id']} leaves a live seam residual")

        rendered = parent[:left_raw_start] + left + retained + right + parent[right_raw_end:]
        tape = normalize(rendered)
        pointer = outside_in(tape)
        forward = hashlib.sha256(tape.encode("ascii")).hexdigest()
        reverse = hashlib.sha256(tape[::-1].encode("ascii")).hexdigest()
        project_exact = project_is_palindrome(rendered)
        if not pointer["exact"] or forward != reverse or not project_exact:
            raise AssertionError(f"independent full-child checks disagree for {proposal['id']}")
        if forward != proposal["expected_sha256"]:
            raise AssertionError(f"unexpected content drift for {proposal['id']}")

        cursor_trace = [
            {
                "cursor": cursor,
                "owner": "left_surface",
                "emitted": char,
                "right_obligation": right_tape[-1 - cursor],
                "consumed": char == right_tape[-1 - cursor],
            }
            for cursor, char in enumerate(left_tape)
        ]
        rows.append({
            "id": proposal["id"],
            "working_status": "exact_568_lineage_child; readability_unproven",
            "rendered": rendered,
            "letters": len(tape),
            "growth_over_pinned_parent": len(tape) - 568,
            "sha256_normalized": forward,
            "audit": {
                "independent_outside_in_exact": pointer["exact"],
                "outside_in_comparisons": pointer["comparisons"],
                "first_mismatch": pointer["first_mismatch"],
                "sha256_forward": forward,
                "sha256_reverse": reverse,
                "hashes_equal": forward == reverse,
                "project_validator_exact": project_exact,
                "normalizer": "independent ASCII-letter casefold",
            },
            "edit": {
                "normalized_parent_spans": [list(LEFT_NORMALIZED_SPAN), list(RIGHT_NORMALIZED_SPAN)],
                "raw_parent_spans": [[left_raw_start, left_raw_end], [right_raw_start, right_raw_end]],
                "replaced_left": OLD_LEFT,
                "replaced_right": OLD_RIGHT,
                "new_left": left,
                "new_right": right,
                "left_letters": len(left_tape),
                "right_letters": len(right_tape),
                "retained_middle_letters": len(normalize(retained)),
                "retained_middle_unchanged": retained == parent[left_raw_end:right_raw_start],
            },
            "live_residual": {
                "equation": {"left": left_tape, "right_obligation": right_tape[::-1]},
                "initial_right_obligation": right_tape[::-1],
                "final_residual": "",
                "cursor_trace": cursor_trace,
                "characters_consumed": len(cursor_trace),
            },
            "anti_shortcut_audit": _token_boundary_audit(left, right),
            "construction_debt": {
                "author_assessment": proposal["construction_note"],
                "human_readability_certified": False,
                "reader_evidence": False,
                "repair_target": "Replace the unrelated clause list with a connected scene while keeping character ownership live across the same actual seam, or widen the seam to [16,52)/[516,552).",
            },
            "provenance": "newly authored surfaces from independent read-only Luna construction lanes; no catalogue/source sentence supplied",
        })

    phrases = [
        phrase
        for proposal in PROPOSALS
        for phrase in re.split(r"[.!?;]+", proposal["left"] + " " + proposal["right"])
        if phrase.strip()
    ]
    rejected_left = "Nadia stops God; Tara saw rewards, Nadia."
    rejected_right = "Aidan's drawer was a rat; Dog spots Aidan."
    rejected_rendered = (
        parent[:left_raw_start]
        + rejected_left
        + retained
        + rejected_right
        + parent[right_raw_end:]
    )
    rejected_tape = normalize(rejected_rendered)
    rejected_scan = outside_in(rejected_tape)
    rejected_sha = hashlib.sha256(rejected_tape.encode("ascii")).hexdigest()
    rejected_reverse_sha = hashlib.sha256(rejected_tape[::-1].encode("ascii")).hexdigest()
    if not rejected_scan["exact"] or rejected_sha != rejected_reverse_sha or not project_is_palindrome(rejected_rendered):
        raise AssertionError("independent checks disagree on the rejected semantic proposal")
    phrases.extend((rejected_left, rejected_right))
    preflight = phrase_preflight(phrases)
    if preflight["status"] != "passed":
        raise AssertionError("authored clause surfaces were found in the tracked preflight corpus")

    return {
        "experiment_id": "incumbent-568-internal-event-resegmentation-20260923",
        "method": "replace an untouched internal seam in the pinned 568 tape; carry the opposing character obligation across independently segmented English clauses",
        "parent": {
            "artifact": str(PARENT_PATH.relative_to(ROOT)),
            "letters": 568,
            "sha256_normalized": PARENT_SHA256,
        },
        "novelty_preflight": preflight,
        "seam": {
            "normalized_spans": [list(LEFT_NORMALIZED_SPAN), list(RIGHT_NORMALIZED_SPAN)],
            "replaced_left": OLD_LEFT,
            "replaced_right": OLD_RIGHT,
            "retained_middle_letters": len(normalize(retained)),
            "equation_is_character_level": True,
        },
        "rows": rows,
        "rejected_proposals": [{
            "id": "internal-drawer-animal-role-mismatch-576",
            "left": rejected_left,
            "right": rejected_right,
            "letters": len(rejected_tape),
            "sha256_normalized": rejected_sha,
            "local_equation_exact": normalize(rejected_left) == normalize(rejected_right)[::-1],
            "independent_outside_in_exact": rejected_scan["exact"],
            "project_validator_exact": project_is_palindrome(rejected_rendered),
            "rejection_reason": "the closure assigns the noun 'rat' to a drawer and does not produce a plausible event relation; exactness is preserved as a rejected control, not a language win",
            "next_operator": "widen the seam and carry entity, predicate, and argument roles across the live character residual before accepting lexical closure",
        }],
        "longest_exact_child_letters": max(row["letters"] for row in rows),
        "reader_eligible": False,
        "next_action": "Test the wider parent seam [16,52)/[516,552) with semantic roles tied across the live residual; do not accept clause-list growth as readability progress.",
    }


if __name__ == "__main__":
    payload = build_payload()
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT_PATH.relative_to(ROOT)), "rows": [{"id": row["id"], "letters": row["letters"], "sha256": row["sha256_normalized"]} for row in payload["rows"]], "novelty": payload["novelty_preflight"]["status"]}, indent=2))
