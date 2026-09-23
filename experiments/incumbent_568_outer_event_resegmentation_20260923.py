#!/usr/bin/env python3
"""Replace an untouched outer seam of the pinned 568-letter candidate.

This is a candidate-producing seam construction, not a readability claim. The
left and right event surfaces are authored independently, then checked against
the live reflected-character obligation while preserving the verified middle.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.validator import is_palindrome as project_is_palindrome


PARENT_PATH = ROOT / "runs" / "incumbent-560-outer-causal-scene-20261002.json"
OUT_PATH = ROOT / "runs" / "incumbent-568-outer-event-resegmentation-20260923.json"
PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"

LEFT_RAW_END = 26
RIGHT_RAW_START = 759
RIGHT_RAW_END = 786
LEFT_NORMALIZED_SPAN = (0, 20)
RIGHT_NORMALIZED_SPAN = (548, 568)
LEFT_SURFACE = "Iris spots a rat. Ari saw a ram."
RIGHT_SURFACE = "Mara was Ira. Tara stops Siri."


def letters(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isascii() and ch.isalpha())


def _pointer_audit(tape: str) -> dict[str, Any]:
    left, right = 0, len(tape) - 1
    trace: list[dict[str, Any]] = []
    while left < right and tape[left] == tape[right]:
        trace.append({"left_cursor": left, "right_cursor": right,
                      "left_char": tape[left], "right_char": tape[right],
                      "matched": True})
        left += 1
        right -= 1
    mismatch = None
    if left < right:
        mismatch = {"left_cursor": left, "right_cursor": right,
                    "left_char": tape[left], "right_char": tape[right]}
        trace.append({**mismatch, "matched": False})
    return {"exact": mismatch is None, "first_mismatch": mismatch,
            "comparisons": len(trace), "trace": trace}


def _token_spans(text: str) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    cursor = 0
    for token in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text):
        width = len(letters(token))
        spans.append({"token": token, "start": cursor, "end": cursor + width})
        cursor += width
    return spans


def _boundary_audit(left: str, right: str) -> dict[str, Any]:
    left_spans = _token_spans(left)
    right_spans = _token_spans(right)
    left_boundaries = sorted({s["end"] for s in left_spans[:-1]})
    right_boundaries = sorted({s["end"] for s in right_spans[:-1]})
    reflected = sorted(len(letters(right)) - b for b in right_boundaries)
    return {
        "left_tokens": [s["token"] for s in left_spans],
        "right_tokens": [s["token"] for s in right_spans],
        "left_internal_boundaries": left_boundaries,
        "reflected_right_internal_boundaries": reflected,
        "shared_reflected_boundaries": sorted(set(left_boundaries) & set(reflected)),
        "interpretation": (
            "Exactness is character-level. Shared word boundaries and reversed-token "
            "pairs are recorded as construction debt, not hidden or treated as human evidence."
        ),
    }


def build_payload() -> dict[str, Any]:
    parent_payload = json.loads(PARENT_PATH.read_text())
    parent_row = parent_payload["rows"][0]
    parent = str(parent_row["rendered"])
    parent_tape = letters(parent)
    parent_sha = hashlib.sha256(parent_tape.encode("ascii")).hexdigest()
    if len(parent_tape) != 568 or parent_sha != PARENT_SHA256:
        raise AssertionError("the pinned 568-letter parent changed")
    if parent_row["audit"]["sha256_forward"] != PARENT_SHA256:
        raise AssertionError("the parent artifact's recorded hash changed")

    old_left = parent[:LEFT_RAW_END]
    retained = parent[LEFT_RAW_END:RIGHT_RAW_START]
    old_right = parent[RIGHT_RAW_START:RIGHT_RAW_END]
    if old_left != "Leon won. Wolf spots Nora." or old_right != " Aron stops flow now, Noel.":
        raise AssertionError("the preflighted outer seam no longer matches the parent")
    if letters(old_left) != parent_tape[slice(*LEFT_NORMALIZED_SPAN)]:
        raise AssertionError("left raw seam does not match its normalized offsets")
    if letters(old_right) != parent_tape[slice(*RIGHT_NORMALIZED_SPAN)]:
        raise AssertionError("right raw seam does not match its normalized offsets")
    retained_tape = letters(retained)
    if retained_tape != retained_tape[::-1]:
        raise AssertionError("the retained middle is not itself exactly mirrored")

    left_tape = letters(LEFT_SURFACE)
    right_tape = letters(RIGHT_SURFACE)
    if left_tape != right_tape[::-1]:
        raise AssertionError("the authored replacement does not close the live seam")

    # Preserve the original single spaces at both splice boundaries.
    rendered = LEFT_SURFACE + retained + " " + RIGHT_SURFACE
    candidate_tape = letters(rendered)
    pointer = _pointer_audit(candidate_tape)
    forward_sha = hashlib.sha256(candidate_tape.encode("ascii")).hexdigest()
    reverse_sha = hashlib.sha256(candidate_tape[::-1].encode("ascii")).hexdigest()
    project_exact = project_is_palindrome(rendered)
    if not pointer["exact"] or forward_sha != reverse_sha or not project_exact:
        raise AssertionError("independent exactness checks disagree")

    # The equation owner is explicit: each left character emits one right-side
    # obligation, consumed from the opposite surface in reverse order.
    seam_trace = [
        {"cursor": i, "owner": "left_surface", "emitted": ch,
         "right_obligation": right_tape[-1 - i], "consumed": ch == right_tape[-1 - i]}
        for i, ch in enumerate(left_tape)
    ]
    if not all(row["consumed"] for row in seam_trace):
        raise AssertionError("live seam trace left an unconsumed character")

    payload = {
        "experiment_id": "incumbent-568-outer-event-resegmentation-20260923",
        "method": "replace a previously untouched 20+20-letter outer event seam with independently authored multi-clause surfaces and a live reflected-character ledger",
        "parent": {
            "artifact": str(PARENT_PATH.relative_to(ROOT)),
            "id": parent_row["id"],
            "letters": 568,
            "sha256": PARENT_SHA256,
        },
        "novelty_preflight": {
            "normalized_parent_spans": [list(LEFT_NORMALIZED_SPAN), list(RIGHT_NORMALIZED_SPAN)],
            "raw_parent_spans": [[0, LEFT_RAW_END], [RIGHT_RAW_START, RIGHT_RAW_END]],
            "prior_exact_clause_matches": 0,
            "known_overlaps_excluded": [
                "[48,520] now/won and event-chain searches",
                "[64,504] sees-chain insertion",
                "[99,469] five/four partial-word probe",
                "[148,420] phrasewise audit",
                "locative inversion and repeated-shell operators",
            ],
            "scope": "exact replacement clauses checked against tracked run, experiment, documentation, and data text before this artifact was added",
        },
        "rows": [{
            "id": "outer-two-clause-event-resegmentation-574",
            "working_status": "exact_growth_frontier_not_readability_claim",
            "rendered": rendered,
            "letters": len(candidate_tape),
            "normalized_sha256": forward_sha,
            "audit": {
                "two_pointer_exact": pointer["exact"],
                "first_mismatch": pointer["first_mismatch"],
                "two_pointer_comparisons": pointer["comparisons"],
                "forward_sha256": forward_sha,
                "reverse_sha256": reverse_sha,
                "sha_equal": forward_sha == reverse_sha,
                "project_validator_exact": project_exact,
                "independent_normalizer": "ASCII alphabetic characters, case-folded",
            },
            "parent_edit": {
                "left_raw_replaced": old_left,
                "right_raw_replaced": old_right,
                "left_surface": LEFT_SURFACE,
                "right_surface": RIGHT_SURFACE,
                "retained_middle_letters": len(retained_tape),
                "retained_middle_sha256": hashlib.sha256(retained_tape.encode("ascii")).hexdigest(),
                "retained_middle_unchanged": retained == parent[LEFT_RAW_END:RIGHT_RAW_START],
                "growth_over_parent": len(candidate_tape) - len(parent_tape),
                "new_event_content": [
                    "Iris spots a rat",
                    "Ari saw a ram",
                    "Mara was Ira",
                    "Tara stops Siri",
                ],
            },
            "live_seam": {
                "equation": {"left": left_tape, "right_obligation": right_tape[::-1]},
                "initial_residual": right_tape[::-1],
                "final_residual": "",
                "final_owner": None,
                "characters_consumed": len(seam_trace),
                "cursor_trace": seam_trace,
                "residual_ownership": "left_surface emits; right_surface carries the opposing character obligation",
            },
            "construction_debt": {
                "boundary_audit": _boundary_audit(LEFT_SURFACE, RIGHT_SURFACE),
                "discourse_coherence": "not established; four grammatical event sentences do not yet form reader-worthy connected prose",
                "reader_evidence": False,
                "human_certified": False,
                "next_reader_facing_test": "After repairing this outer seam, compare it with the pinned 568 parent and intact/shuffled controls in randomized blinded human ratings; retain exactness as an independent mechanical gate.",
            },
            "provenance": "newly authored clause surfaces from a read-only Luna candidate lane; exact phrase preflight found no tracked occurrence before this experiment; no source sentence or catalogue palindrome was copied",
        }],
        "rejected_proposals": [{
            "id": "outer-26-letter-tokenwise-alternative-580",
            "proposer": "independent Luna transducer lane",
            "left_surface": "Aron stops Aidan. Nadia saw Aram.",
            "right_surface": "Mara was Aidan. Nadia spots Nora.",
            "letters_if_spliced": 580,
            "local_equation_exact": True,
            "rejection_reasons": [
                "all six left words pair one-for-one with reversed right words (Aron/Nora, stops/spots, Aidan/Nadia, Nadia/Aidan, saw/was, Aram/Mara), a tokenwise shortcut",
                "the exact clause `Nadia spots Nora` already occurs in runs/incumbent-568-remaining-shell-global-gate-20261002.json and runs/incumbent-568-repeated-shell-event-lattice-20261002.json",
            ],
            "pivot": "retain the 574 exact child only as rough construction evidence; move to a seam/operator that joins characters across more than isolated token reversals and uses no prior clause surface",
        }, {
            "id": "outer-52-letter-repeated-event-proposal",
            "proposer": "independent Luna seam-inventory lane",
            "normalized_parent_spans": [[0, 48], [520, 568]],
            "left_surface": "Nadia saw desserts; Aidan saw a ram. Mara saw a rat; Sara stops a ram.",
            "right_surface": "Mara spots Aras; Tara was Aram. Mara was Nadia; stressed was Aidan.",
            "letters_per_surface": 52,
            "local_equation_exact": True,
            "full_child_admitted": False,
            "rejection_reasons": [
                "repeats the saw/a-ram event frame and Mara identity scaffold",
                "the closure still factors into the familiar desserts/stressed, rat/Tara, and stops/spots reverse-word pairs",
                "the lane found exactness but no connected reader-worthy discourse, so it is not admitted as a candidate win",
            ],
            "pivot": "leave this outer topology and test the untouched actual parent seam [20,48)/[520,548) with a causal/reporting structure selected before lexical realization",
        }],
        "next_operator": "change the actual 568 edit target to normalized seam [20,48)/[520,548); select a connected causal/reporting topology before lexicalization, carry residual and grammar ownership together, and reject any tokenwise reverse shortcut",
    }

    # A concrete seam repair changed the odd identity/copula lines and forced
    # a different word-boundary equation. Keep the original 574 child above.
    repair_left = "Iris spots a yak. Ari stops a hen."
    repair_right = "Neha spots Ira. Kaya stops Siri."
    repair_left_tape = letters(repair_left)
    repair_right_tape = letters(repair_right)
    if repair_left_tape != repair_right_tape[::-1]:
        raise AssertionError("repaired outer seam does not close its residual")
    repair_rendered = repair_left + retained + " " + repair_right
    repair_tape = letters(repair_rendered)
    repair_pointer = _pointer_audit(repair_tape)
    repair_forward_sha = hashlib.sha256(repair_tape.encode("ascii")).hexdigest()
    repair_reverse_sha = hashlib.sha256(repair_tape[::-1].encode("ascii")).hexdigest()
    repair_project_exact = project_is_palindrome(repair_rendered)
    if not repair_pointer["exact"] or repair_forward_sha != repair_reverse_sha or not repair_project_exact:
        raise AssertionError("the repaired outer seam failed an independent exactness check")
    repair_trace = [
        {"cursor": i, "owner": "left_surface", "emitted": ch,
         "right_obligation": repair_right_tape[-1 - i], "consumed": ch == repair_right_tape[-1 - i]}
        for i, ch in enumerate(repair_left_tape)
    ]
    payload["rows"].append({
        "id": "outer-event-resegmentation-repair-578",
        "working_status": "exact_growth_frontier_repair_not_readability_claim",
        "rendered": repair_rendered,
        "letters": len(repair_tape),
        "normalized_sha256": repair_forward_sha,
        "audit": {
            "two_pointer_exact": repair_pointer["exact"],
            "first_mismatch": repair_pointer["first_mismatch"],
            "two_pointer_comparisons": repair_pointer["comparisons"],
            "forward_sha256": repair_forward_sha,
            "reverse_sha256": repair_reverse_sha,
            "sha_equal": repair_forward_sha == repair_reverse_sha,
            "project_validator_exact": repair_project_exact,
            "independent_normalizer": "ASCII alphabetic characters, case-folded",
        },
        "parent_edit": {
            "left_raw_replaced": old_left,
            "right_raw_replaced": old_right,
            "left_surface": repair_left,
            "right_surface": repair_right,
            "retained_middle_letters": len(retained_tape),
            "retained_middle_sha256": hashlib.sha256(retained_tape.encode("ascii")).hexdigest(),
            "retained_middle_unchanged": retained == parent[LEFT_RAW_END:RIGHT_RAW_START],
            "growth_over_parent": len(repair_tape) - len(parent_tape),
            "repair_of": "outer-two-clause-event-resegmentation-574",
            "new_event_content": [
                "Iris spots a yak",
                "Ari stops a hen",
                "Neha spots Ira",
                "Kaya stops Siri",
            ],
        },
        "live_seam": {
            "equation": {"left": repair_left_tape, "right_obligation": repair_right_tape[::-1]},
            "initial_residual": repair_right_tape[::-1],
            "final_residual": "",
            "final_owner": None,
            "characters_consumed": len(repair_trace),
            "cursor_trace": repair_trace,
            "residual_ownership": "left_surface emits; right_surface carries the opposing character obligation",
        },
        "construction_debt": {
            "boundary_audit": _boundary_audit(repair_left, repair_right),
            "discourse_coherence": "sentences are locally well-formed, but event-to-event discourse remains weak and is not reader-certified",
            "reader_evidence": False,
            "human_certified": False,
            "next_reader_facing_test": "Only after a discourse repair, compare against the pinned 568 parent and intact/shuffled controls in randomized blinded human ratings.",
        },
        "provenance": "newly authored seam repair from an independent Luna candidate lane; tracked-source phrase preflight found no exact clause occurrence before recording",
    })
    payload["best_new_exact_child"] = {
        "id": payload["rows"][-1]["id"],
        "letters": payload["rows"][-1]["letters"],
        "sha256": payload["rows"][-1]["normalized_sha256"],
        "note": "best new exact outer-seam child; not promoted over the user-pinned 568 working parent and not a readability claim",
    }
    return payload


def run() -> dict[str, Any]:
    payload = build_payload()
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return payload


if __name__ == "__main__":
    result = run()
    for row in result["rows"]:
        print(json.dumps({"id": row["id"], "letters": row["letters"],
                          "sha256": row["normalized_sha256"],
                          "two_pointer_exact": row["audit"]["two_pointer_exact"],
                          "project_validator_exact": row["audit"]["project_validator_exact"]},
                         sort_keys=True))
    print(result["rows"][-1]["rendered"])
