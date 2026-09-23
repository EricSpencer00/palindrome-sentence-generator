"""Bounded Luna-6 discourse-topology probe against the pinned 568-letter tape.

The ABBA preflight is closed as a duplicate: multiple authored paragraph
seams, residual-conditioned ABBA, hierarchical object/state ABBA, and
staggered paragraph products already cover that family.  This one-shot pivot
tests a causal, cross-paragraph event chain at a previously unused pair of
sentence boundaries.  It is a construction probe, not a readability claim.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT_PATH = ROOT / "runs/incumbent-560-outer-causal-scene-20261002.json"
OUT = ROOT / "runs/luna6-abba-discourse-20260923.json"
PARENT_SHA = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
LEFT_CUT = 64
RIGHT_CUT = 504
LEFT_EVENT = "The map showed a flooded path; Mara set out to find the bridge."
RIGHT_EVENT = "At last, she found the bridge; the flood had covered the path on the map."


def tape(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def token_masks(text: str) -> dict:
    offsets, cursor = [], 0
    for token in re.findall(r"[A-Za-z]+", text):
        offsets.append({"token": token, "start": cursor, "end": cursor + len(token)})
        cursor += len(token)
    return {"tokens": offsets,
            "boundary_offsets": sorted({p for row in offsets for p in (row["start"], row["end"])})}


def cut_at_sentence_boundary(text: str, offset: int) -> tuple[str, str]:
    seen = 0
    for i, char in enumerate(text):
        if char.isalpha():
            seen += 1
        if char in ".?!" and seen == offset:
            j = i + 1
            while j < len(text) and text[j].isspace():
                j += 1
            return text[:i + 1], text[j:]
    raise ValueError(f"sentence boundary after {offset} letters not found")


def outside_in(value: str) -> dict:
    i, j = 0, len(value) - 1
    while i < j and value[i] == value[j]:
        i += 1
        j -= 1
    return {
        "letters": len(value),
        "exact": i >= j,
        "matched_pairs": i,
        "first_mismatch": None if i >= j else {
            "offset_from_left": i,
            "offset_from_right": len(value) - 1 - i,
            "left": value[i],
            "right": value[j],
        },
        "left_residual": value[i:i + 32],
        "right_reverse_residual": value[max(0, j - 31):j + 1][::-1],
    }


def independent_project_check(text: str) -> bool:
    import sys
    sys.path.insert(0, str(ROOT))
    from llm_palindrome.validator import is_palindrome
    return bool(is_palindrome(text))


def independent_checks(text: str) -> dict:
    normalized = tape(text)
    pointer = outside_in(normalized)
    forward = hashlib.sha256(normalized.encode()).hexdigest()
    reverse = hashlib.sha256(normalized[::-1].encode()).hexdigest()
    project = independent_project_check(text)
    return {
        "letters": len(normalized),
        "normalized_tape": normalized,
        "outside_in_exact": pointer["exact"],
        "outside_in_trace": pointer,
        "project_validator_exact": project,
        "sha256_forward": forward,
        "sha256_reverse_obligation": reverse,
        "sha_equal": forward == reverse,
        "all_exact_checks_agree": pointer["exact"] == project == (forward == reverse),
    }


def main() -> dict:
    parent_data = json.loads(PARENT_PATH.read_text())
    parent = next(row for row in parent_data["rows"]
                  if row.get("sha256") == PARENT_SHA
                  or row.get("audit", {}).get("sha256_forward") == PARENT_SHA)
    base = parent["rendered"]
    parent_audit = independent_checks(base)
    if parent_audit["letters"] != 568 or not parent_audit["all_exact_checks_agree"]:
        raise AssertionError("pinned parent failed independent verification")
    if parent_audit["sha256_forward"] != PARENT_SHA:
        raise AssertionError("pinned parent SHA does not match the authorized baseline")

    left, rest = cut_at_sentence_boundary(base, LEFT_CUT)
    # RIGHT_CUT is in the original parent's coordinate system; the second
    # split therefore subtracts only the already-consumed left prefix.
    middle, right = cut_at_sentence_boundary(rest, RIGHT_CUT - LEFT_CUT)
    rendered = left + " " + LEFT_EVENT + " " + middle + " " + RIGHT_EVENT + " " + right
    result = independent_checks(rendered)

    left_tape, right_tape = tape(LEFT_EVENT), tape(RIGHT_EVENT)
    obligation = right_tape[::-1]
    matched = 0
    while matched < min(len(left_tape), len(obligation)) and left_tape[matched] == obligation[matched]:
        matched += 1
    # The cursor reports the first live equation failure for this one concrete
    # semantic graft; it does not turn a failed row into a candidate.
    equation = {
        "left_event_tape": left_tape,
        "right_event_tape": right_tape,
        "left_length": len(left_tape),
        "right_length": len(right_tape),
        "right_reverse_obligation": right_tape[::-1],
        "matched_characters_from_event_seam": matched,
        "first_event_seam_mismatch": None if matched == min(len(left_tape), len(obligation)) else {
            "offset": matched, "left": left_tape[matched], "required": obligation[matched]},
        "residual_after_match": left_tape[matched:matched + 32],
        "remaining_reverse_obligation": obligation[matched:matched + 32],
    }

    preflight = {
        "abba_authored_paragraph_seam_20260930": {
            "branches": 81, "exact_gt38": 0, "max_supported_depth": 0},
        "abba_residual_conditioned_paragraph_20260930": {
            "joint_candidates": 225, "exact_gt38": 0, "max_supported_depth": 0},
        "hierarchical_paragraph_abba_20261002": {
            "exact_topology_controls": 2, "longest_letters": 106,
            "both_rejected_for_word_order_symmetry_and_repeated_scaffold": True},
        "multisentence_generation_abba_20261002": {
            "rows": 16, "exact_closures": 0, "max_supported_depth": 0},
        "packed_staggered_paragraph_strict_symmetric_20260922": {
            "reachable_product_states": 216, "coaccessible_product_states": 0},
        "decision": "ABBA and staggered paragraph products are saturated; no larger sweep is justified.",
    }

    return {
        "experiment_id": "luna6-abba-discourse-20260923",
        "method": "one-shot causal cross-paragraph event graft at fresh sentence boundaries",
        "novelty_preflight": preflight,
        "pivot": {
            "topology": "map evidence -> journey -> discovery, with the same map/bridge/flood event graph spanning two paragraph regions",
            "not_abba_clause_pairing": True,
            "not_a_bank_sweep": True,
            "not_a_word-reversal_operator": True,
        },
        "parent": {
            "artifact": "runs/incumbent-560-outer-causal-scene-20261002.json",
            "sha256": PARENT_SHA,
            "letters": 568,
            "rendered": base,
            "independent_audit": parent_audit,
        },
        "attempt": {
            "sentence_boundary_letter_offsets": [LEFT_CUT, RIGHT_CUT],
            "left_event": LEFT_EVENT,
            "right_event": RIGHT_EVENT,
            "rendered": rendered,
            "letters": result["letters"],
            "length_gain": result["letters"] - 568,
            "provenance": {
                "newly_authored_event_content": True,
                "event_graph": ["map shows flooded path", "Mara seeks bridge", "Mara finds bridge", "flood covered path"],
                "parent_content_borrowed_as_new": False,
                "finished_tape_reversal": False,
                "repeated_generated_sentence": False,
                "self_palindromic_inserted_unit": False,
                "whole_word_reversal_shortcut_claimed": False,
                "reader_evidence": "none; failed exact rows are not reader candidates",
            },
            "independent_audit": result,
            "inserted_seam_equation": equation,
            "mask_audit": {
        "left_event": token_masks(LEFT_EVENT),
        "right_event": token_masks(RIGHT_EVENT),
                "strict_admission": "not run on a non-exact failed construction",
            },
            "status": "exact child" if result["all_exact_checks_agree"] and result["outside_in_exact"] else "rejected: exact equation does not close",
        },
        "next_operator": {
            "action": "Change the editable event seam, not the ABBA/causal paragraph order: jointly lexicalize one short shared-referent clause on each side around the residual `" + equation["residual_after_match"] + "`.",
            "guard": "Require an exact whole-output child above 568, no word-aligned reverse pairs, and the existing strict mechanical admission before any reader packet.",
        },
    }


if __name__ == "__main__":
    OUT.write_text(json.dumps(main(), indent=2) + "\n")
    print(json.dumps({"experiment": "luna6-abba-discourse-20260923", **main()["attempt"]["independent_audit"]}, sort_keys=True))
