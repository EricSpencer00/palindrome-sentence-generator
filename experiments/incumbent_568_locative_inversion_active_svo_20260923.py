#!/usr/bin/env python3
"""Bounded locative-inversion/active-SVO probes on the pinned 568 tape.

The construction is deliberately lexicalized only after its word-length
masks are fixed.  It records local controls and the live residual; no failed
clause pair is rendered as a palindrome candidate.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT_PATH = "runs/incumbent-560-outer-causal-scene-20261002.json"
PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
OUTPUT_PATH = "runs/incumbent-568-locative-inversion-active-svo-20260923.json"


def tape(text: str) -> str:
    return "".join(char.lower() for char in text
                   if char.isascii() and char.isalpha())


def word_lengths(text: str) -> list[int]:
    return [len(tape(word)) for word in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)]


def internal_boundaries(lengths: list[int]) -> list[int]:
    total = 0
    out = []
    for width in lengths[:-1]:
        total += width
        out.append(total)
    return out


def local_equation(left: str, right: str, expected_left: list[int], expected_right: list[int]) -> dict[str, object]:
    left_tape, right_tape = tape(left), tape(right)
    left_lengths, right_lengths = word_lengths(left), word_lengths(right)
    left_boundaries = internal_boundaries(left_lengths)
    right_boundaries = internal_boundaries(right_lengths)
    reflected_right = sorted(len(right_tape) - offset for offset in right_boundaries)
    common = sorted(set(left_boundaries) & set(reflected_right))
    paired = right_tape[::-1]
    mismatch = next((i for i, (a, b) in enumerate(zip(left_tape, paired)) if a != b), None)
    prefix = min(len(left_tape), len(paired)) if mismatch is None else mismatch
    return {
        "left_rendered": left,
        "right_rendered": right,
        "left_tape": left_tape,
        "right_tape": right_tape,
        "left_letters": len(left_tape),
        "right_letters": len(right_tape),
        "left_word_lengths": left_lengths,
        "right_word_lengths": right_lengths,
        "left_internal_boundaries": left_boundaries,
        "reflected_right_internal_boundaries": reflected_right,
        "shared_reflected_boundaries": common,
        "boundary_masks_match_preregistered": (
            left_lengths == expected_left and right_lengths == expected_right
        ),
        "matched_outer_characters": prefix,
        "first_mismatch": None if mismatch is None else {
            "offset": mismatch,
            "left": left_tape[mismatch],
            "right_reversed": paired[mismatch],
        },
        "exact_local_equation": (
            len(left_tape) == len(right_tape) and left_tape == paired
        ),
        "phrasewise_gate_offsets_if_exact": (
            common if len(left_tape) == len(right_tape) and left_tape == paired else []
        ),
    }


def run() -> dict[str, object]:
    parent_artifact = json.loads((ROOT / PARENT_PATH).read_text())
    parent_rendered = parent_artifact["rows"][0]["rendered"]
    parent_tape = tape(parent_rendered)
    parent_sha = hashlib.sha256(parent_tape.encode("ascii")).hexdigest()
    if len(parent_tape) != 568 or parent_sha != PARENT_SHA256:
        raise AssertionError("pinned 568 parent changed")

    # The 23-letter locative-inversion/canonical-motion mask was tried first.
    wetland_left_mask = [5, 3, 5, 4, 1, 5]
    wetland_right_mask = [3, 4, 5, 4, 7]
    wetland = local_equation(
        "Among the reeds rose a heron.",
        "The tern soars over islands.",
        wetland_left_mask,
        wetland_right_mask,
    )
    if wetland["left_letters"] != 23 or wetland["right_letters"] != 23:
        raise AssertionError("wetland control does not fit its preregistered 23-letter mask")
    wetland_reflected = sorted(23 - x for x in [3, 7, 12, 16])
    if set([5, 8, 13, 17, 18]) & set(wetland_reflected):
        raise AssertionError("preregistered wetland masks unexpectedly overlap")

    # A distinct 27-letter repair pairs temporal/locative inversion with an
    # active transitive observation. The second row changes only the internal
    # boundary of the same 4-letter left slot, from `dawn` to `a dam`.
    active_left_mask = [2, 4, 5, 3, 8, 5]
    active_right_mask = [7, 4, 7, 5, 4]
    active = local_equation(
        "At dawn stood one watchful guard.",
        "Rangers read current field data.",
        active_left_mask,
        active_right_mask,
    )
    repaired = local_equation(
        "At a dam stood one watchful guard.",
        "Rangers read current field data.",
        [2, 1, 3, 5, 3, 8, 5],
        active_right_mask,
    )
    magma = local_equation(
        "At a dam stood one watchful guard.",
        "Rangers read current magma data.",
        [2, 1, 3, 5, 3, 8, 5],
        active_right_mask,
    )
    llama = local_equation(
        "At a dam stood one watchful guard.",
        "Rangers read current llama data.",
        [2, 1, 3, 5, 3, 8, 5],
        active_right_mask,
    )
    for row in (active, repaired, magma, llama):
        if row["left_letters"] != 27 or row["right_letters"] != 27:
            raise AssertionError("active-SVO control does not fit the 27-letter mask")
        if row["shared_reflected_boundaries"]:
            raise AssertionError("preregistered active-SVO boundaries are not complementary")

    # At the opposite live frontier, an ordinary animate subject must begin
    # with the reverse of the left subject after the final adjective letters
    # are accounted for. Preserve the strongest observed partial overlaps.
    role_slots = []
    for left_animal, right_animate in (("adder", "readers"), ("eider", "readers")):
        forced = left_animal[::-1]
        shared = 0
        while shared < min(len(forced), len(right_animate)) and forced[shared] == right_animate[shared]:
            shared += 1
        role_slots.append({
            "left_animate": left_animal,
            "right_animate_candidate": right_animate,
            "forced_right_prefix": forced,
            "matched_prefix_letters": shared,
            "first_mismatch": None if shared == len(forced) else {
                "offset": shared,
                "required": forced[shared],
                "observed": right_animate[shared],
            },
        })

    return {
        "experiment_id": "incumbent-568-locative-inversion-active-svo-20260923",
        "status": "bounded_exact_equation_search_no_child",
        "method": "pre-register locative-inversion/active-SVO word masks, carry opposite character cursors through typed slots, and repair only the first live residual",
        "parent": {
            "artifact": PARENT_PATH,
            "normalized_letters": len(parent_tape),
            "sha256": parent_sha,
            "target_seam": [[148, 163], [405, 420]],
        },
        "attempts": [
            {
                "topology": "locative-inversion/canonical-motion",
                "preregistered_left_mask": wetland_left_mask,
                "preregistered_right_mask": wetland_right_mask,
                "left_boundaries": [5, 8, 13, 17, 18],
                "reflected_right_boundaries": wetland_reflected,
                "probe": wetland,
                "lexical_obstruction": "The ending of the 5-letter animate subject must reverse to a 3-letter determiner; ordinary wetland-animal slots supply no grammatical determiner. The only near hit is archaic `annet`, and `a annet` is ungrammatical.",
            },
            {
                "topology": "temporal-locative-inversion/active-SVO",
                "preregistered_left_mask": active_left_mask,
                "preregistered_right_mask": active_right_mask,
                "probe": active,
                "residual_directed_mask_repair": repaired,
                "single_slot_repairs": [
                    {
                        "slot": "right seven-letter modifier before data",
                        "substitution": "field -> magma",
                        "probe": magma,
                        "residual_obstruction": {
                            "offset": 6,
                            "left_residual": magma["left_tape"][6:],
                            "mirrored_right_residual": magma["right_tape"][::-1][6:],
                            "mismatch": magma["first_mismatch"],
                        },
                    },
                    {
                        "slot": "right seven-letter modifier before data",
                        "substitution": "field -> llama",
                        "probe": llama,
                        "residual_obstruction": {
                            "offset": 6,
                            "left_residual": llama["left_tape"][6:],
                            "mirrored_right_residual": llama["right_tape"][::-1][6:],
                            "mismatch": llama["first_mismatch"],
                        },
                    },
                ],
                "lexical_obstruction": "The first control closes `at`; `At a dam` advances through `atad`. Replacing `field` with `magma` advances the live match through `atadam` but leaves offset 6 `s != g`; `llama` also stops at offset 6. No exact local equation exists in this topology, so the next experiment changes the actual parent seam rather than widening this lexical slot.",
            },
        ],
        "role_slot_frontier": role_slots,
        "stats": {
            "rendered_clause_pairs": 5,
            "exact_local_equations": 0,
            "independently_exact_children": 0,
            "longest_local_clause_letters_per_side": 27,
            "prospective_growth_if_27_letter_pair_closed": 24,
            "prospective_child_letters": 592,
            "shared_reflected_word_boundaries": 0,
        },
        "shortcut_checks": {
            "finished_tape_reversed_to_construct": False,
            "borrowed_catalogue_text": False,
            "reader_claim": False,
            "candidate_admitted": False,
        },
        "provenance": {
            "construction_source": "registry-checked locative inversion topology proposed by independent skeptical review; lexical controls and residuals preserved before any child closure",
            "parent_lineage": PARENT_SHA256,
        },
        "next_action": "Retire this 27-letter locative-inversion/active-SVO topology after the offset-6 residual: `stood...` begins `s`, while `magma` and `llama` expose `g` and `a`. Change to a different actual seam on the pinned 568 tape and preregister a new grammar topology before lexicalizing. Longer descendants remain preserved comparisons, not replacements for the user-pinned parent. Do not return to the endpoint-infeasible five-letter animal/determiner mask.",
    }


if __name__ == "__main__":
    artifact = run()
    (ROOT / OUTPUT_PATH).write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({
        "artifact": OUTPUT_PATH,
        "exact_local_equations": artifact["stats"]["exact_local_equations"],
        "candidate_children": artifact["stats"]["independently_exact_children"],
        "largest_clause_side": artifact["stats"]["longest_local_clause_letters_per_side"],
        "prospective_child_letters": artifact["stats"]["prospective_child_letters"],
        "cursor_repair": {
            "before": artifact["attempts"][1]["probe"]["first_mismatch"],
            "after": artifact["attempts"][1]["residual_directed_mask_repair"]["first_mismatch"],
        },
        "single_slot_repairs": [
            row["probe"]["first_mismatch"]
            for row in artifact["attempts"][1]["single_slot_repairs"]
        ],
    }, indent=2))
