"""Add one authored scene pair at the live midpoint of the 586-letter tape.

This is a candidate-specific construction, not a claim of a new general
search algorithm. The inserted clauses use cross-boundary resegmentation:
"Nora saw a rat on a mat" reverses to "Tama, no! Tara was Aron."
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

from llm_palindrome.admission import is_boundary_aligned_word_mirror, tokenize
from llm_palindrome.validator import is_palindrome, normalize


PARENT = ROOT / "runs" / "incumbent-586-outer-flow-frame-repair-20260923.json"
OUT = ROOT / "runs" / "incumbent-586-live-center-scene-extension-20260923.json"
PARENT_ID = "outer-wolf-flow-scene-586"
PARENT_SHA256 = "00cf2cf66f4cf63a0c3b48a204511eeb3436b080fa0033b9477de4328e1a941e"
CENTER = 293
LEFT_SCENE = "Nora saw a rat on a mat."
RIGHT_SCENE = "Tama, no! Tara was Aron."
LEFT_TAPE = "norasawaratonamat"
RIGHT_TAPE = "tamanotarawasaron"


def tape(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def two_pointer_exact(value: str) -> bool:
    left, right = 0, len(value) - 1
    while left < right:
        if value[left] != value[right]:
            return False
        left += 1
        right -= 1
    return bool(value)


def raw_offset_for_tape_boundary(rendered: str, boundary: int) -> int:
    letter_offsets = [
        index
        for index, character in enumerate(rendered)
        if character.isascii() and character.isalpha()
    ]
    assert 0 < boundary < len(letter_offsets)
    return letter_offsets[boundary]


def live_residual_trace(left: str, right: str) -> list[dict[str, Any]]:
    assert left == right[::-1]
    residual = left
    trace: list[dict[str, Any]] = []
    for cursor, emitted in enumerate(left):
        right_cursor = len(right) - 1 - cursor
        expected = right[right_cursor]
        assert emitted == expected
        residual = residual[1:]
        trace.append(
            {
                "owner": "left-scene-emitter",
                "left_cursor": cursor,
                "emitted": emitted,
                "right_cursor": right_cursor,
                "expected": expected,
                "residual_after": residual,
            }
        )
    assert residual == ""
    return trace


def build_payload() -> dict[str, Any]:
    parent_payload = json.loads(PARENT.read_text())
    parent = parent_payload["candidate"]
    assert parent["id"] == PARENT_ID
    parent_rendered = str(parent["rendered"])
    parent_tape = tape(parent_rendered)
    parent_hash = hashlib.sha256(parent_tape.encode("ascii")).hexdigest()
    assert len(parent_tape) == 586
    assert parent_hash == PARENT_SHA256
    assert two_pointer_exact(parent_tape)
    assert is_palindrome(parent_rendered)
    assert normalize(parent_rendered) == parent_tape
    assert len(parent_tape) % 2 == 0

    left_tape = tape(LEFT_SCENE)
    right_tape = tape(RIGHT_SCENE)
    inserted_pair = tokenize(f"{LEFT_SCENE} {RIGHT_SCENE}")
    word_order_mirror = is_boundary_aligned_word_mirror(inserted_pair)
    assert left_tape == LEFT_TAPE
    assert right_tape == RIGHT_TAPE
    assert len(left_tape) == len(right_tape) == 17
    assert left_tape == right_tape[::-1]
    assert not word_order_mirror

    raw_center = raw_offset_for_tape_boundary(parent_rendered, CENTER)
    # The midpoint falls between two words and follows a sentence boundary;
    # no existing letters or punctuation are rewritten.
    assert parent_rendered[raw_center - 1].isspace()
    inserted = f"{LEFT_SCENE} {RIGHT_SCENE} "
    rendered = parent_rendered[:raw_center] + inserted + parent_rendered[raw_center:]
    candidate_tape = tape(rendered)
    forward_sha = hashlib.sha256(candidate_tape.encode("ascii")).hexdigest()
    reverse_sha = hashlib.sha256(candidate_tape[::-1].encode("ascii")).hexdigest()
    trace = live_residual_trace(left_tape, right_tape)

    assert candidate_tape == (
        parent_tape[:CENTER] + left_tape + right_tape + parent_tape[CENTER:]
    )
    assert len(candidate_tape) == 620
    assert two_pointer_exact(candidate_tape)
    assert is_palindrome(rendered)
    assert normalize(rendered) == candidate_tape
    assert forward_sha == reverse_sha
    assert rendered.count(LEFT_SCENE) == 1
    assert rendered.count(RIGHT_SCENE) == 1

    return {
        "experiment_id": "incumbent-586-live-center-scene-extension-20260923",
        "method": "single incumbent-specific midpoint clause-pair insertion with a live 17-character residual",
        "method_scope": {
            "candidate_specific": True,
            "general_algorithm_claim": False,
            "overlap_note": "Center insertion and clitic/determiner seam families exist in the historical registry; this record counts a new incumbent-specific scene equation, not a new general method family.",
        },
        "status": "exact_620_letter_child_with_readability_debt",
        "parent": {
            "artifact": str(PARENT.relative_to(ROOT)),
            "id": PARENT_ID,
            "letters": len(parent_tape),
            "sha256": parent_hash,
            "center_boundary": CENTER,
            "raw_insertion_offset": raw_center,
            "raw_context_before": parent_rendered[max(0, raw_center - 40) : raw_center],
            "raw_context_after": parent_rendered[raw_center : raw_center + 40],
        },
        "candidate": {
            "id": "live-center-scene-nora-tama-620",
            "rendered": rendered,
            "letters": len(candidate_tape),
            "length_delta": len(candidate_tape) - len(parent_tape),
            "sha256": forward_sha,
            "audit": {
                "independent_two_pointer_exact": two_pointer_exact(candidate_tape),
                "project_validator_exact": bool(is_palindrome(rendered)),
                "project_normalizer_agrees": normalize(rendered) == candidate_tape,
                "sha256_forward": forward_sha,
                "sha256_reverse": reverse_sha,
                "sha_equal": forward_sha == reverse_sha,
            },
            "live_seam": {
                "normalized_parent_boundary": CENTER,
                "left_scene": LEFT_SCENE,
                "right_scene": RIGHT_SCENE,
                "equation": f"{left_tape} = reverse({right_tape})",
                "left_tape": left_tape,
                "right_obligation": right_tape,
                "owner": "left-scene-emitter",
                "trace": trace,
                "final_owner": None,
                "final_residual": "",
                "committed_character_contradictions": 0,
                "cross_boundary_resegmentation": [
                    {"left_tokens": ["mat", "a"], "right_token": "Tama"},
                    {"left_tokens": ["on"], "right_token": "no"},
                    {"left_tokens": ["rat", "a"], "right_token": "Tara"},
                    {"left_tokens": ["saw"], "right_token": "was"},
                    {"left_tokens": ["Nora"], "right_token": "Aron"},
                ],
            },
            "provenance": {
                "source_is_generated_parent": True,
                "borrowed_or_catalogue_text": False,
                "exact_phrase_collision_count_in_runs_experiments_docs_data_tests": 0,
                "punctuation_changed_existing_letter_tape": False,
                "finished_tape_reversal_used_to_construct": False,
                "new_scene_content": True,
                "whole_word_order_mirror": word_order_mirror,
            },
            "repair_debt": {
                "inserted_scene_pair_is_a_proper_palindromic_span": True,
                "inherited_proper_palindromic_spans": True,
                "inherited_repeated_scaffolding": True,
                "surrounding_prose_human_readable_certified": False,
                "blinded_readers_run": False,
                "effect": "retain as a permissive exact length frontier; do not call reader-worthy or promote without surface repair and blinded evaluation",
            },
        },
        "novelty_preflight": {
            "status": "passed_for_exact_rendered_scene_pair",
            "search_roots": ["runs/", "experiments/", "docs/", "data/", "tests/"],
            "queries": [LEFT_SCENE, RIGHT_SCENE, left_tape, right_tape],
            "prior_exact_phrase_collisions": 0,
            "algorithm_family_overlap_disclosed": True,
        },
        "evaluation": {
            "exact_closure": "one 17-character residual closed exactly",
            "yield": "one 34-letter child, +34 over its 586-letter parent",
            "scaling_evidence": "none from a single authored equation",
            "readability_evidence": "none; no humans rated this full 620-letter output",
            "interpretation": "Candidate-producing incumbent seam edit, not a scalable search result or readability certification.",
        },
        "next_reader_facing_test": {
            "trigger": "only after the surrounding prose is sufficiently coherent for a fair reading",
            "design": "randomized blinded order; show the intact candidate and a shuffled-letter control beside intact English prose controls; collect independent understanding and preference ratings; include a reproducible rater packet",
            "current_candidate_ready": False,
        },
        "next_construction": {
            "target": "the remaining A-tub/question event seam in the 620-letter child, preserving the 586 parent and the non-mirror 620 child",
            "operator": "rederive the wider locative/event pair while carrying valency and character residual together; require a determiner/noun boundary to resegment across the mirror rather than pairing whole reversed words",
            "avoid": ["fixed 13-letter question probe", "boundary-aligned whole-word mirrors", "mirrored deletion as length progress", "fresh-seed search", "reader study on obviously rough whole-line prose"],
        },
    }


def main() -> None:
    payload = build_payload()
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload["evaluation"], sort_keys=True))
    print(payload["candidate"]["sha256"])
    print(payload["candidate"]["rendered"])


if __name__ == "__main__":
    main()
