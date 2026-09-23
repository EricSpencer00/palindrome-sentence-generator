"""Replace one repeated event seam in the exact 620-letter lineage.

The 13-letter equation changes ``Mara stops rats`` / ``star spots Aram`` to
``Noel stops flow`` / ``wolf spots Leon``. It is a local repair, not a new
general search algorithm or readability claim.
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


PARENT = ROOT / "runs" / "incumbent-586-live-center-scene-extension-20260923.json"
OUT = ROOT / "runs" / "incumbent-620-repeated-shell-repair-20260923.json"
PARENT_ID = "live-center-scene-nora-tama-620"
PARENT_SHA256 = "7dcd33846b8cfe54eca664e3b3be3d27774c486f6e7b054f49aa697047f4f621"
LEFT_START = 108
SPAN = 13
OLD_LEFT = "Mara stops rats"
NEW_LEFT = "Noel stops flow"
OLD_RIGHT = "star spots Aram"
NEW_RIGHT = "wolf spots Leon"


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


def raw_offsets(rendered: str) -> list[int]:
    return [
        index
        for index, character in enumerate(rendered)
        if character.isascii() and character.isalpha()
    ]


def live_trace(left: str, right: str) -> list[dict[str, Any]]:
    assert len(left) == len(right) == SPAN
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
                "owner": "left-event-emitter",
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
    assert len(parent_tape) == 620
    assert parent_hash == PARENT_SHA256
    assert two_pointer_exact(parent_tape)
    assert is_palindrome(parent_rendered)
    assert normalize(parent_rendered) == parent_tape

    left = tape(OLD_LEFT)
    right = tape(OLD_RIGHT)
    left_new = tape(NEW_LEFT)
    right_new = tape(NEW_RIGHT)
    right_start = len(parent_tape) - LEFT_START - SPAN
    assert parent_tape[LEFT_START : LEFT_START + SPAN] == left
    assert parent_tape[right_start : right_start + SPAN] == right
    assert left == right[::-1]
    assert len(left_new) == len(right_new) == SPAN
    assert left_new == right_new[::-1]
    phrase_pair = tokenize(f"{NEW_LEFT}. A {NEW_RIGHT}.")
    phrase_pair_word_order_mirror = is_boundary_aligned_word_mirror(phrase_pair)
    assert phrase_pair_word_order_mirror

    offsets = raw_offsets(parent_rendered)
    spans = [
        (offsets[LEFT_START], offsets[LEFT_START + SPAN - 1] + 1, NEW_LEFT),
        (offsets[right_start], offsets[right_start + SPAN - 1] + 1, NEW_RIGHT),
    ]
    rendered = parent_rendered
    for start, stop, replacement in sorted(spans, reverse=True):
        rendered = rendered[:start] + replacement + rendered[stop:]

    candidate_tape = tape(rendered)
    forward_sha = hashlib.sha256(candidate_tape.encode("ascii")).hexdigest()
    reverse_sha = hashlib.sha256(candidate_tape[::-1].encode("ascii")).hexdigest()
    trace = live_trace(left_new, right_new)
    assert len(candidate_tape) == len(parent_tape) == 620
    assert candidate_tape[LEFT_START : LEFT_START + SPAN] == left_new
    assert candidate_tape[right_start : right_start + SPAN] == right_new
    assert two_pointer_exact(candidate_tape)
    assert is_palindrome(rendered)
    assert normalize(rendered) == candidate_tape
    assert forward_sha == reverse_sha
    assert rendered.count("Mara stops rats") == 1
    assert rendered.count("star spots Aram") == 1
    assert rendered.count("Noel stops flow") == 1
    assert rendered.count("wolf spots Leon") == 1

    return {
        "experiment_id": "incumbent-620-repeated-shell-repair-20260923",
        "method": "single incumbent-specific 13-character repeated-event substitution with live residual ownership",
        "method_scope": {
            "candidate_specific": True,
            "general_algorithm_claim": False,
            "overlap_note": "Semantic seam repair is an established family; this edit changes one duplicated event pair on the 620-letter parent and does not count as a new general method family.",
        },
        "status": "exact_same_length_control_rejected_for_word_order_symmetry",
        "parent": {
            "artifact": str(PARENT.relative_to(ROOT)),
            "id": PARENT_ID,
            "letters": len(parent_tape),
            "sha256": parent_hash,
            "left_span": [LEFT_START, LEFT_START + SPAN],
            "right_span": [right_start, right_start + SPAN],
        },
        "candidate": {
            "id": "noel-stops-flow-wolf-spots-leon-620-rejected-control",
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
                "before_left": left,
                "before_right": right,
                "after_left": left_new,
                "after_right": right_new,
                "equation": f"{left_new} = reverse({right_new})",
                "owner": "left-event-emitter",
                "trace": trace,
                "final_owner": None,
                "final_residual": "",
                "committed_character_contradictions": 0,
            },
            "repair": {
                "before": [OLD_LEFT, OLD_RIGHT],
                "after": [NEW_LEFT, f"A {NEW_RIGHT}"],
                "duplicate_occurrences_before": 2,
                "duplicate_occurrences_after": 1,
                "interpretation": "Noel stops a flow; a wolf spots Leon. The paired clauses are complete, though not yet integrated into coherent long discourse.",
            },
            "provenance": {
                "source_is_generated_parent": True,
                "borrowed_or_catalogue_text": False,
                "exact_phrase_collision_count_in_runs_experiments_docs_data_tests": 0,
                "punctuation_changed_existing_letter_tape": False,
                "finished_tape_reversal_used_to_construct": False,
                "new_event_content": True,
                "whole_word_order_mirror": phrase_pair_word_order_mirror,
            },
            "repair_debt": {
                "inherited_proper_palindromic_spans": True,
                "inherited_repeated_scaffolding": True,
                "replacement_pair_is_boundary_aligned_word_order_mirror": True,
                "admitted_to_working_frontier": False,
                "surrounding_prose_human_readable_certified": False,
                "blinded_readers_run": False,
                "effect": "reject this repair as a shortcut; preserve only as a negative control for the seam-selection rule",
            },
        },
        "novelty_preflight": {
            "status": "passed_for_exact_replacement_phrases",
            "search_roots": ["runs/", "experiments/", "docs/", "data/", "tests/"],
            "queries": [NEW_LEFT, NEW_RIGHT, left_new, right_new],
            "prior_exact_phrase_collisions": 0,
            "algorithm_family_overlap_disclosed": True,
        },
        "evaluation": {
            "exact_closure": "one 13-character residual closed exactly",
            "yield": "same-length exact control; it removes one repeated event shell but is rejected because the replacement is boundary-aligned word-order symmetry",
            "scaling_evidence": "none from a single seam edit",
            "readability_evidence": "none; full candidate has not been rated by humans",
            "failure_signature": "Noel stops flow / wolf spots Leon reverses by whole words in mirrored order.",
            "interpretation": "Do not promote. Require real boundary-crossing resegmentation in the next repaired seam.",
        },
        "next_reader_facing_test": {
            "trigger": "after the remaining repeated shell and A-tub question seam are replaced by coherent content",
            "design": "randomized blinded order; intact full prose and shuffled-letter control plus intact English controls; independent understanding and preference ratings; release the rater instructions, candidate provenance, normalization, and analysis script",
            "current_candidate_ready": False,
        },
        "next_construction": {
            "target": "the second remaining A-tub/question event boundary in the 620-letter child",
            "operator": "rederive a wider complete locative/event pair across the question seam, carrying valency and character residual together; require a determiner/noun boundary to resegment across the mirror rather than pairing whole reversed words",
            "avoid": ["same phrase-bank substitutions", "mirrored deletion as length progress", "fresh seeds", "reader certification by automatic scores"],
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
