"""Replace the remaining A-tub shell with a wider 30-letter scene pair.

The edit rederives a 27-letter left/right span as 30 letters per side. The
left's locative and determiner boundaries cross under reversal, yielding
complete clauses rather than a question fragment or whole-word mirror.
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
OUT = ROOT / "runs" / "incumbent-620-wide-locative-seam-expansion-20260923.json"
PARENT_ID = "live-center-scene-nora-tama-620"
PARENT_SHA256 = "7dcd33846b8cfe54eca664e3b3be3d27774c486f6e7b054f49aa697047f4f621"
LEFT_START = 108
OLD_SPAN = 27
LEFT_SCENE = "Nora stops a ram on a mat. Leon saw a rat"
RIGHT_SCENE = "Tara was Noel. Tama, no! Mara spots Aron"
LEFT_EQUATION = "norastopsaramonamatleonsawarat"
RIGHT_EQUATION = "tarawasnoeltamanomaraspotsaron"


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
    assert len(left) == len(right)
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
                "owner": "left-locative-scene-emitter",
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

    old_left = tape("Mara stops rats. A tub? He maps Nora")
    old_right = tape("Aron, spam. Eh, but a star spots Aram")
    left = tape(LEFT_SCENE)
    right = tape(RIGHT_SCENE)
    left_start = parent_tape.find(old_left)
    assert left_start == LEFT_START
    right_start = len(parent_tape) - left_start - OLD_SPAN
    assert len(old_left) == len(old_right) == OLD_SPAN
    assert parent_tape[left_start : left_start + OLD_SPAN] == old_left
    assert parent_tape[right_start : right_start + OLD_SPAN] == old_right
    assert old_left == old_right[::-1]
    assert left == LEFT_EQUATION
    assert right == RIGHT_EQUATION
    assert len(left) == len(right) == 30
    assert left == right[::-1]

    pair_word_mirror = is_boundary_aligned_word_mirror(
        tokenize(f"{LEFT_SCENE}. {RIGHT_SCENE}.")
    )
    assert not pair_word_mirror

    offsets = raw_offsets(parent_rendered)
    replacements = [
        (
            offsets[left_start],
            offsets[left_start + OLD_SPAN - 1] + 1,
            LEFT_SCENE,
        ),
        (
            offsets[right_start],
            offsets[right_start + OLD_SPAN - 1] + 1,
            RIGHT_SCENE,
        ),
    ]
    rendered = parent_rendered
    for raw_start, raw_stop, replacement in sorted(replacements, reverse=True):
        rendered = rendered[:raw_start] + replacement + rendered[raw_stop:]

    candidate_tape = tape(rendered)
    forward_sha = hashlib.sha256(candidate_tape.encode("ascii")).hexdigest()
    reverse_sha = hashlib.sha256(candidate_tape[::-1].encode("ascii")).hexdigest()
    trace = live_trace(left, right)
    assert candidate_tape == (
        parent_tape[:left_start]
        + left
        + parent_tape[left_start + OLD_SPAN : right_start]
        + right
        + parent_tape[right_start + OLD_SPAN :]
    )
    assert len(candidate_tape) == 626
    assert two_pointer_exact(candidate_tape)
    assert is_palindrome(rendered)
    assert normalize(rendered) == candidate_tape
    assert forward_sha == reverse_sha
    assert rendered.count("A tub?") == 1
    assert rendered.count("Mara stops rats") == 1
    assert rendered.count("star spots Aram") == 1

    return {
        "experiment_id": "incumbent-620-wide-locative-seam-expansion-20260923",
        "method": "candidate-specific wider 27-to-30-letter locative scene rederivation with a live 30-character residual",
        "method_scope": {
            "candidate_specific": True,
            "general_algorithm_claim": False,
            "overlap_note": "Seam repair and center/residual methods are established families; the distinct action here is one wider replacement of the actual A-tub event span with a newly authored, cross-boundary scene pair.",
        },
        "status": "exact_626_letter_child_with_remaining_readability_debt",
        "parent": {
            "artifact": str(PARENT.relative_to(ROOT)),
            "id": PARENT_ID,
            "letters": len(parent_tape),
            "sha256": parent_hash,
            "replaced_left_span": [left_start, left_start + OLD_SPAN],
            "replaced_right_span": [right_start, right_start + OLD_SPAN],
        },
        "candidate": {
            "id": "wide-locative-nora-tama-scene-626",
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
                "old_left": old_left,
                "old_right": old_right,
                "left_scene": LEFT_SCENE,
                "right_scene": RIGHT_SCENE,
                "left_tape": left,
                "right_obligation": right,
                "equation": f"{left} = reverse({right})",
                "owner": "left-locative-scene-emitter",
                "trace": trace,
                "final_owner": None,
                "final_residual": "",
                "committed_character_contradictions": 0,
                "typed_clause_pairs": [
                    {
                        "left": "Nora stops a ram on a mat",
                        "right": "Tama, no! Mara spots Aron",
                        "left_letters": 19,
                        "right_letters": 19,
                    },
                    {
                        "left": "Leon saw a rat",
                        "right": "Tara was Noel",
                        "left_letters": 11,
                        "right_letters": 11,
                    },
                ],
                "cross_boundary_resegmentation": [
                    {"left_tokens": ["mat", "a"], "right_token": "Tama"},
                    {"left_tokens": ["on"], "right_token": "no"},
                    {"left_tokens": ["ram", "a"], "right_token": "Mara"},
                    {"left_tokens": ["stops"], "right_token": "spots"},
                    {"left_tokens": ["Nora"], "right_token": "Aron"},
                    {"left_tokens": ["rat", "a"], "right_token": "Tara"},
                    {"left_tokens": ["saw"], "right_token": "was"},
                    {"left_tokens": ["Leon"], "right_token": "Noel"},
                ],
                "boundary_aligned_word_order_mirror": pair_word_mirror,
            },
            "repair": {
                "before_left": "Mara stops rats. A tub? He maps Nora.",
                "after_left": "Nora stops a ram on a mat. Leon saw a rat.",
                "before_right": "Aron, spam. Eh, but a star spots Aram.",
                "after_right": "Tara was Noel. Tama, no! Mara spots Aron.",
                "a_tub_question_occurrences_before": 2,
                "a_tub_question_occurrences_after": 1,
                "interpretation": "The fragmentary question is replaced by complete locative and event clauses; a reciprocal identity/action scene occupies its reflected side.",
            },
            "provenance": {
                "source_is_generated_parent": True,
                "borrowed_or_catalogue_text": False,
                "exact_phrase_collision_count_in_runs_experiments_docs_data_tests": 0,
                "punctuation_rewritten_inside_selected_span": True,
                "punctuation_changes_normalized_letter_tape": False,
                "finished_tape_reversal_used_to_construct": False,
                "new_event_content": True,
                "whole_word_order_mirror": False,
            },
            "repair_debt": {
                "inserted_scene_pair_is_a_proper_palindromic_span": True,
                "remaining_repeated_scaffolding": True,
                "other_rough_prose": True,
                "surrounding_prose_human_readable_certified": False,
                "blinded_readers_run": False,
                "effect": "preserve as a permissive exact length child; no readability certification or reader promotion",
            },
        },
        "novelty_preflight": {
            "status": "passed_for_exact_rendered_scene_pair",
            "search_roots": ["runs/", "experiments/", "docs/", "data/", "tests/"],
            "queries": [LEFT_SCENE, RIGHT_SCENE, left, right],
            "prior_exact_phrase_collisions": 0,
            "algorithm_family_overlap_disclosed": True,
        },
        "evaluation": {
            "exact_closure": "one 30-character residual closed exactly, decomposable into 19- and 11-character event equations",
            "yield": "one 626-letter exact child, +6 over the 620 parent and +58 over the preserved 568-letter lineage",
            "fragment_reduction": "one A-tub question shell removed; one remains",
            "scaling_evidence": "none from one authored seam equation",
            "readability_evidence": "none; human readers have not rated this full output",
            "interpretation": "A candidate-producing wider seam edit with complete local clauses and genuine boundary crossing; the whole-line prose remains a work in progress.",
        },
        "next_reader_facing_test": {
            "trigger": "only after the remaining repeated shell and outer event fragments are repaired enough to give the full line a fair reading",
            "design": "randomized blinded order; intact candidate and shuffled-letter control with intact English controls; independent comprehension and preference ratings; provide a reproducible rater package",
            "current_candidate_ready": False,
        },
        "next_construction": {
            "target": "the remaining A-tub question fragment and adjacent repeated event pair in the 626-letter child",
            "operator": "rederive the next wider complete scene arc with the same dual state: event valency plus live character residual; exclude whole-word mirrors and require distinct event content",
            "avoid": ["fixed 13-letter question probes", "whole-word mirror controls", "same clause-bank sweep", "fresh seed search", "programmatic readability certification"],
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
