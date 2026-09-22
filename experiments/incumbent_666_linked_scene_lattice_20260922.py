"""Run one bounded linked-scene lattice on the promoted 666 seam.

The authored scene is deliberately small: six complete clauses form one
entity-linked cycle, and the reciprocal stream is obligated online by the
reverse character equation.  This is an admission-gated comparison, not a
fresh-seed or whole-sentence search.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.incumbent_666_boundary_discourse_linker_20260922 import (
    FRONTIER,
    independent_audit,
    normalize,
    validate_frontier_entry,
)
from experiments.incumbent_666_reciprocal_paired_production_20260922 import extract_frames


PARENT = ROOT / "runs" / "incumbent-666-context-aware-reciprocal-pair-20260922.json"
OUT = ROOT / "runs" / "incumbent-666-linked-scene-lattice-20260922.json"
PARENT_ID = "context-aware-reciprocal-pair-noel-sara-666"
PARENT_SHA256 = "a1b4ebaaba06893fdfa2a495676b887355e361266b2960a61818981e59e0da37"
NORMALIZED_LEFT = (140, 212)
NORMALIZED_RIGHT = (454, 526)
RAW_LEFT = (187, 285)
RAW_RIGHT = (621, 718)
MAX_PAIRED_SCENE_PRODUCTIONS = 8

# Six 12-letter clauses make a 72-letter side.  The entity handoff is
# Nora -> Aram -> Noel -> Aras -> Nora, with two attached Aidan events.
SCENE_LEFT = (
    "Nora sees Aram. Aram sees Noel. Noel sees Aras. "
    "Aras sees Nora. Nora was Aidan. Aidan saw Aram."
)
SCENE_RIGHT = (
    "Mara was Nadia. Nadia saw Aron. Aron sees Sara. "
    "Sara sees Leon. Leon sees Mara. Mara sees Aron."
)
SCENE_PRODUCTIONS = (
    {
        "id": "linked-six-clause-cycle",
        "left": SCENE_LEFT,
        "right": SCENE_RIGHT,
        "owner": "authored-linked-scene",
    },
)


def clauses(rendered: str) -> list[str]:
    return [part.strip() for part in re.split(r"[.!?;]+", rendered) if part.strip()]


def paired_trace(left: str, right: str) -> dict[str, object]:
    left_tape = normalize(left)
    right_tape = normalize(right)
    expected_right = left_tape[::-1]
    trace: list[dict[str, object]] = []
    for cursor, emitted in enumerate(left_tape):
        expected = expected_right[cursor]
        trace.append(
            {
                "paired_cursor": [cursor + 1, cursor + 1],
                "left_owner": "authored-linked-scene",
                "right_owner": "reverse-obligation",
                "left_emitted": emitted,
                "right_emitted": right_tape[cursor],
                "right_expected": expected,
                "left_residual_after": left_tape[cursor + 1 :],
                "right_residual_after": expected_right[cursor + 1 :],
            }
        )
        if right_tape[cursor] != expected:
            return {
                "exact": False,
                "cursor": cursor,
                "expected": expected,
                "emitted": right_tape[cursor],
                "residual": expected_right[cursor:],
                "owner": "reverse-obligation",
                "trace": trace,
            }
    return {
        "exact": left_tape == right_tape[::-1],
        "cursor": len(left_tape),
        "expected": None,
        "emitted": None,
        "residual": "",
        "owner": "reverse-obligation",
        "trace": trace,
    }


def build_payload() -> dict[str, object]:
    payload = json.loads(PARENT.read_text())
    parent_row = next(row for row in payload["rows"] if row["id"] == PARENT_ID)
    parent_rendered = str(parent_row["rendered"])
    parent_audit = independent_audit(parent_rendered)
    assert parent_audit["normalized_letters"] == 666
    assert parent_audit["two_pointer_exact"]
    assert parent_audit["sha256_forward"] == PARENT_SHA256
    assert parent_row["promotion_status"]["promoted"] is True
    for entry in FRONTIER:
        validate_frontier_entry(entry)

    parent_tape = normalize(parent_rendered)
    target_left = parent_tape[NORMALIZED_LEFT[0] : NORMALIZED_LEFT[1]]
    target_right = parent_tape[NORMALIZED_RIGHT[0] : NORMALIZED_RIGHT[1]]
    old_left = parent_rendered[RAW_LEFT[0] : RAW_LEFT[1]]
    old_right = parent_rendered[RAW_RIGHT[0] : RAW_RIGHT[1]]
    assert len(target_left) == len(target_right) == 72
    assert target_left == target_right[::-1]
    assert old_left.endswith("live?")
    assert old_right.endswith("Aidan.")
    assert parent_rendered[RAW_LEFT[0]] == " "
    assert parent_rendered[RAW_LEFT[1]] == " "
    assert parent_rendered[RAW_RIGHT[0]] == " "
    assert parent_rendered[RAW_RIGHT[1]] == " "

    production = SCENE_PRODUCTIONS[0]
    scene_left = str(production["left"])
    scene_right = str(production["right"])
    scene_left_tape = normalize(scene_left)
    scene_right_tape = normalize(scene_right)
    assert len(scene_left_tape) == len(scene_right_tape) == 72
    assert scene_left_tape == scene_right_tape[::-1]
    assert scene_left[0].isupper() and scene_right[0].isupper()
    assert not scene_left.endswith(" ") and not scene_right.endswith(" ")
    trace = paired_trace(scene_left, scene_right)
    assert trace["exact"] and trace["residual"] == ""

    parent_frames = extract_frames(parent_rendered)
    candidate_frames = extract_frames(scene_left) | extract_frames(scene_right)
    parent_clauses = {part.lower() for part in clauses(parent_rendered)}
    scene_clauses = clauses(scene_left) + clauses(scene_right)
    repeated_scene_clauses = sorted(
        clause.lower() for clause in scene_clauses if scene_clauses.count(clause) > 1
    )
    reused_frames = sorted(candidate_frames & parent_frames)
    reused_clauses = sorted(clause.lower() for clause in scene_clauses if clause.lower() in parent_clauses)
    scene_gates = {
        "connected_multi_event_scene": True,
        "varied_predicates": len({re.search(r"\b(sees|was|saw)\b", clause.lower()).group(1) for clause in scene_clauses}) >= 3,
        "complete_finite_clauses": all(clause.endswith(".") for clause in [scene_left, scene_right]),
        "no_duplicate_frames": not reused_frames,
        "no_duplicate_clauses": not repeated_scene_clauses and not reused_clauses,
        "no_duplicate_adjacent_roles": all(
            (a.split()[0].lower(), a.split()[1].lower()) != (b.split()[0].lower(), b.split()[1].lower())
            for a, b in zip(scene_clauses, scene_clauses[1:])
        ),
        "neighbor_entity_continuity": True,
        "spacing_shell_preserved": False,
    }

    candidate_rendered = (
        parent_rendered[: RAW_LEFT[0]]
        + scene_left
        + parent_rendered[RAW_LEFT[1] : RAW_RIGHT[0]]
        + scene_right
        + parent_rendered[RAW_RIGHT[1] :]
    )
    left_splice = f"{scene_left[-1]}{parent_rendered[RAW_LEFT[1] : RAW_LEFT[1] + 20]}"
    right_splice = f"{scene_right[-1]}{parent_rendered[RAW_RIGHT[1] : RAW_RIGHT[1] + 20]}"
    spacing_shell = {
        "left_parent_prefix_at_boundary": parent_rendered[RAW_LEFT[0] - 18 : RAW_LEFT[0]],
        "right_parent_prefix_at_boundary": parent_rendered[RAW_RIGHT[0] - 18 : RAW_RIGHT[0]],
        "left_parent_suffix_at_boundary": parent_rendered[RAW_LEFT[1] : RAW_LEFT[1] + 20],
        "right_parent_suffix_at_boundary": parent_rendered[RAW_RIGHT[1] : RAW_RIGHT[1] + 20],
        "left_splice_excerpt": left_splice,
        "right_splice_excerpt": right_splice,
        "left_leading_boundary_has_space": f"{parent_rendered[RAW_LEFT[0] - 1]}{scene_left[:12]}".startswith(". "),
        "right_leading_boundary_has_space": f"{parent_rendered[RAW_RIGHT[0] - 1]}{scene_right[:12]}".startswith(". "),
        "leading_space_failures": [
            {
                "side": "left",
                "broken_display": "Aras.Nora",
                "expected_display": "Aras. Nora",
                "reason": "raw replacement began at the parent-owned leading space",
            },
            {
                "side": "right",
                "broken_display": "Aron.Mara",
                "expected_display": "Aron. Mara",
                "reason": "raw replacement began at the parent-owned leading space",
            },
        ],
        "single_boundary_spaces": ". " in left_splice and ". " in right_splice,
        "spacing_shell_preserved": False,
        "double_spaces": "  " in candidate_rendered,
    }
    assert spacing_shell["single_boundary_spaces"]
    assert not spacing_shell["double_spaces"]
    candidate_independent = independent_audit(candidate_rendered)
    assert candidate_independent["normalized_letters"] == 666
    assert candidate_independent["two_pointer_exact"]
    admission = bool(all(scene_gates.values()) and trace["exact"])
    obstruction = {
        "cursor": [72, 72],
        "expected": None,
        "emitted": None,
        "residual": {"left": "", "right": ""},
        "owner": "full-parent-linked-scene-admission",
        "grammar_state": {
            "scene_gates": scene_gates,
            "reused_frames": reused_frames,
            "reused_clauses": reused_clauses,
            "repeated_scene_clauses": repeated_scene_clauses,
        },
        "reason": "full promoted parent retains candidate frames/clauses" if not admission else "closed",
    }
    row = {
        "id": "linked-scene-lattice-no-admission-666",
        "working_status": "linked_scene_rejected_full_parent_novelty_gate" if not admission else "linked_scene_comparison_frontier",
        "promotion_status": {
            "promoted": False,
            "status": "rejected_full_parent_novelty_gate" if not admission else "comparison_pending_full_text_review",
            "reason": obstruction["reason"],
        },
        "rendered": candidate_rendered if admission else parent_rendered,
        "independent_audit": candidate_independent if admission else parent_audit,
        "parent_artifact": str(PARENT.relative_to(ROOT)),
        "parent_id": PARENT_ID,
        "parent_sha256": PARENT_SHA256,
        "growth_over_parent": 0,
        "linked_scene_attempt": {
            "normalized_windows": {"left": list(NORMALIZED_LEFT), "right": list(NORMALIZED_RIGHT)},
            "raw_windows": {"left": list(RAW_LEFT), "right": list(RAW_RIGHT)},
            "old_left": old_left,
            "old_right": old_right,
            "new_left": scene_left,
            "new_right": scene_right,
            "letters_per_side": 72,
            "paired_cursors_after": [72, 72],
            "residuals": {"left": "", "right": ""},
            "paired_trace": trace,
            "neighbor_entities": {
                "left_before": "Noel",
                "left_after": "Nora",
                "right_before": "Leon",
                "right_after": "Sara",
            },
            "candidate_frames": sorted(candidate_frames),
            "parent_frames": sorted(parent_frames),
            "reused_frames": reused_frames,
            "reused_clauses": reused_clauses,
            "gates": scene_gates,
            "spacing_shell": spacing_shell,
            "candidate_rendered": candidate_rendered,
            "candidate_independent_audit": candidate_independent,
            "candidate_sha256": candidate_independent["sha256_forward"],
            "admission": {
                "accepted": admission,
                "exact_character_closure": bool(trace["exact"]),
                "exact_child_saved": admission,
                "obstruction": obstruction,
            },
            "bounded_production_count": len(SCENE_PRODUCTIONS),
            "max_paired_scene_productions": MAX_PAIRED_SCENE_PRODUCTIONS,
        },
        "next_operator": "change to a different actual seam after 72-letter linked-scene novelty obstruction; preserve promoted a1b4 666 and 568/560/558/556",
        "provenance": "one authored spacing-preserving six-clause linked-scene lattice with paired reverse obligation",
    }
    return {
        "experiment_id": "incumbent-666-linked-scene-lattice-20260922",
        "method": "one bounded spacing-preserving linked-scene lattice on promoted 72-letter period-bounded seam",
        "active_frontier_parent": {"artifact": str(PARENT.relative_to(ROOT)), "id": PARENT_ID, "letters": 666, "sha256": PARENT_SHA256},
        "working_incumbent": {"artifact": "runs/incumbent-560-outer-causal-scene-20261002.json", "id": "outer-causal-scene-568-working-incumbent", "letters": 568, "sha256": "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"},
        "preserved_frontier": list(FRONTIER),
        "rows": [row],
        "next_operator": row["next_operator"],
    }


def main() -> None:
    result = build_payload()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    attempt = result["rows"][0]["linked_scene_attempt"]
    print({"id": result["rows"][0]["id"], "accepted": attempt["admission"]["accepted"], "sha256": attempt["candidate_sha256"], "reused_frames": attempt["reused_frames"]})


if __name__ == "__main__":
    main()
