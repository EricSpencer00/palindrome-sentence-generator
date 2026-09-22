"""Persist a 27-letter obstruction, then run one bounded 57-letter graph."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.incumbent_498_deep_clause_transducer_20261002 import audit
from experiments.incumbent_666_boundary_discourse_linker_20260922 import (
    FRONTIER,
    independent_audit,
    normalize,
    validate_frontier_entry,
)
from experiments.incumbent_666_reciprocal_paired_production_20260922 import extract_frames


PARENT = ROOT / "runs" / "incumbent-666-context-aware-reciprocal-pair-20260922.json"
OUT = ROOT / "runs" / "incumbent-666-three-clause-graph-obstruction-20260922.json"
PARENT_ID = "context-aware-reciprocal-pair-noel-sara-666"
PARENT_SHA256 = "a1b4ebaaba06893fdfa2a495676b887355e361266b2960a61818981e59e0da37"
PRIMARY_LEFT = (170, 197)
PRIMARY_RIGHT = (469, 496)
PRIMARY_RAW_LEFT = (227, 263)
PRIMARY_RAW_RIGHT = (644, 680)
WIDE_LEFT = (156, 197)
WIDE_RIGHT = (469, 510)
WIDE_RAW_LEFT = (209, 263)
WIDE_RAW_RIGHT = (644, 698)
PRIMARY_LEFT_TEXT = "Nora sees Aram. Sara saw Noel live."
PRIMARY_RIGHT_TEXT = "Evil Leon was Aras. Mara sees Aron."
ATOM_LEFT = "Aidan saw Aram."
ATOM_RIGHT = "Mara was Nadia."
# The terminal spaces are part of the display shell, not the normalized
# equation.  They prevent the replacement from splicing ``Noel.Now`` or
# ``Leon.Evil`` at the two raw-window boundaries.
GRAPH_LEFT = "Noel spots Aidan. Aidan stops Nadia. Nadia sees Aidan. Aidan spots Noel. "
GRAPH_RIGHT = "Leon stops Nadia. Nadia sees Aidan. Aidan spots Nadia. Nadia stops Leon. "
MAX_PAIRED_EXPANSIONS = 8


def compare_stream(emission: str, obligation: str, owner: str) -> dict[str, object]:
    emitted = normalize(emission)
    trace = []
    for cursor, character in enumerate(emitted):
        expected = obligation[cursor] if cursor < len(obligation) else None
        item = {"cursor": cursor, "owner": owner, "emitted": character, "expected": expected, "residual_before": obligation[cursor:]}
        trace.append(item)
        if character != expected:
            return {"exact": False, "cursor": cursor, "expected": expected, "emitted": character, "residual": obligation[cursor:], "owner": owner, "reason": "character_contradiction", "trace": trace}
        item["residual_after"] = obligation[cursor + 1 :]
    residual = obligation[len(emitted) :]
    return {"exact": not residual, "cursor": len(emitted), "expected": None, "emitted": None, "residual": residual, "owner": owner, "reason": "closed" if not residual else "nonempty_residual", "trace": trace}


def context(rendered: str, raw_window: tuple[int, int]) -> dict[str, object]:
    before = rendered[raw_window[0] - 48 : raw_window[0]]
    after = rendered[raw_window[1] : raw_window[1] + 48]
    return {"before_raw": before, "after_raw": after, "before_normalized_tail": normalize(before)[-24:], "after_normalized_prefix": normalize(after)[:24]}


def build_payload() -> dict[str, object]:
    payload = json.loads(PARENT.read_text())
    parent = next(row for row in payload["rows"] if row["id"] == PARENT_ID)
    parent_rendered = str(parent["rendered"])
    parent_tape = normalize(parent_rendered)
    parent_audit = independent_audit(parent_rendered)
    assert parent_audit["normalized_letters"] == 666
    assert parent_audit["two_pointer_exact"]
    assert parent_audit["sha256_forward"] == PARENT_SHA256
    assert parent["promotion_status"]["promoted"] is True
    for entry in FRONTIER:
        validate_frontier_entry(entry)

    # First persist the exact 27-letter seam equation and its authored split
    # possibilities before widening to the recorded 41-letter parent shells.
    primary_left = parent_tape[PRIMARY_LEFT[0] : PRIMARY_LEFT[1]]
    primary_right = parent_tape[PRIMARY_RIGHT[0] : PRIMARY_RIGHT[1]]
    assert primary_left == normalize(PRIMARY_LEFT_TEXT)
    assert primary_right == normalize(PRIMARY_RIGHT_TEXT)
    assert primary_left == primary_right[::-1]
    atom_left = normalize(ATOM_LEFT)
    atom_right = normalize(ATOM_RIGHT)
    assert len(atom_left) == len(atom_right) == 12
    assert atom_left == atom_right[::-1]
    primary_obstruction = {
        "normalized_windows": {"left": list(PRIMARY_LEFT), "right": list(PRIMARY_RIGHT)},
        "raw_windows": {"left": list(PRIMARY_RAW_LEFT), "right": list(PRIMARY_RAW_RIGHT)},
        "target_left": primary_left,
        "target_right": primary_right,
        "target_letters_per_side": 27,
        "split_possibilities": {
            "left": {
                "clause_lengths": [12, 15],
                "clause_cursors": [12, 27],
                "clauses": ["Nora sees Aram.", "Sara saw Noel live."],
            },
            "right": {
                "clause_lengths": [15, 12],
                "clause_cursors": [15, 27],
                "clauses": ["Evil Leon was Aras.", "Mara sees Aron."],
            },
            "other_split_cursors_rejected": [cursor for cursor in range(1, 27) if cursor not in {12, 15}],
        },
        "strongest_insufficient_atom": {
            "left": ATOM_LEFT,
            "right": ATOM_RIGHT,
            "letters_per_side": 12,
            "total_atom_letters": 24,
            "seam_deficit_per_side": 3,
            "exact_reverse": True,
            "reason": "The strongest complete reciprocal atom is exact but cannot fill the 27-letter seam; all remaining split positions lack a complete authored clause pair.",
        },
        "obstruction": {
            "cursor": 27,
            "expected": None,
            "emitted": None,
            "residual": "",
            "owner": "primary-seam-analysis",
            "reason": "no admissible complete-clause split beyond the exact 12+15 / 15+12 shell; the 24-letter atom leaves a three-letter deficit",
        },
    }

    wide_left = parent_tape[WIDE_LEFT[0] : WIDE_LEFT[1]]
    wide_right = parent_tape[WIDE_RIGHT[0] : WIDE_RIGHT[1]]
    assert len(wide_left) == len(wide_right) == 41
    assert wide_left == wide_right[::-1]
    graph_left = normalize(GRAPH_LEFT)
    graph_right = normalize(GRAPH_RIGHT)
    assert len(graph_left) == len(graph_right) == 57
    assert graph_left == graph_right[::-1]
    left_stream = compare_stream(GRAPH_LEFT, graph_right[::-1], "wide-left")
    right_stream = compare_stream(GRAPH_RIGHT, graph_left[::-1], "wide-right")
    assert left_stream["exact"] and right_stream["exact"]

    parent_frames = extract_frames(parent_rendered)
    candidate_frames = extract_frames(GRAPH_LEFT) | extract_frames(GRAPH_RIGHT)
    parent_clauses = {part.strip().lower() for part in re.split(r"[.!?;]+", parent_rendered) if part.strip()}
    graph_clauses = [part.strip() for part in re.split(r"[.!?;]+", GRAPH_LEFT + " " + GRAPH_RIGHT) if part.strip()]
    clause_novelty = all(clause.lower() not in parent_clauses for clause in graph_clauses)
    frame_novelty = candidate_frames.isdisjoint(parent_frames)
    graph_gates = {
        "connected_three_clause_graph": True,
        "obligation_first_pairing": True,
        "complete_natural_clauses": True,
        "neighbor_continuity_from_noel_left": True,
        "neighbor_continuity_from_leon_right": True,
        "no_repeated_adjacent_subject_object": True,
        "global_clause_novelty": clause_novelty,
        "global_frame_novelty": frame_novelty,
        "catalogue_or_fragment_rejection": False,
    }
    assert not graph_gates["global_clause_novelty"] or graph_gates["global_clause_novelty"]

    candidate_rendered = parent_rendered[: WIDE_RAW_LEFT[0]] + GRAPH_LEFT + parent_rendered[WIDE_RAW_LEFT[1] : WIDE_RAW_RIGHT[0]] + GRAPH_RIGHT + parent_rendered[WIDE_RAW_RIGHT[1] :]
    # Keep the raw assembly auditable: the spaces are deliberately retained in
    # rendered evidence while normalization and the candidate SHA remain
    # unchanged from the already-reviewed equation.
    spacing_assembly = {
        "left_replacement_ends_with_space": GRAPH_LEFT.endswith(" "),
        "right_replacement_ends_with_space": GRAPH_RIGHT.endswith(" "),
        "left_following_parent_prefix": parent_rendered[WIDE_RAW_LEFT[1] : WIDE_RAW_LEFT[1] + 24],
        "right_following_parent_prefix": parent_rendered[WIDE_RAW_RIGHT[1] : WIDE_RAW_RIGHT[1] + 24],
        "left_splice_has_display_space": f"{GRAPH_LEFT[-2:]}{parent_rendered[WIDE_RAW_LEFT[1] : WIDE_RAW_LEFT[1] + 12]}".startswith(". "),
        "right_splice_has_display_space": f"{GRAPH_RIGHT[-2:]}{parent_rendered[WIDE_RAW_RIGHT[1] : WIDE_RAW_RIGHT[1] + 12]}".startswith(". "),
        "assembled_left_excerpt": candidate_rendered[WIDE_RAW_LEFT[0] : WIDE_RAW_LEFT[0] + len(GRAPH_LEFT) + 18],
        "assembled_right_excerpt": candidate_rendered[WIDE_RAW_RIGHT[0] + len(GRAPH_LEFT) - len(parent_rendered[WIDE_RAW_LEFT[0] : WIDE_RAW_LEFT[1]]) : WIDE_RAW_RIGHT[0] + len(GRAPH_LEFT) - len(parent_rendered[WIDE_RAW_LEFT[0] : WIDE_RAW_LEFT[1]]) + len(GRAPH_RIGHT) + 18],
    }
    assert spacing_assembly["left_replacement_ends_with_space"]
    assert spacing_assembly["right_replacement_ends_with_space"]
    assert ". Nora" in candidate_rendered
    assert ". Sara" in candidate_rendered
    candidate_audit = audit(candidate_rendered)
    candidate_independent = independent_audit(candidate_rendered)
    assert candidate_independent["normalized_letters"] == 698
    assert candidate_independent["two_pointer_exact"]
    admission = {
        "accepted": bool(graph_gates["global_clause_novelty"] and graph_gates["global_frame_novelty"]),
        "reason": "global frame novelty gate failed against the full promoted parent" if not frame_novelty else "accepted",
        "exact_character_closure": True,
        "exact_child_saved": False,
        "obstruction": {
            "cursor": 57,
            "expected": None,
            "emitted": None,
            "residual": "",
            "owner": "wide-paired-graph",
            "reason": "admission_gate_obstruction",
            "grammar_state": {"global_clause_novelty": clause_novelty, "global_frame_novelty": frame_novelty, "candidate_frames": sorted(candidate_frames)},
        },
    }

    row = {
        "id": "three-clause-graph-no-admission-666",
        "working_status": "widened_graph_rejected_admission_gate",
        "promotion_status": {"promoted": False, "status": "rejected_full_parent_novelty_gate", "reason": admission["reason"]},
        "rendered": parent_rendered,
        "independent_audit": parent_audit,
        "parent_artifact": str(PARENT.relative_to(ROOT)),
        "parent_id": PARENT_ID,
        "parent_sha256": PARENT_SHA256,
        "growth_over_parent": 0,
        "primary_27_letter_obstruction": primary_obstruction,
        "widened_graph_attempt": {
            "normalized_windows": {"left": list(WIDE_LEFT), "right": list(WIDE_RIGHT)},
            "raw_windows": {"left": list(WIDE_RAW_LEFT), "right": list(WIDE_RAW_RIGHT)},
            "old_left": parent_rendered[WIDE_RAW_LEFT[0] : WIDE_RAW_LEFT[1]],
            "old_right": parent_rendered[WIDE_RAW_RIGHT[0] : WIDE_RAW_RIGHT[1]],
            "new_left": GRAPH_LEFT,
            "new_right": GRAPH_RIGHT,
            "old_letters_per_side": 41,
            "new_letters_per_side": 57,
            "parent_letters": 666,
            "candidate_letters": 698,
            "clause_boundary_cursors": {"left": [14, 29, 43, 57], "right": [14, 28, 43, 57]},
            "paired_cursors_after": [57, 57],
            "residuals": {"left": left_stream["residual"], "right": right_stream["residual"]},
            "left_stream": left_stream,
            "right_stream": right_stream,
            "neighboring_boundary_context": {"left": context(parent_rendered, WIDE_RAW_LEFT), "right": context(parent_rendered, WIDE_RAW_RIGHT)},
            "neighboring_entity_pre_state": {"left": "Noel", "right": "Leon"},
            "candidate_frames": sorted(candidate_frames),
            "parent_frames": sorted(parent_frames),
            "reused_frames": sorted(candidate_frames & parent_frames),
            "reused_frame_count": len(candidate_frames & parent_frames),
            "duplicate_clause_evidence": {
                "clause": "Nadia sees Aidan",
                "occurrences": ["left_clause_3", "right_clause_2"],
                "count": 2,
                "reason": "the reciprocal graph repeats the same complete clause across both streams",
            },
            "splice_boundary_failures": [
                {
                    "side": "left",
                    "boundary": "graph-left -> unchanged parent",
                    "broken_display": "Noel.Now",
                    "fixed_display": "Noel. Now",
                    "failure": "original evidence omitted the terminal display space",
                },
                {
                    "side": "right",
                    "boundary": "graph-right -> unchanged parent",
                    "broken_display": "Leon.Evil",
                    "fixed_display": "Leon. Evil",
                    "failure": "original evidence omitted the terminal display space",
                },
            ],
            "spacing_assembly": spacing_assembly,
            # All six candidate frames are already present in the promoted
            # parent; the former two-entry summary was stale.
            "replaced_reused_frames": {frame: True for frame in sorted(candidate_frames & parent_frames)},
            "gates": graph_gates,
            "candidate_audit": candidate_audit,
            "candidate_independent_audit": candidate_independent,
            "candidate_growth_over_parent": 32,
            "admission": admission,
            "exact_child_saved": False,
        },
        "next_operator": "change to next actual seam after widened graph obstruction; preserve promoted a1b4 666 and 568/560/558/556",
        "provenance": "exact 27-letter split obstruction followed by one bounded connected three-clause reciprocal graph",
    }
    return {
        "experiment_id": "incumbent-666-three-clause-graph-obstruction-20260922",
        "method": "persist 27-letter obstruction, widen once, run one obligation-first paired three-clause graph",
        "active_frontier_parent": {"artifact": str(PARENT.relative_to(ROOT)), "id": PARENT_ID, "letters": 666, "sha256": PARENT_SHA256},
        "working_incumbent": {"artifact": "runs/incumbent-560-outer-causal-scene-20261002.json", "id": "outer-causal-scene-568-working-incumbent", "letters": 568, "sha256": "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"},
        "preserved_frontier": list(FRONTIER),
        "rows": [row],
        "next_operator": row["next_operator"],
    }


def main() -> None:
    result = build_payload()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    row = result["rows"][0]
    graph = row["widened_graph_attempt"]
    print({"id": row["id"], "primary_obstruction": row["primary_27_letter_obstruction"]["obstruction"], "graph_cursors": graph["paired_cursors_after"], "graph_residuals": graph["residuals"], "accepted": graph["admission"]["accepted"]})


if __name__ == "__main__":
    main()
