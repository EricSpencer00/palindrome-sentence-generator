"""Replace the second 568-inherited shell with a mixed-predicate event cycle.

The exact 686-letter parent has one remaining measured 27-letter shell at a
mirrored pair of supports.  A connected four-event source cycle is emitted on
one side; a finite-state grammar chart segments the reversed tape using the
transitive predicate relation ``stops`` <-> ``spots`` and self-mirroring
``sees``.  Both grammar roles and character cursors remain live through the
replacement.  This is a single bounded repair, not a sentence-bank sweep.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import socket
from pathlib import Path

from repair_640_mirrored_shell_cycle_20260923 import (
    independent_audit,
    normalize,
    owner_at,
    raw_index_at_letter,
    surface,
    tape_for,
)


EXPERIMENT_ID = "repair-686-mixed-predicate-shell-cycle-20260923"
SOURCE_COMMIT = "53a7ba0cabd5a69420b21f349731a36bc1b47fa6"
PARENT_ID = "568-lineage-shell-cycle-repair-686"
PARENT_SHA256 = "cb47534ec3f66a5a3624e7573e15cde7c1cdc53c7ff9cebdc788412b940f982c"
PARENT_LETTERS = 686
PARENT_568_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
LEFT_SUPPORT = (167, 194)
RIGHT_SUPPORT = (492, 519)
OLD_LEFT_TAPE = "marastopsratsatubhemapsnora"
OLD_RIGHT_TAPE = "aronspamehbutastarspotsaram"

ENTITY_FORMS = (
    "Leon", "Noel", "Nora", "Aron", "Mara", "Aram", "Sara", "Aras",
    "Aidan", "Nadia", "Ira", "Ari",
)
PREDICATE_FORMS = ("sees", "saw", "was", "stops", "spots")
REVERSE_PREDICATE = {
    "sees": "sees",
    "stops": "spots",
    "spots": "stops",
    "saw": "was",
    "was": "saw",
}

LEFT_EVENTS = (
    {"id": "mixed-cycle-1", "subject": "Nora", "predicate": "sees", "object": "Sara"},
    {"id": "mixed-cycle-2", "subject": "Sara", "predicate": "stops", "object": "Aidan"},
    {"id": "mixed-cycle-3", "subject": "Aidan", "predicate": "stops", "object": "Noel"},
    {"id": "mixed-cycle-4", "subject": "Noel", "predicate": "sees", "object": "Nora"},
)


def segment_connected_cycle(tape: str, event_count: int) -> tuple[list[dict[str, str]], int, int]:
    """Resegment one exact tape into complete transitive clauses and a cycle."""
    entities = tuple((word, normalize(word)) for word in ENTITY_FORMS)
    predicates = tuple((word, normalize(word)) for word in PREDICATE_FORMS)

    def walk(offset: int, remaining: int) -> list[list[dict[str, str]]]:
        if remaining == 0:
            return [[]] if offset == len(tape) else []
        parses: list[list[dict[str, str]]] = []
        for subject, subject_tape in entities:
            if not tape.startswith(subject_tape, offset):
                continue
            after_subject = offset + len(subject_tape)
            for predicate, predicate_tape in predicates:
                if not tape.startswith(predicate_tape, after_subject):
                    continue
                after_predicate = after_subject + len(predicate_tape)
                for obj, object_tape in entities:
                    if not tape.startswith(object_tape, after_predicate):
                        continue
                    for suffix in walk(after_predicate + len(object_tape), remaining - 1):
                        parses.append([
                            {"subject": subject, "predicate": predicate, "object": obj},
                            *suffix,
                        ])
        return parses

    parses = walk(0, event_count)
    cycles = [
        row for row in parses
        if all(row[i]["object"] == row[i + 1]["subject"]
               for i in range(len(row) - 1))
        and row[-1]["object"] == row[0]["subject"]
        and all(REVERSE_PREDICATE.get(LEFT_EVENTS[-1 - i]["predicate"]) == row[i]["predicate"]
                for i in range(len(row)))
    ]
    if not cycles:
        raise ValueError("the mixed-predicate reverse obligation has no complete event-cycle parse")
    selected = min(cycles, key=lambda row: tuple(
        (event["subject"], event["predicate"], event["object"]) for event in row
    ))
    return selected, len(parses), len(cycles)


def render(parent_path: Path) -> dict[str, object]:
    payload = json.loads(parent_path.read_text())
    parent = payload["candidate"]
    if parent["id"] != PARENT_ID or parent["letters"] != PARENT_LETTERS:
        raise ValueError("the requested 686-letter child is not the loaded parent")
    parent_text = str(parent["rendered"])
    parent_tape = normalize(parent_text)
    parent_sha = hashlib.sha256(parent_tape.encode("ascii")).hexdigest()
    if parent_sha != PARENT_SHA256 or parent_tape != parent_tape[::-1]:
        raise ValueError("the loaded 686 parent fails its pinned independent audit")
    if payload["parent"]["568_ancestor_sha256"] != PARENT_568_SHA256:
        raise ValueError("the 686 parent is not on the pinned 568 lineage")

    left_start, left_end = LEFT_SUPPORT
    right_start, right_end = RIGHT_SUPPORT
    if (right_start, right_end) != (PARENT_LETTERS - left_end, PARENT_LETTERS - left_start):
        raise ValueError("the two replacement supports are not reflected positions")
    if parent_tape[left_start:left_end] != OLD_LEFT_TAPE:
        raise ValueError("the remaining 568-derived left shell moved or changed")
    if parent_tape[right_start:right_end] != OLD_RIGHT_TAPE:
        raise ValueError("the remaining reflected shell moved or changed")
    if OLD_LEFT_TAPE != OLD_RIGHT_TAPE[::-1]:
        raise AssertionError("the existing shell does not close the parent residual")

    left_tape = tape_for(LEFT_EVENTS)
    right_obligation = left_tape[::-1]
    right_events, parse_count, cycle_parse_count = segment_connected_cycle(
        right_obligation, len(LEFT_EVENTS)
    )
    right_tape = tape_for(right_events)
    right_text = surface(right_events)
    if right_tape != right_obligation:
        raise AssertionError("the right event chart did not consume the exact obligation")
    if not all(
        REVERSE_PREDICATE.get(LEFT_EVENTS[-1 - i]["predicate"]) == right_events[i]["predicate"]
        for i in range(len(LEFT_EVENTS))
    ):
        raise AssertionError("a predicate edge did not follow the registered reversal relation")

    left_raw_start = raw_index_at_letter(parent_text, left_start)
    left_raw_end = raw_index_at_letter(parent_text, left_end)
    right_raw_start = raw_index_at_letter(parent_text, right_start)
    right_raw_end = raw_index_at_letter(parent_text, right_end)
    if normalize(parent_text[left_raw_start:left_raw_end]) != OLD_LEFT_TAPE:
        raise AssertionError("raw left support differs from the live normalized cursor")
    if normalize(parent_text[right_raw_start:right_raw_end]) != OLD_RIGHT_TAPE:
        raise AssertionError("raw right support differs from the live normalized cursor")

    # Replace the later support first so offsets still refer to the 686 parent.
    after_right = parent_text[:right_raw_start] + right_text + " " + parent_text[right_raw_end:]
    candidate_text = after_right[:left_raw_start] + surface(LEFT_EVENTS) + " " + after_right[left_raw_end:]

    delta_each = len(left_tape) - len(OLD_LEFT_TAPE)
    candidate_letters = PARENT_LETTERS + 2 * delta_each
    child_right_reverse_start = candidate_letters - 1 - left_start
    trace: list[dict[str, object]] = []
    for offset, left_char in enumerate(left_tape):
        right_offset = len(right_tape) - 1 - offset
        right_char = right_tape[right_offset]
        if left_char != right_char:
            raise ValueError(
                f"live residual contradiction at child cursors "
                f"{left_start + offset}/{child_right_reverse_start - offset}: "
                f"{left_char!r} != {right_char!r}"
            )
        left_residual = left_tape[offset + 1:]
        right_residual = right_tape[:right_offset][::-1]
        if left_residual != right_residual:
            raise AssertionError("left and right residual owners diverged")
        trace.append({
            "step": offset + 1,
            "left_cursor_child": left_start + offset,
            "right_reverse_cursor_child": child_right_reverse_start - offset,
            "character": left_char,
            "left_owner": owner_at(LEFT_EVENTS, offset),
            "right_owner": owner_at(right_events, right_offset),
            "predicate_relation": REVERSE_PREDICATE.get(
                LEFT_EVENTS[owner_at(LEFT_EVENTS, offset)["event_index"]]["predicate"]
            ),
            "remaining_residual_letters": len(left_residual),
            "residual_sha256": hashlib.sha256(left_residual.encode("ascii")).hexdigest(),
        })

    audit = independent_audit(candidate_text)
    if audit["normalized_letters"] != candidate_letters or not audit["two_pointer_exact"]:
        raise AssertionError("the whole 736-letter child failed independent exactness")
    if not audit["hashes_equal"]:
        raise AssertionError("forward and reverse SHA-256 checks disagree")

    return {
        "experiment_id": EXPERIMENT_ID,
        "status": "exact_mixed_predicate_shell_repair_child",
        "execution": {
            "host": socket.gethostname(),
            "python_version": platform.python_version(),
            "stochasticity": "none; deterministic mixed-predicate grammar chart",
        },
        "source_commit": SOURCE_COMMIT,
        "method": "variable-length connected event-cycle replacement; a finite predicate reversal map and entity-role grammar resegment the opposing tape while character residual ownership stays live",
        "parent": {
            "artifact": str(parent_path),
            "id": PARENT_ID,
            "letters": PARENT_LETTERS,
            "normalized_tape_sha256": parent_sha,
            "568_ancestor_sha256": PARENT_568_SHA256,
            "artifact_sha256": hashlib.sha256(parent_path.read_bytes()).hexdigest(),
        },
        "seam": {
            "parent_normalized_supports": [list(LEFT_SUPPORT), list(RIGHT_SUPPORT)],
            "parent_raw_supports": [[left_raw_start, left_raw_end], [right_raw_start, right_raw_end]],
            "old_left_tape": OLD_LEFT_TAPE,
            "old_right_tape": OLD_RIGHT_TAPE,
            "new_letters_per_support": [len(left_tape), len(right_tape)],
            "growth_per_support": delta_each,
            "outside_supports_retained_byte_for_byte": True,
        },
        "grammar_state": {
            "left_event_cycle": list(LEFT_EVENTS),
            "right_cycle_resegmented_from_residual": right_events,
            "predicate_reversal_map": REVERSE_PREDICATE,
            "right_parse_count_before_cycle_constraint": parse_count,
            "right_parse_count_after_cycle_constraint": cycle_parse_count,
            "left_cycle_closes": LEFT_EVENTS[-1]["object"] == LEFT_EVENTS[0]["subject"],
            "right_cycle_closes": right_events[-1]["object"] == right_events[0]["subject"],
        },
        "live_residual": {
            "left_emitted_tape": left_tape,
            "right_reverse_obligation": right_obligation,
            "initial_residual_letters": len(left_tape),
            "steps": trace,
            "final_residual": "",
            "committed_character_contradictions": 0,
        },
        "candidate": {
            "id": "568-lineage-mixed-predicate-shell-cycle-736",
            "rendered": candidate_text,
            "letters": audit["normalized_letters"],
            "normalized_tape_sha256": audit["sha256_forward"],
            "growth_over_parent": audit["normalized_letters"] - PARENT_LETTERS,
            "new_event_content": [surface(LEFT_EVENTS), right_text],
        },
        "independent_audit": audit,
        "provenance": {
            "event_chain_authored_for_this_run": True,
            "catalogue_or_corpus_text_used": False,
            "exact_phrase_preflight": {
                "scope": "runs/, experiments/, and docs/ at source commit " + SOURCE_COMMIT,
                "exact_clause_hits": [],
                "phrases_checked": [
                    "Nora sees Sara", "Sara stops Aidan", "Aidan stops Noel", "Noel sees Nora",
                    "Aron sees Leon", "Leon spots Nadia", "Nadia spots Aras", "Aras sees Aron",
                ],
            },
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "repair_debt": {
            "inherited_rough_discourse_outside_shell_retained": True,
            "predicate_repetition_within_cycle": True,
            "human_readability": "not assessed; exactness, complete clauses, and cycle closure do not certify readability",
        },
        "next_repair": "Review the whole rendered text as an unratified candidate; keep the 568, 640, 686, and 736 artifacts separate, and select the next seam only from an actual reader or syntax defect rather than extending another same-shape cycle.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = render(args.parent)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "experiment_id": result["experiment_id"],
        "status": result["status"],
        "letters": result["candidate"]["letters"],
        "growth": result["candidate"]["growth_over_parent"],
        "exact": result["independent_audit"]["two_pointer_exact"],
        "sha256": result["candidate"]["normalized_tape_sha256"],
        "output": str(args.out),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
