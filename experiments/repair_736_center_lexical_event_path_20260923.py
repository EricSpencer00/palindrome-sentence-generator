"""Replace the malformed central lexical pair with two complete event paths.

This bounded repair opens the exact center of the 736-letter lineage.  The
left path is typed as two connected transitive events.  A role chart finds the
opposing path from the reversed character tape using ``stops`` <-> ``spots``
and ``sees`` <-> ``sees``.  It is intentionally an open path, not another
closed event cycle or a whole-sentence sweep.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
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


EXPERIMENT_ID = "repair-736-center-lexical-event-path-20260923"
SOURCE_COMMIT = "6bfdeac764a49e88aa74985fffb120666b0fac44"
PARENT_ID = "568-lineage-mixed-predicate-shell-cycle-736"
PARENT_SHA256 = "608733c3af8b3d838bc18cfa998ae7202622a4e5abba1aa489733d3dfead78a2"
PARENT_LETTERS = 736
PARENT_568_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
LEFT_SUPPORT = (353, 368)
RIGHT_SUPPORT = (368, 383)
OLD_LEFT_TAPE = "arideliversmaps"
OLD_RIGHT_TAPE = "spamsreviledira"

ENTITY_FORMS = ("Leon", "Noel", "Nora", "Aron", "Mara", "Aram", "Sara", "Aras", "Aidan", "Nadia", "Ira", "Ari")
PREDICATES = ("sees", "stops", "spots")
PREDICATE_REVERSE = {"sees": "sees", "stops": "spots", "spots": "stops"}

LEFT_EVENTS = (
    {"id": "center-path-1", "subject": "Ari", "predicate": "sees", "object": "Leon"},
    {"id": "center-path-2", "subject": "Leon", "predicate": "stops", "object": "Ira"},
)


def segment_connected_path(tape: str, event_count: int) -> tuple[list[dict[str, str]], int, int]:
    entities = tuple((word, normalize(word)) for word in ENTITY_FORMS)
    predicates = tuple((word, normalize(word)) for word in PREDICATES)

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
    linked = [
        row for row in parses
        if all(row[i]["object"] == row[i + 1]["subject"]
               for i in range(len(row) - 1))
        and all(PREDICATE_REVERSE.get(LEFT_EVENTS[-1 - i]["predicate"]) == row[i]["predicate"]
                for i in range(len(row)))
    ]
    if not linked:
        raise ValueError("the central opposing tape has no complete connected event-path parse")
    selected = min(linked, key=lambda row: tuple(
        (event["subject"], event["predicate"], event["object"]) for event in row
    ))
    return selected, len(parses), len(linked)


def build(parent_path: Path) -> dict[str, object]:
    payload = json.loads(parent_path.read_text())
    parent = payload["candidate"]
    if parent["id"] != PARENT_ID or parent["letters"] != PARENT_LETTERS:
        raise ValueError("the requested 736-letter child is not the loaded parent")
    parent_text = str(parent["rendered"])
    parent_tape = normalize(parent_text)
    parent_sha = hashlib.sha256(parent_tape.encode("ascii")).hexdigest()
    if parent_sha != PARENT_SHA256 or parent_tape != parent_tape[::-1]:
        raise ValueError("the loaded 736 parent fails its pinned exactness check")
    if payload["parent"]["568_ancestor_sha256"] != PARENT_568_SHA256:
        raise ValueError("the 736 parent is not on the pinned 568 lineage")

    left_start, left_end = LEFT_SUPPORT
    right_start, right_end = RIGHT_SUPPORT
    if (right_start, right_end) != (PARENT_LETTERS - left_end, PARENT_LETTERS - left_start):
        raise ValueError("the two central supports are not mirrored")
    if parent_tape[left_start:left_end] != OLD_LEFT_TAPE:
        raise ValueError("the central left phrase changed from the recorded baseline")
    if parent_tape[right_start:right_end] != OLD_RIGHT_TAPE:
        raise ValueError("the central right phrase changed from the recorded baseline")
    if OLD_LEFT_TAPE != OLD_RIGHT_TAPE[::-1]:
        raise AssertionError("the old central residual is not exact")

    left_tape = tape_for(LEFT_EVENTS)
    right_obligation = left_tape[::-1]
    right_events, parse_count, linked_parse_count = segment_connected_path(
        right_obligation, len(LEFT_EVENTS)
    )
    right_tape = tape_for(right_events)
    right_text = surface(right_events)
    if right_tape != right_obligation:
        raise AssertionError("the right path did not consume the reversed character tape")

    left_raw_start = raw_index_at_letter(parent_text, left_start)
    left_raw_end = raw_index_at_letter(parent_text, left_end)
    right_raw_start = raw_index_at_letter(parent_text, right_start)
    right_raw_end = raw_index_at_letter(parent_text, right_end)
    if normalize(parent_text[left_raw_start:left_raw_end]) != OLD_LEFT_TAPE:
        raise AssertionError("the raw left cut differs from its normalized character cursor")
    if normalize(parent_text[right_raw_start:right_raw_end]) != OLD_RIGHT_TAPE:
        raise AssertionError("the raw right cut differs from its normalized character cursor")

    # The supports touch at the midpoint; applying the right replacement first
    # preserves both original raw offsets and the exact 736 parent outside them.
    after_right = parent_text[:right_raw_start] + right_text + " " + parent_text[right_raw_end:]
    candidate_text = after_right[:left_raw_start] + surface(LEFT_EVENTS) + " " + after_right[left_raw_end:]

    growth_each = len(left_tape) - len(OLD_LEFT_TAPE)
    candidate_letters = PARENT_LETTERS + 2 * growth_each
    child_right_cursor = candidate_letters - 1 - left_start
    trace: list[dict[str, object]] = []
    for offset, left_char in enumerate(left_tape):
        right_offset = len(right_tape) - 1 - offset
        right_char = right_tape[right_offset]
        if left_char != right_char:
            raise ValueError(
                f"live residual contradiction at child cursors "
                f"{left_start + offset}/{child_right_cursor - offset}: "
                f"{left_char!r} != {right_char!r}"
            )
        left_residual = left_tape[offset + 1:]
        right_residual = right_tape[:right_offset][::-1]
        if left_residual != right_residual:
            raise AssertionError("the left and right character residuals diverged")
        left_owner = owner_at(LEFT_EVENTS, offset)
        trace.append({
            "step": offset + 1,
            "left_cursor_child": left_start + offset,
            "right_reverse_cursor_child": child_right_cursor - offset,
            "character": left_char,
            "left_owner": left_owner,
            "right_owner": owner_at(right_events, right_offset),
            "predicate_relation": PREDICATE_REVERSE[LEFT_EVENTS[int(left_owner["event_index"])]["predicate"]],
            "remaining_residual_letters": len(left_residual),
            "residual_sha256": hashlib.sha256(left_residual.encode("ascii")).hexdigest(),
        })

    audit = independent_audit(candidate_text)
    if audit["normalized_letters"] != candidate_letters or not audit["two_pointer_exact"]:
        raise AssertionError("the whole candidate failed the independent exactness audit")
    if not audit["hashes_equal"]:
        raise AssertionError("forward and reverse SHA-256 checks disagree")

    return {
        "experiment_id": EXPERIMENT_ID,
        "status": "exact_center_event_path_repair_child",
        "execution": {
            "host": socket.gethostname(),
            "python_version": platform.python_version(),
            "stochasticity": "none; deterministic role-chart traversal",
        },
        "source_commit": SOURCE_COMMIT,
        "method": "a two-event open path is resegmented against the live central reverse obligation, with subject/object continuity and the stops/spots predicate relation carried by the chart",
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
            "growth_per_support": growth_each,
            "outside_supports_retained_byte_for_byte": True,
        },
        "grammar_state": {
            "left_open_event_path": list(LEFT_EVENTS),
            "right_path_resegmented_from_residual": right_events,
            "predicate_reversal_map": PREDICATE_REVERSE,
            "right_parse_count_before_path_constraint": parse_count,
            "right_parse_count_after_path_constraint": linked_parse_count,
            "left_path_connected": LEFT_EVENTS[0]["object"] == LEFT_EVENTS[1]["subject"],
            "right_path_connected": right_events[0]["object"] == right_events[1]["subject"],
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
            "id": "568-lineage-central-event-path-752",
            "rendered": candidate_text,
            "letters": audit["normalized_letters"],
            "normalized_tape_sha256": audit["sha256_forward"],
            "growth_over_parent": audit["normalized_letters"] - PARENT_LETTERS,
            "new_event_content": [surface(LEFT_EVENTS), right_text],
        },
        "independent_audit": audit,
        "provenance": {
            "event_path_authored_for_this_run": True,
            "catalogue_or_corpus_text_used": False,
            "exact_phrase_preflight": {
                "scope": "runs/, experiments/, and docs/ at source commit " + SOURCE_COMMIT,
                "exact_clause_hits": [],
                "phrases_checked": [
                    "Ari sees Leon", "Leon stops Ira", "Ari spots Noel", "Noel sees Ira",
                ],
            },
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "repair_debt": {
            "inherited_rough_prose_and_repetition_retained": True,
            "central_reversed_event_pair_is_a_proper_palindromic_span": True,
            "human_readability": "not assessed; this local grammar improvement is not a readability certification",
        },
        "next_repair": "Stop expanding local event chains. The next construction must replace a different measured clause seam with a fresh connected event pair whose prose is evaluated in context; preserve the 568, 640, 686, 736, and 752 texts separately.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.parent)
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
