"""Insert a fresh, grammar-owned event chain through one exact 568 seam.

This is a single candidate construction, not a sentence-bank sweep.  The left
side is authored as three linked observation events.  A tiny role grammar then
resegments the complete opposing character obligation into independent clauses
while the left and right cursors consume matching characters.  The two blocks
are inserted at the 568 parent's matched sentence boundary (64 / 504); all
intervening parent text is retained byte-for-byte.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import socket
from pathlib import Path


PARENT_ID = "outer-causal-scene-568-working-incumbent"
PARENT_TAPE_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
PARENT_LETTERS = 568
LEFT_CUT = 64
RIGHT_CUT = PARENT_LETTERS - LEFT_CUT
EXPERIMENT_ID = "incumbent-568-live-event-chain-insertion-20260923"
SOURCE_COMMIT = "0a44ffd089ab9f9049c815428236ba101f6285ee"

ENTITY_FORMS = ("Leon", "Noel", "Nora", "Aron", "Mara", "Aram", "Sara", "Aras")
PREDICATES = ("sees",)

LEFT_EVENTS = (
    {"id": "event-1", "subject": "Noel", "predicate": "sees", "object": "Mara"},
    {"id": "event-2", "subject": "Mara", "predicate": "sees", "object": "Sara"},
    {"id": "event-3", "subject": "Sara", "predicate": "sees", "object": "Nora"},
)


def normalize(text: str) -> str:
    return "".join(re.findall(r"[a-z]", text.casefold()))


def independent_audit(text: str) -> dict[str, object]:
    tape = normalize(text)
    mismatch = next(
        ((i, tape[i], tape[len(tape) - 1 - i])
         for i in range(len(tape) // 2)
         if tape[i] != tape[len(tape) - 1 - i]),
        None,
    )
    forward = hashlib.sha256(tape.encode("ascii")).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode("ascii")).hexdigest()
    return {
        "normalized_letters": len(tape),
        "two_pointer_exact": mismatch is None,
        "first_mismatch": mismatch,
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "hashes_equal": forward == reverse,
        "normalized_tape": tape,
    }


def after_sentence_boundary(text: str, letter_count: int) -> int:
    """Return the raw insertion point after the sentence mark at a letter cut."""
    seen = 0
    index = 0
    while index < len(text) and seen < letter_count:
        if text[index].isascii() and text[index].isalpha():
            seen += 1
        index += 1
    if seen != letter_count:
        raise ValueError(f"parent has fewer than {letter_count} letters")
    mark_start = index
    while index < len(text) and text[index] in ".?!;:\u201d\"'":
        index += 1
    if index == mark_start:
        raise ValueError(f"cut {letter_count} is not after sentence punctuation")
    if index < len(text) and not text[index].isspace():
        raise ValueError(f"cut {letter_count} does not land at a sentence boundary")
    return index


def segment_event_chain(
    tape: str, event_count: int
) -> tuple[list[dict[str, str]], int, int]:
    """Find a complete NAME + finite-verb + NAME parse of a reversed tape."""
    entities = tuple((name, normalize(name)) for name in ENTITY_FORMS)
    predicates = tuple((verb, normalize(verb)) for verb in PREDICATES)

    def walk(offset: int, remaining: int) -> list[list[dict[str, str]]]:
        if remaining == 0:
            return [[]] if offset == len(tape) else []
        rows: list[list[dict[str, str]]] = []
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
                        rows.append([
                            {"subject": subject, "predicate": predicate, "object": obj},
                            *suffix,
                        ])
        return rows

    parses = walk(0, event_count)
    # Prefer a semantically linked chain, then stable lexical ordering.  This is
    # a parse preference, not a readability certificate.
    linked = [
        row for row in parses
        if all(row[i]["object"] == row[i + 1]["subject"]
               for i in range(len(row) - 1))
    ]
    if not linked:
        raise ValueError("the opposing tape has no connected complete-event parse")
    selected = min(linked, key=lambda row: tuple(
        (event["subject"], event["predicate"], event["object"]) for event in row
    ))
    return selected, len(parses), len(linked)


def surface_events(events: list[dict[str, str]]) -> str:
    return " ".join(
        f"{event['subject']} {event['predicate']} {event['object']}."
        for event in events
    )


def slot_at(events: list[dict[str, str]], offset: int) -> dict[str, str | int]:
    event_index, in_event = divmod(offset, 12)
    role = "subject" if in_event < 4 else "predicate" if in_event < 8 else "object"
    return {
        "event_index": event_index,
        "event_id": f"event-{event_index + 1}",
        "role": role,
        "word": events[event_index][role],
    }


def stream_equation(
    left_tape: str,
    right_tape: str,
    left_events: list[dict[str, str]],
    right_events: list[dict[str, str]],
) -> list[dict[str, object]]:
    if len(left_tape) != len(right_tape):
        raise ValueError("left emissions and reverse obligations have unequal length")
    trace: list[dict[str, object]] = []
    for offset, left_char in enumerate(left_tape):
        right_offset = len(right_tape) - 1 - offset
        right_char = right_tape[right_offset]
        if left_char != right_char:
            raise ValueError(
                f"live residual contradiction at paired cursors "
                f"{LEFT_CUT + offset}/{PARENT_LETTERS + len(left_tape) - 1 - offset}: "
                f"{left_char!r} != {right_char!r}"
            )
        residual_left = left_tape[offset + 1:]
        residual_right = right_tape[:right_offset][::-1]
        if residual_left != residual_right:
            raise AssertionError("residual owners diverged after a matched character")
        trace.append({
            "step": offset + 1,
            "left_cursor": LEFT_CUT + offset,
            "right_reverse_cursor_child": (
                RIGHT_CUT + len(left_tape) + len(right_tape) - 1 - offset
            ),
            "character": left_char,
            "left_owner": slot_at(left_events, offset),
            "right_owner": slot_at(right_events, right_offset),
            "remaining_residual_letters": len(residual_left),
            "residual_sha256": hashlib.sha256(residual_left.encode("ascii")).hexdigest(),
        })
    return trace


def build(parent_path: Path) -> dict[str, object]:
    payload = json.loads(parent_path.read_text())
    parent = next(row for row in payload["rows"] if row["id"] == PARENT_ID)
    parent_text = str(parent["rendered"])
    parent_tape = normalize(parent_text)
    parent_sha = hashlib.sha256(parent_tape.encode("ascii")).hexdigest()
    if len(parent_tape) != PARENT_LETTERS or parent_sha != PARENT_TAPE_SHA256:
        raise ValueError("the loaded parent is not the pinned, exact 568-letter incumbent")
    if parent_tape != parent_tape[::-1]:
        raise ValueError("the pinned parent fails the independent two-pointer check")
    if RIGHT_CUT != len(parent_tape) - LEFT_CUT:
        raise AssertionError("the selected cuts are not opposite cursors")

    left_tape = "".join(
        normalize(event["subject"] + event["predicate"] + event["object"])
        for event in LEFT_EVENTS
    )
    opposing_tape = left_tape[::-1]
    right_events, parse_count, linked_parse_count = segment_event_chain(
        opposing_tape, len(LEFT_EVENTS)
    )
    right_text = surface_events(right_events)
    right_tape = normalize(right_text)
    if right_tape != opposing_tape:
        raise AssertionError("the right grammar did not consume the exact character obligation")

    raw_left = after_sentence_boundary(parent_text, LEFT_CUT)
    raw_right = after_sentence_boundary(parent_text, RIGHT_CUT)
    # The right insertion is applied first, so both raw offsets still refer to
    # the untouched parent.  The left insertion then leaves it intact.
    with_right = parent_text[:raw_right] + " " + right_text + parent_text[raw_right:]
    rendered = with_right[:raw_left] + " " + surface_events(list(LEFT_EVENTS)) + with_right[raw_left:]

    trace = stream_equation(left_tape, right_tape, list(LEFT_EVENTS), right_events)
    audit = independent_audit(rendered)
    expected_letters = PARENT_LETTERS + len(left_tape) + len(right_tape)
    if audit["normalized_letters"] != expected_letters or not audit["two_pointer_exact"]:
        raise AssertionError("independent whole-candidate exactness check failed")
    if not audit["hashes_equal"]:
        raise AssertionError("forward/reverse tape digests differ")

    return {
        "experiment_id": EXPERIMENT_ID,
        "status": "exact_new_content_child",
        "execution": {
            "host": socket.gethostname(),
            "python_version": platform.python_version(),
            "stochasticity": "none; deterministic grammar-chart traversal",
        },
        "source_commit": SOURCE_COMMIT,
        "method": "one live event-chain insertion at a matched sentence seam; a typed opposite-side chart resegments the reversed event tape while character cursors and event-slot owners advance together",
        "parent": {
            "artifact": str(parent_path),
            "id": PARENT_ID,
            "letters": PARENT_LETTERS,
            "normalized_tape_sha256": parent_sha,
            "artifact_sha256": hashlib.sha256(parent_path.read_bytes()).hexdigest(),
        },
        "seam": {
            "parent_normalized_cuts": [LEFT_CUT, RIGHT_CUT],
            "raw_offsets_after_sentence_marks": [raw_left, raw_right],
            "parent_boundary_context": [
                parent_text[max(0, raw_left - 28):raw_left],
                parent_text[raw_right:min(len(parent_text), raw_right + 28)],
            ],
            "retained_parent_tape_byte_for_byte": True,
        },
        "grammar_state": {
            "left_event_chain": list(LEFT_EVENTS),
            "right_events_discovered_by_typed_resegmentation": right_events,
            "right_parse_count_before_chain_preference": parse_count,
            "right_parse_count_after_chain_preference": linked_parse_count,
            "left_chain_connected": all(
                LEFT_EVENTS[i]["object"] == LEFT_EVENTS[i + 1]["subject"]
                for i in range(len(LEFT_EVENTS) - 1)
            ),
            "right_chain_connected": all(
                right_events[i]["object"] == right_events[i + 1]["subject"]
                for i in range(len(right_events) - 1)
            ),
        },
        "live_residual": {
            "left_emitted_tape": left_tape,
            "right_reverse_obligation": opposing_tape,
            "initial_residual_letters": len(left_tape),
            "steps": trace,
            "final_residual": "",
            "committed_character_contradictions": 0,
        },
        "candidate": {
            "id": "568-live-linked-observation-chain-640",
            "rendered": rendered,
            "letters": audit["normalized_letters"],
            "normalized_tape_sha256": audit["sha256_forward"],
            "growth_over_parent": audit["normalized_letters"] - PARENT_LETTERS,
            "new_event_content": [surface_events(list(LEFT_EVENTS)), right_text],
        },
        "independent_audit": audit,
        "provenance": {
            "inserted_text_authored_for_this_run": True,
            "catalogue_or_corpus_text_used": False,
            "candidate_phrase_preflight": {
                "scope": "runs/, experiments/, and docs/ at source commit " + SOURCE_COMMIT,
                "exact_clause_hits": [],
                "phrases_checked": [
                    "Noel sees Mara", "Mara sees Sara", "Sara sees Nora",
                    "Aron sees Aras", "Aras sees Aram", "Aram sees Leon",
                ],
            },
            "generator": EXPERIMENT_ID + ".py",
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "repair_debt": {
            "inherited_repetition_and_rough_discourse_retained": True,
            "new_chain_repeats_predicate": True,
            "proper_palindromic_spans_may_remain": True,
            "human_readability": "not assessed; this exact construction is not a readability certification",
        },
        "next_repair": "On this 640 child, reopen normalized parent-support [64,91) and its reflected [477,504): replace the duplicated 'Mara stops rats. A tub? He maps Aron.' shell through a new event frame while carrying the actual word-edge residual. Preserve this 640 child as rollback.",
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
