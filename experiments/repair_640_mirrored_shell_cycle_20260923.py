"""Replace one inherited 640-letter shell with a fresh connected event cycle.

The selected old spans are the exact mirrored shells retained from the 568
parent.  A variable-length, role-constrained grammar chart resegments the
reverse of one new four-event chain into a second connected chain.  The paired
supports are replaced online against their live character residual; the rest
of the 640 parent is retained verbatim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import socket
from pathlib import Path


EXPERIMENT_ID = "repair-640-mirrored-shell-cycle-20260923"
SOURCE_COMMIT = "3ed6aab05d244983c4da1d43bc41af7a11f88b99"
PARENT_ID = "568-live-linked-observation-chain-640"
PARENT_SHA256 = "c3013fe4411ae2be7ab3211a41e59606faab7385dc4fe3dc8185d47f8cbedac9"
PARENT_LETTERS = 640
LEFT_SUPPORT = (100, 127)
RIGHT_SUPPORT = (513, 540)
OLD_LEFT_TAPE = "marastopsratsatubhemapsaron"
OLD_RIGHT_TAPE = "noraspamehbutastarspotsaram"
PARENT_568_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"

ENTITY_FORMS = ("Leon", "Noel", "Nora", "Aron", "Mara", "Aram", "Sara", "Aras", "Aidan", "Nadia", "Ira", "Ari")
PREDICATES = ("sees",)
LEFT_EVENTS = (
    {"id": "cycle-1", "subject": "Leon", "predicate": "sees", "object": "Sara"},
    {"id": "cycle-2", "subject": "Sara", "predicate": "sees", "object": "Aidan"},
    {"id": "cycle-3", "subject": "Aidan", "predicate": "sees", "object": "Nora"},
    {"id": "cycle-4", "subject": "Nora", "predicate": "sees", "object": "Leon"},
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


def raw_index_at_letter(text: str, target: int) -> int:
    """Raw offset of the first ASCII letter after ``target`` letters."""
    seen = 0
    for index, char in enumerate(text):
        if char.isascii() and char.isalpha():
            if seen == target:
                return index
            seen += 1
    if seen == target:
        return len(text)
    raise ValueError(f"parent has fewer than {target} letters")


def segment_connected_chain(tape: str, event_count: int) -> tuple[list[dict[str, str]], int, int]:
    """Parse a complete character obligation as NAME-sees-NAME clauses."""
    entities = tuple((name, normalize(name)) for name in ENTITY_FORMS)
    verb_tapes = tuple((verb, normalize(verb)) for verb in PREDICATES)

    def walk(offset: int, remaining: int) -> list[list[dict[str, str]]]:
        if remaining == 0:
            return [[]] if offset == len(tape) else []
        parses: list[list[dict[str, str]]] = []
        for subject, subject_tape in entities:
            if not tape.startswith(subject_tape, offset):
                continue
            after_subject = offset + len(subject_tape)
            for verb, verb_tape in verb_tapes:
                if not tape.startswith(verb_tape, after_subject):
                    continue
                after_verb = after_subject + len(verb_tape)
                for obj, object_tape in entities:
                    if not tape.startswith(object_tape, after_verb):
                        continue
                    for suffix in walk(after_verb + len(object_tape), remaining - 1):
                        parses.append([
                            {"subject": subject, "predicate": verb, "object": obj},
                            *suffix,
                        ])
        return parses

    parses = walk(0, event_count)
    chains = [
        row for row in parses
        if all(row[i]["object"] == row[i + 1]["subject"]
               for i in range(len(row) - 1))
        and row[-1]["object"] == row[0]["subject"]
    ]
    if not chains:
        raise ValueError("the opposing tape has no complete connected event cycle")
    selected = min(chains, key=lambda row: tuple(
        (event["subject"], event["predicate"], event["object"]) for event in row
    ))
    return selected, len(parses), len(chains)


def surface(events: list[dict[str, str]] | tuple[dict[str, str], ...]) -> str:
    return " ".join(
        f"{event['subject']} {event['predicate']} {event['object']}."
        for event in events
    )


def tape_for(events: list[dict[str, str]] | tuple[dict[str, str], ...]) -> str:
    return "".join(
        normalize(event["subject"] + event["predicate"] + event["object"])
        for event in events
    )


def owner_at(events: list[dict[str, str]] | tuple[dict[str, str], ...], offset: int) -> dict[str, object]:
    cursor = 0
    for index, event in enumerate(events):
        for role in ("subject", "predicate", "object"):
            width = len(normalize(event[role]))
            if cursor <= offset < cursor + width:
                return {
                    "event_index": index,
                    "event_id": event["id"] if "id" in event else f"cycle-{index + 1}",
                    "role": role,
                    "word": event[role],
                }
            cursor += width
    raise IndexError(f"no grammar owner for character offset {offset}")


def render(parent_path: Path) -> dict[str, object]:
    payload = json.loads(parent_path.read_text())
    parent = payload["candidate"]
    if parent["id"] != PARENT_ID or parent["letters"] != PARENT_LETTERS:
        raise ValueError("the requested 640-letter child is not the loaded parent")
    parent_text = str(parent["rendered"])
    parent_tape = normalize(parent_text)
    parent_sha = hashlib.sha256(parent_tape.encode("ascii")).hexdigest()
    if parent_sha != PARENT_SHA256 or parent_tape != parent_tape[::-1]:
        raise ValueError("the loaded 640 parent fails its pinned independent audit")
    if payload["parent"]["normalized_tape_sha256"] != PARENT_568_SHA256:
        raise ValueError("the 640 parent is not on the pinned 568 lineage")

    left_start, left_end = LEFT_SUPPORT
    right_start, right_end = RIGHT_SUPPORT
    if (right_start, right_end) != (PARENT_LETTERS - left_end, PARENT_LETTERS - left_start):
        raise ValueError("the repair supports are not mirrored positions in the 640 parent")
    if parent_tape[left_start:left_end] != OLD_LEFT_TAPE:
        raise ValueError("the named repeated left shell moved or changed")
    if parent_tape[right_start:right_end] != OLD_RIGHT_TAPE:
        raise ValueError("the mirrored right shell moved or changed")
    if OLD_LEFT_TAPE != OLD_RIGHT_TAPE[::-1]:
        raise AssertionError("the loaded shell support is not an exact residual pair")

    left_tape = tape_for(LEFT_EVENTS)
    right_obligation = left_tape[::-1]
    right_events, parse_count, connected_parse_count = segment_connected_chain(
        right_obligation, len(LEFT_EVENTS)
    )
    right_text = surface(right_events)
    right_tape = normalize(right_text)
    if right_tape != right_obligation:
        raise AssertionError("typed event chart failed to consume the reverse obligation")
    growth_each_support = len(left_tape) - len(OLD_LEFT_TAPE)

    left_raw_start = raw_index_at_letter(parent_text, left_start)
    left_raw_end = raw_index_at_letter(parent_text, left_end)
    right_raw_start = raw_index_at_letter(parent_text, right_start)
    right_raw_end = raw_index_at_letter(parent_text, right_end)
    if normalize(parent_text[left_raw_start:left_raw_end]) != OLD_LEFT_TAPE:
        raise AssertionError("raw left support and normalized cursor support differ")
    if normalize(parent_text[right_raw_start:right_raw_end]) != OLD_RIGHT_TAPE:
        raise AssertionError("raw right support and normalized cursor support differ")

    # Work from the later raw support to the earlier one so the recorded cuts
    # remain those of the loaded 640 parent.
    after_right = (
        parent_text[:right_raw_start] + right_text + " " + parent_text[right_raw_end:]
    )
    candidate_text = (
        after_right[:left_raw_start] + surface(LEFT_EVENTS) + " "
        + after_right[left_raw_end:]
    )

    trace: list[dict[str, object]] = []
    child_right_start = right_start + growth_each_support
    child_right_reverse_start = child_right_start + len(right_tape) - 1
    for offset, left_char in enumerate(left_tape):
        right_offset = len(right_tape) - 1 - offset
        right_char = right_tape[right_offset]
        if left_char != right_char:
            raise ValueError(
                f"residual contradiction at child cursors "
                f"{left_start + offset}/{child_right_reverse_start - offset}: "
                f"{left_char!r} != {right_char!r}"
            )
        left_residual = left_tape[offset + 1:]
        right_residual = right_tape[:right_offset][::-1]
        if left_residual != right_residual:
            raise AssertionError("live residual strings diverged after a matched character")
        trace.append({
            "step": offset + 1,
            "left_cursor_child": left_start + offset,
            "right_reverse_cursor_child": child_right_reverse_start - offset,
            "character": left_char,
            "left_owner": owner_at(LEFT_EVENTS, offset),
            "right_owner": owner_at(right_events, right_offset),
            "remaining_residual_letters": len(left_residual),
            "residual_sha256": hashlib.sha256(left_residual.encode("ascii")).hexdigest(),
        })

    audit = independent_audit(candidate_text)
    expected_length = PARENT_LETTERS + 2 * growth_each_support
    if audit["normalized_letters"] != expected_length or not audit["two_pointer_exact"]:
        raise AssertionError("the whole repaired candidate failed independent exactness")
    if not audit["hashes_equal"]:
        raise AssertionError("the candidate's forward/reverse tape hashes differ")

    return {
        "experiment_id": EXPERIMENT_ID,
        "status": "exact_shell_repair_child",
        "execution": {
            "host": socket.gethostname(),
            "python_version": platform.python_version(),
            "stochasticity": "none; deterministic role-chart traversal",
        },
        "source_commit": SOURCE_COMMIT,
        "method": "variable-length connected event-cycle replacement at one measured mirrored shell; the opposite event chain is segmented from the reversed character obligation while role owners and cursors advance",
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
            "old_shell_relation": "the mirrored question/fragment shell was removed as a paired support",
            "new_left_clause_count": len(LEFT_EVENTS),
            "new_right_clause_count": len(right_events),
            "new_letters_per_support": [len(left_tape), len(right_tape)],
            "growth_per_support": growth_each_support,
            "outside_supports_retained_byte_for_byte": True,
        },
        "grammar_state": {
            "left_connected_event_cycle": list(LEFT_EVENTS),
            "right_cycle_resegmented_from_residual": right_events,
            "right_parse_count_before_cycle_constraint": parse_count,
            "right_parse_count_after_cycle_constraint": connected_parse_count,
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
            "id": "568-lineage-shell-cycle-repair-686",
            "rendered": candidate_text,
            "letters": audit["normalized_letters"],
            "normalized_tape_sha256": audit["sha256_forward"],
            "growth_over_parent": audit["normalized_letters"] - PARENT_LETTERS,
            "new_event_content": [surface(LEFT_EVENTS), right_text],
        },
        "independent_audit": audit,
        "provenance": {
            "candidate_phrases_authored_for_this_run": True,
            "external_or_catalogue_text_used": False,
            "exact_phrase_preflight": {
                "scope": "runs/, experiments/, and docs/ at source commit " + SOURCE_COMMIT,
                "exact_clause_hits": [],
                "phrases_checked": [
                    "Leon sees Sara", "Sara sees Aidan", "Aidan sees Nora", "Nora sees Leon",
                    "Noel sees Aron", "Aron sees Nadia", "Nadia sees Aras", "Aras sees Noel",
                ],
            },
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "repair_debt": {
            "inherited_repetition_outside_repaired_shell": True,
            "new_chain_repeats_predicate": True,
            "human_readability": "not assessed; exactness and connected grammar do not certify readability",
        },
        "next_repair": "Preserve this 686-letter child and reopen the next repeated clause support only after checking its exact child coordinates against the live tape; keep the 568 and 640 artifacts as rollback points.",
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
