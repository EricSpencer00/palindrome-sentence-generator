"""Dialogue/question-to-answer and instruction-to-acknowledgment search.

This is a materially separate semantic inventory: all lexical entries and
templates are hand-authored dialogue acts, not Brown/POS/event frames or
semordnilap pairs.  Question and answer clauses are independently generated,
then joined only when the answer tape is the exact reverse residual.  Existing
repository palindrome tapes are fingerprinted and excluded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

MIN_LETTERS = 39
MAX_LETTERS = 180

# Semantic inventories are intentionally hand-authored and scoped to dialogue
# acts.  There is no corpus/POS/event extraction in this branch.
MODALS = ("can", "could", "will", "would", "should", "may", "might")
ACTION_VERBS = ("bring", "call", "check", "close", "find", "hold", "keep", "mail", "mark", "open", "read", "save", "send", "show", "take", "tell", "use", "write")
PAST_ACTIONS = ("brought", "called", "checked", "closed", "found", "held", "kept", "mailed", "marked", "opened", "read", "saved", "sent", "showed", "took", "told", "used", "wrote")
OBJECTS = ("a book", "a card", "a chart", "a file", "a key", "a letter", "a map", "a message", "a note", "a paper", "a plan", "a report", "a song", "a story", "a task", "the book", "the card", "the chart", "the file", "the key", "the letter", "the map", "the message", "the note", "the paper", "the plan", "the report", "the song", "the story", "the task")
ACKS = ("yes", "okay", "alright", "agreed", "done", "sure", "thanks")


@dataclass(frozen=True)
class Act:
    family: str
    template: str
    words: tuple[str, ...]
    slots: tuple[tuple[str, str], ...]

    @property
    def tape(self) -> str:
        return "".join(self.words)


def _objects() -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(obj.split()) for obj in OBJECTS)


def _questions_and_instructions() -> tuple[Act, ...]:
    rows = []
    for modal in MODALS:
        for verb in ACTION_VERBS:
            for obj in _objects():
                rows.append(Act("question", "modal_you_action_object", (modal, "you", verb, *obj), (("modal", modal), ("listener", "you"), ("action", verb), ("object", " ".join(obj)))))
    for verb in ACTION_VERBS:
        for obj in _objects():
            rows.append(Act("instruction", "please_action_object", ("please", verb, *obj), (("politeness", "please"), ("action", verb), ("object", " ".join(obj)))))
    return tuple(rows)


def _answers_and_acknowledgments() -> tuple[Act, ...]:
    rows = []
    for modal in MODALS:
        for verb in ACTION_VERBS:
            for obj in _objects():
                rows.append(Act("answer", "speaker_modal_action_object", ("i", modal, verb, *obj), (("speaker", "i"), ("modal", modal), ("action", verb), ("object", " ".join(obj)))))
    for ack in ACKS:
        for past in PAST_ACTIONS:
            for obj in _objects():
                rows.append(Act("ack", "ack_past_action_object", (ack, "i", past, *obj), (("ack", ack), ("speaker", "i"), ("action", past), ("object", " ".join(obj)))))
    return tuple(rows)


def _repo_tape_fingerprint(output: Path | None = None) -> tuple[frozenset[str], dict]:
    tapes: set[str] = set()
    files = 0
    output = output.resolve() if output else None
    for path in sorted((ROOT / "runs").rglob("*.json")):
        if output and path.resolve() == output:
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        files += 1
        def walk(value):
            if isinstance(value, str):
                try:
                    tape = normalize_letters(value)
                except ValueError:
                    return
                if len(tape) >= MIN_LETTERS and tape == tape[::-1]:
                    tapes.add(tape)
            elif isinstance(value, dict):
                for child in value.values():
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)
        walk(payload)
    digest = hashlib.sha256("\n".join(sorted(tapes)).encode()).hexdigest()
    return frozenset(tapes), {"files_scanned": files, "palindrome_tapes": len(tapes), "fingerprint_sha256": digest}


def _word_mirror(left: Act, right: Act) -> bool:
    return tuple(word[::-1] for word in reversed(left.words)) == right.words


def _audit(left: Act, right: Act, existing_collision: bool) -> dict:
    text = " ".join(left.words).capitalize() + "; " + " ".join(right.words) + "."
    tape = normalize_letters(text)
    checks = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    return {
        "rendered": text,
        "letters": len(tape),
        "normalized_letters": tape,
        "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest(),
        "independent_ascii_exact": bool(tape) and tape == tape[::-1],
        "independent_two_pointer": all(tape[i] == tape[-1 - i] for i in range(len(tape) // 2)),
        "left_act": {"family": left.family, "template": left.template, "slots": left.slots},
        "right_act": {"family": right.family, "template": right.template, "slots": right.slots},
        "word_order_shortcut": _word_mirror(left, right),
        "existing_repository_tape_collision": existing_collision,
        "central_admission": checks,
        "mechanically_admitted": all(checks.values()) and not _word_mirror(left, right) and not existing_collision,
        "reader_status": "not_run; dialogue-act structure and exactness do not certify readability",
    }


def run(output: Path | None = None) -> dict:
    existing_tapes, fingerprint = _repo_tape_fingerprint(output)
    lefts = _questions_and_instructions()
    rights = _answers_and_acknowledgments()
    right_index: dict[str, list[Act]] = defaultdict(list)
    for right in rights:
        right_index[right.tape].append(right)
    stats = Counter({"left_acts": len(lefts), "right_acts": len(rights), "indexed_right_tapes": len(right_index)})
    candidates = []
    residual_frontier = []
    seen = set()
    for left in lefts:
        stats["left_states_considered"] += 1
        matches = right_index.get(left.tape[::-1], ())
        if not matches:
            stats["residual_misses"] += 1
            if len(residual_frontier) < 100:
                probe = " ".join(left.words).capitalize() + ";"
                probe_tape = normalize_letters(probe)
                residual_frontier.append({
                    "left_tape": left.tape,
                    "left_template": left.template,
                    "left_letters": len(left.tape),
                    "reverse_target": left.tape[::-1],
                    "rendered_probe": probe,
                    "probe_exact_audit": {
                        "exact": bool(probe_tape) and probe_tape == probe_tape[::-1],
                        "letters": len(probe_tape),
                        "normalized_sha256": hashlib.sha256(probe_tape.encode()).hexdigest(),
                    },
                    "reason": "no independently lexicalized answer residual",
                })
            continue
        stats["residual_matches"] += len(matches)
        for right in matches:
            tape = left.tape + right.tape
            if len(tape) < MIN_LETTERS:
                stats["short_rejections"] += 1
                continue
            collision = tape in existing_tapes
            if collision:
                stats["repository_collision_rejections"] += 1
                continue
            if _word_mirror(left, right):
                stats["word_order_rejections"] += 1
                continue
            if tape in seen:
                continue
            seen.add(tape)
            row = _audit(left, right, collision)
            candidates.append(row)
            stats["candidates"] += 1
            if row["mechanically_admitted"]:
                stats["mechanically_admitted"] += 1
    candidates.sort(key=lambda row: (row["mechanically_admitted"], row["letters"]), reverse=True)
    return {
        "status": "complete_dialogue_acknowledgment_residual_inventory_no_reader_promotion",
        "family_id": "dialogue-acknowledgment-residual-inventory",
        "state_space_signature": "dialogue-act-question-answer-instruction-acknowledgment|hand-authored-cross-product|independent-residual-tape-index",
        "config": {
            "semantic_family": "question->answer and instruction->acknowledgment",
            "lexical_source": "hand-authored dialogue-act inventory only",
            "brown_pos_event_semordnilap_sources": False,
            "independent_residual_matching": True,
            "minimum_letters": MIN_LETTERS,
            "maximum_letters": MAX_LETTERS,
            "repository_tape_exclusion": True,
            "anti_shortcut_gate": True,
        },
        "stats": dict(stats),
        "repository_fingerprint": fingerprint,
        "admitted": [row for row in candidates if row["mechanically_admitted"]],
        "near_misses": candidates[:100],
        "residual_frontier": residual_frontier,
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_text_copied": False,
            "lexical_inventory_version": "dialogue_ack_v1_hand_authored",
        },
        "next_operator": (
            "Add elliptical answer acts and imperative acknowledgments with independently authored discourse "
            "slots, while retaining the repository-tape fingerprint and exact residual audit."
        ),
        "reader_gate": "No output is human evidence; future closures require intact-prose and shuffled-control readers.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite output: {args.out}")
    result = run(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stats": result["stats"], "admitted": len(result["admitted"])}, indent=2))


if __name__ == "__main__":
    main()
