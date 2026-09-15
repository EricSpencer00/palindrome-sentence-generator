"""Elliptical answer and imperative-acknowledgment residual search.

This is a distinct dialogue signature from the earlier full question/answer
inventory.  It composes paired prompts on one side and short elliptical or
imperative acknowledgments on the other.  The lexical pools are no larger
than that prior run, and all repository palindrome tapes are fingerprinted
and excluded.  Exact closures remain reader-gated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

MIN_LETTERS = 39
MAX_LETTERS = 180
ACTIONS = ("bring", "call", "check", "close", "find", "hold", "keep", "mail", "mark", "open", "read", "save", "send", "show", "take", "tell", "use", "write")
OBJECTS = ("a book", "a card", "a chart", "a file", "a key", "a letter", "a map", "a message", "a note", "a paper", "a plan", "a report", "a song", "a story", "a task")
MODALS = ("can", "could", "will", "would", "should", "may", "might")
ELLIPTICAL_PREFIX = ("yes", "sure", "okay", "alright")
ELLIPTICAL_TAILS = (("i", "can"), ("i", "will"), ("i", "did"), ("will", "do"), ("can", "do"), ("done",), ("got", "it"), ("understood",))


@dataclass(frozen=True)
class Act:
    family: str
    template: str
    words: tuple[str, ...]
    slots: tuple[tuple[str, str], ...]

    @property
    def tape(self) -> str:
        return "".join(self.words)


@dataclass(frozen=True)
class Turn:
    family: str
    acts: tuple[Act, ...]

    @property
    def words(self) -> tuple[str, ...]:
        return tuple(word for i, act in enumerate(self.acts) for word in ((";",) if i else ()) + act.words)

    @property
    def tape(self) -> str:
        return "".join(act.tape for act in self.acts)

    @property
    def slots(self):
        return tuple(slot for act in self.acts for slot in act.slots)


def _prompt_acts() -> tuple[Act, ...]:
    rows = []
    for action in ACTIONS:
        for obj in OBJECTS:
            rows.append(Act("prompt", "can_we_action_object", ("can", "we", action, *obj.split()), (("modal", "can"), ("speaker", "we"), ("action", action), ("object", obj))))
            rows.append(Act("prompt", "what_about_object", ("what", "about", *obj.split()), (("question", "what_about"), ("object", obj))))
    # Keep this inventory exactly bounded and distinct from the prior modal-you
    # and please-action templates.
    return tuple(rows)


def _response_acts() -> tuple[Act, ...]:
    rows = []
    for prefix in ELLIPTICAL_PREFIX:
        for tail in ELLIPTICAL_TAILS:
            rows.append(Act("elliptical_answer", "elliptical_prefix_tail", (prefix, *tail), (("prefix", prefix), ("elliptical_tail", " ".join(tail)))))
    for modal in MODALS:
        rows.append(Act("elliptical_answer", "speaker_modal_ellipsis", ("i", modal), (("speaker", "i"), ("modal", modal))))
    for ack in ("do it", "go ahead", "that works", "all set"):
        rows.append(Act("imperative_ack", "imperative_acknowledgment", tuple(ack.split()), (("ack", ack))))
    return tuple(rows)


def _repo_tape_fingerprint() -> tuple[frozenset[str], dict]:
    tapes: set[str] = set()
    files = 0
    for path in sorted((ROOT / "runs").rglob("*.json")):
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
    return frozenset(tapes), {"files_scanned": files, "palindrome_tapes": len(tapes), "fingerprint_sha256": hashlib.sha256("\n".join(sorted(tapes)).encode()).hexdigest()}


def _audit(left: Turn, right: Turn, collision: bool) -> dict:
    text = " ".join(left.words).capitalize() + ". " + " ".join(right.words) + "."
    tape = normalize_letters(text)
    checks = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    word_mirror = tuple(word[::-1] for word in reversed(left.words) if word != ";") == tuple(right.words)
    return {
        "rendered": text,
        "letters": len(tape),
        "normalized_letters": tape,
        "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest(),
        "independent_ascii_exact": bool(tape) and tape == tape[::-1],
        "independent_two_pointer": all(tape[i] == tape[-1 - i] for i in range(len(tape) // 2)),
        "left_turn": {"family": left.family, "slots": left.slots},
        "right_turn": {"family": right.family, "slots": right.slots},
        "word_order_shortcut": word_mirror,
        "existing_repository_tape_collision": collision,
        "central_admission": checks,
        "mechanically_admitted": all(checks.values()) and not word_mirror and not collision,
        "reader_status": "not_run; elliptical discourse structure and exactness do not certify readability",
    }


def run() -> dict:
    existing, fingerprint = _repo_tape_fingerprint()
    prompts = _prompt_acts()
    responses = _response_acts()
    lefts = tuple(Turn("paired_prompt", (first, second)) for first in prompts for second in prompts if first is not second)
    rights = tuple(Turn("paired_elliptical_ack", (first, second)) for first in responses for second in responses if first is not second)
    # Keep the bounded successor distinct from prior single-act outputs.
    right_index: dict[str, list[Turn]] = defaultdict(list)
    for right in rights:
        right_index[right.tape].append(right)
    stats = Counter({"prompt_acts": len(prompts), "response_acts": len(responses), "left_turns": len(lefts), "right_turns": len(rights), "indexed_right_tapes": len(right_index)})
    candidates = []
    frontier = []
    seen = set()
    for left in lefts:
        stats["left_states_considered"] += 1
        matches = right_index.get(left.tape[::-1], ())
        if not matches:
            stats["residual_misses"] += 1
            if len(frontier) < 100:
                frontier.append({"left_tape": left.tape, "left_templates": [act.template for act in left.acts], "reason": "no elliptical acknowledgment residual"})
            continue
        stats["residual_matches"] += len(matches)
        for right in matches:
            tape = left.tape + right.tape
            if len(tape) < MIN_LETTERS:
                stats["short_rejections"] += 1
                continue
            collision = tape in existing
            if collision:
                stats["repository_collision_rejections"] += 1
                continue
            key = tape
            if key in seen:
                continue
            seen.add(key)
            row = _audit(left, right, collision)
            candidates.append(row)
            stats["candidates"] += 1
            if row["mechanically_admitted"]:
                stats["mechanically_admitted"] += 1
    candidates.sort(key=lambda row: (row["mechanically_admitted"], row["letters"]), reverse=True)
    return {
        "status": "complete_dialogue_elliptical_ack_residual_inventory_no_reader_promotion",
        "config": {
            "semantic_family": "paired prompts -> elliptical answers and imperative acknowledgments",
            "inventory_signature": "can-we/what-about prompt acts plus elliptical-tail/imperative-ack acts",
            "prior_single_act_reuse": False,
            "pool_size_change": "not larger than preceding dialogue inventory",
            "repository_tape_exclusion": True,
            "minimum_letters": MIN_LETTERS,
            "maximum_letters": MAX_LETTERS,
            "independent_residual_matching": True,
            "anti_shortcut_gate": True,
        },
        "stats": dict(stats),
        "repository_fingerprint": fingerprint,
        "admitted": [row for row in candidates if row["mechanically_admitted"]],
        "near_misses": candidates[:100],
        "residual_frontier": frontier,
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "source_text_copied": False, "inventory_version": "dialogue_elliptical_ack_v1"},
        "next_operator": "Add a bounded discourse-topic slot shared across prompt and elliptical response while preserving repository-wide tape exclusion and exact residual audit.",
        "reader_gate": "No output is human evidence; future closures require intact-prose and shuffled-control readers.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists(): parser.error(f"refusing to overwrite output: {args.out}")
    result = run(); args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stats": result["stats"], "admitted": len(result["admitted"])}, indent=2))


if __name__ == "__main__": main()
