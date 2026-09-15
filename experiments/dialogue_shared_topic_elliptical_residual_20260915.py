"""Shared-topic successor for the elliptical dialogue inventory."""
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
TOPICS = ("book", "card", "chart", "file", "key", "letter", "map", "message", "note", "paper", "plan", "report", "song", "story", "task")
MODALS = ("can", "could", "will", "would", "should", "may", "might")
ACK_PREFIX = ("yes", "sure", "okay", "alright")
ACK_TAIL = (("i", "can"), ("i", "will"), ("i", "did"), ("will", "do"), ("can", "do"), ("done",), ("got", "it"), ("understood",))


@dataclass(frozen=True)
class Act:
    family: str
    words: tuple[str, ...]
    topic: str
    slots: tuple[tuple[str, str], ...]

    @property
    def tape(self):
        return "".join(self.words)


@dataclass(frozen=True)
class Turn:
    family: str
    topic: str
    acts: tuple[Act, ...]

    @property
    def words(self):
        out = []
        for index, act in enumerate(self.acts):
            if index:
                out.append(";")
            out.extend(act.words)
        return tuple(out)

    @property
    def tape(self):
        return "".join(act.tape for act in self.acts)

    @property
    def slots(self):
        return tuple(slot for act in self.acts for slot in act.slots)


def _prompt_acts():
    rows = []
    for topic in TOPICS:
        for action in ACTIONS:
            rows.append(Act("topic_prompt", ("can", "we", action, "the", topic), topic, (("topic", topic), ("action", action), ("speaker", "we"))))
        rows.append(Act("topic_prompt", ("what", "about", "the", topic), topic, (("topic", topic), ("question", "what_about"))))
    return tuple(rows)


def _response_acts():
    rows = []
    for topic in TOPICS:
        for prefix in ACK_PREFIX:
            for tail in ACK_TAIL:
                rows.append(Act("topic_elliptical", (prefix, *tail, "about", topic), topic, (("topic", topic), ("prefix", prefix), ("tail", " ".join(tail)))))
        for modal in MODALS:
            rows.append(Act("topic_elliptical", ("i", modal, "about", topic), topic, (("topic", topic), ("speaker", "i"), ("modal", modal))))
        rows.append(Act("topic_ack", ("done", "with", topic), topic, (("topic", topic), ("ack", "done_with"))))
    return tuple(rows)


def _repo_fingerprint():
    tapes = set(); files = 0
    for path in sorted((ROOT / "runs").rglob("*.json")):
        try: payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError): continue
        files += 1
        def walk(value):
            if isinstance(value, str):
                try: tape = normalize_letters(value)
                except ValueError: return
                if len(tape) >= MIN_LETTERS and tape == tape[::-1]: tapes.add(tape)
            elif isinstance(value, dict):
                for child in value.values(): walk(child)
            elif isinstance(value, list):
                for child in value: walk(child)
        walk(payload)
    return frozenset(tapes), {"files_scanned": files, "palindrome_tapes": len(tapes), "fingerprint_sha256": hashlib.sha256("\n".join(sorted(tapes)).encode()).hexdigest()}


def _audit(left: Turn, right: Turn, collision: bool):
    text = " ".join(left.words).capitalize() + ". " + " ".join(right.words) + "."
    tape = normalize_letters(text)
    checks = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    mirror = tuple(word[::-1] for word in reversed(left.words) if word != ";") == right.words
    return {"rendered": text, "letters": len(tape), "normalized_letters": tape, "normalized_sha256": hashlib.sha256(tape.encode()).hexdigest(), "independent_ascii_exact": bool(tape) and tape == tape[::-1], "independent_two_pointer": all(tape[i] == tape[-1-i] for i in range(len(tape)//2)), "left_state": {"family": left.family, "topic": left.topic, "slots": left.slots}, "right_state": {"family": right.family, "topic": right.topic, "slots": right.slots}, "shared_topic_checked": left.topic == right.topic, "word_order_shortcut": mirror, "existing_repository_tape_collision": collision, "central_admission": checks, "mechanically_admitted": all(checks.values()) and not mirror and not collision, "reader_status": "not_run; topic coherence and exactness do not certify readability"}


def run():
    existing, fingerprint = _repo_fingerprint()
    prompts, responses = _prompt_acts(), _response_acts()
    lefts = tuple(Turn("paired_topic_prompt", topic, (a, b)) for topic in TOPICS for a in prompts if a.topic == topic for b in prompts if b.topic == topic and a is not b)
    rights = tuple(Turn("paired_topic_elliptical_ack", topic, (a, b)) for topic in TOPICS for a in responses if a.topic == topic for b in responses if b.topic == topic and a is not b)
    index = defaultdict(list)
    for right in rights: index[right.tape].append(right)
    stats = Counter({"prompt_acts": len(prompts), "response_acts": len(responses), "left_turns": len(lefts), "right_turns": len(rights), "indexed_right_tapes": len(index)})
    frontier = []; candidates = []; seen = set()
    for left in lefts:
        stats["left_states_considered"] += 1
        matches = index.get(left.tape[::-1], ())
        if not matches:
            stats["residual_misses"] += 1
            if len(frontier) < 100: frontier.append({"left_tape": left.tape, "topic": left.topic, "reason": "no shared-topic response residual"})
            continue
        stats["residual_matches"] += len(matches)
        for right in matches:
            tape = left.tape + right.tape
            if len(tape) < MIN_LETTERS: stats["short_rejections"] += 1; continue
            collision = tape in existing
            if collision: stats["repository_collision_rejections"] += 1; continue
            if tape in seen: continue
            seen.add(tape); row = _audit(left, right, collision); candidates.append(row); stats["candidates"] += 1
            if row["mechanically_admitted"]: stats["mechanically_admitted"] += 1
    candidates.sort(key=lambda row: (row["mechanically_admitted"], row["letters"]), reverse=True)
    return {"status": "complete_shared_topic_elliptical_residual_no_reader_promotion", "config": {"semantic_family": "paired prompts and paired elliptical acknowledgments sharing one discourse topic", "lexical_inventory": "same bounded hand-authored action/topic inventory; no pool enlargement", "prior_act_replay": False, "repository_tape_exclusion": True, "minimum_letters": MIN_LETTERS, "maximum_letters": MAX_LETTERS, "independent_residual_matching": True, "anti_shortcut_gate": True}, "stats": dict(stats), "repository_fingerprint": fingerprint, "admitted": [row for row in candidates if row["mechanically_admitted"]], "near_misses": candidates[:100], "residual_frontier": frontier, "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "source_text_copied": False, "inventory_version": "dialogue_shared_topic_v1"}, "next_operator": "Stop the dialogue family; choose a non-dialogue semantic inventory with independent topic/state variables and the same exact residual audit.", "reader_gate": "No output is human evidence; future closures require intact-prose and shuffled-control readers."}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", required=True, type=Path); args = parser.parse_args()
    if args.out.exists(): parser.error(f"refusing to overwrite output: {args.out}")
    result = run(); args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps({"stats": result["stats"], "admitted": len(result["admitted"])}, indent=2))


if __name__ == "__main__": main()
