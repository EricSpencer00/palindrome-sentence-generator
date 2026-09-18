"""Center-out chart search from a small, authored English event.

This is a construction experiment, not a readability scorer.  The event graph
is fixed first; only then does a two-boundary chart choose lexical words and
emit their letters from the outside toward the center.  A word is never
invented to satisfy a character seam.  The run records the live frontier when
the chart closes, including an independently replayable ledger.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

MIN_LETTERS, MAX_LETTERS = 30, 260


# A short, coherent event is deliberately authored before search.  Alternatives
# are ordinary lexical realizations of typed roles, not a catalogue phrase.
EVENT = {
    "subject": ("a", "calm", "baker"),
    "predicate": ("serves",),
    "object": ("warm", "bread"),
    "recipient": ("to", "kind", "neighbors"),
}
SLOT_OPTIONS = (
    ("det", "a", ("a",)),
    ("subject_adj", "subject", ("calm", "patient", "skilled")),
    ("subject_noun", "subject", ("baker", "carpenter", "teacher")),
    ("verb", "predicate", ("serves", "prepares", "offers")),
    ("object_adj", "object", ("warm", "fresh", "plain")),
    ("object_noun", "object", ("bread", "meals", "tea")),
    ("prep", "recipient", ("to",)),
    ("recipient_adj", "recipient", ("kind", "nearby", "quiet")),
    ("recipient_noun", "recipient", ("neighbors", "guests", "students")),
)


def replay_ledger(ledger: list[dict[str, object]]) -> dict[str, object]:
    residual = ""
    cancellations = 0
    for event in ledger:
        char = str(event["char"])
        if residual:
            if char != residual[0]:
                return {"ok": False, "events_replayed": len(ledger), "cancellations": cancellations, "residual": residual}
            residual = residual[1:]
            cancellations += 1
        else:
            residual = char
    return {"ok": not residual, "events_replayed": len(ledger), "cancellations": cancellations, "residual": residual}


def render(words: tuple[str, ...]) -> str:
    return " ".join(words).capitalize() + "."


def independent_parse(text: str) -> bool:
    """Reparse the fixed event grammar without trusting the chart state."""
    words = tokenize(text)
    return len(words) == len(SLOT_OPTIONS) and words[0] == "a" and words[3] in {"serves", "prepares", "offers"} and words[6] == "to"


def audit(text: str, kind: str, provenance: tuple[str, ...]) -> dict[str, object]:
    tape = normalize_letters(text)
    gate = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    codes = [key for key, value in gate.items() if not value]
    if not independent_parse(text):
        codes.append("independent_complete_reparse_failed")
    return {
        "record_kind": kind,
        "rendered": text,
        "provenance": provenance,
        "independent_exact_audit": {"exact": bool(tape) and tape == tape[::-1], "letters": len(tape), "normalized_sha256": sha256(tape.encode()).hexdigest()},
        "independent_parse": independent_parse(text),
        "central_admission": gate,
        "mechanically_admitted": not codes,
        "rejection_codes": codes,
        "reader_status": "unreviewed; programmatic checks do not certify readability",
    }


def _words_for(role: str, assigned: dict[str, str]) -> tuple[str, ...]:
    return next(options for _, option_role, options in SLOT_OPTIONS if option_role == role and role not in assigned)


def search(*, state_limit: int, stats: Counter) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Explore the lexical chart from both word boundaries toward the center."""
    exact: list[dict[str, object]] = []
    words = [""] * len(SLOT_OPTIONS)
    deepest: dict[str, object] = {"ledger": [], "rejection": None}

    def visit(left: int, right: int, left_stream: tuple[int, str, int] | None, right_stream: tuple[int, str, int] | None,
              residual: str, owner: str, ledger: list[dict[str, object]], assigned: dict[str, str]) -> None:
        if stats["states"] >= state_limit or exact:
            return
        stats["states"] += 1
        if left_stream and left_stream[2] >= len(left_stream[1]): left_stream = None
        if right_stream and right_stream[2] >= len(right_stream[1]): right_stream = None
        if left > right and left_stream is None and right_stream is None:
            row = audit(render(tuple(words)), "centerout_event_chart_complete", tuple(words))
            if row["independent_exact_audit"]["exact"]: exact.append(row)
            return
        sides = ("right",) if residual and owner == "left" else ("left",) if residual else ("left", "right")
        for side in sides:
            stream = left_stream if side == "left" else right_stream
            index = left if side == "left" else right
            if (side == "left" and left > right) or (side == "right" and right < left): continue
            if stream is None:
                role = SLOT_OPTIONS[index][1]
                for candidate in SLOT_OPTIONS[index][2]:
                    if candidate in assigned.values() and candidate not in {"a", "to"}: continue
                    assigned[role] = candidate; words[index] = candidate
                    chars = normalize_letters(candidate)
                    nxt = (index, chars[::-1] if side == "left" else chars, 0)
                    visit(left - 1 if side == "left" else left, right + 1 if side == "right" else right, nxt if side == "left" else left_stream, nxt if side == "right" else right_stream, residual, owner, ledger, assigned)
                    assigned.pop(role, None); words[index] = ""
                continue
            index, chars, position = stream
            char = chars[position]
            event = {"side": side, "slot": index, "word": words[index], "char": char, "residual_before": residual}
            if residual and char != residual[0]:
                event.update(action="contradiction", expected=residual[0])
                if len(ledger) + 1 > len(deepest["ledger"]): deepest.update(ledger=ledger[:], rejection=event)
                continue
            residual2, owner2 = ((residual[1:], "") if residual else (char, side))
            event.update(action="cancel" if residual else "open", residual_after=residual2)
            nxt = (index, chars, position + 1)
            visit(left, right, nxt if side == "left" else left_stream, nxt if side == "right" else right_stream, residual2, owner2, ledger + [event], assigned)

    visit(0, len(SLOT_OPTIONS) - 1, None, None, "", "", [], {})
    deepest["independent_replay"] = replay_ledger(deepest["ledger"])
    deepest["emissions_including_rejection"] = len(deepest["ledger"]) + bool(deepest["rejection"])
    return exact, deepest


def run(*, state_limit: int = 100_000) -> dict[str, object]:
    stats = Counter(states=0)
    exact, deepest = search(state_limit=state_limit, stats=stats)
    seed = render(tuple(word for role in ("subject", "predicate", "object", "recipient") for word in EVENT[role]))
    control = audit(seed, "authored_event_seed_control", tuple(tokenize(seed)))
    return {
        "status": "centerout_semantics_constrained_two_boundary_chart",
        "config": {"event_fixed_before_search": True, "two_boundary_lexical_chart": True, "sentence_order_deferred": True, "no_catalogue_seed": True, "one_character_emission": True, "independent_reparse": True},
        "event_graph": EVENT,
        "seed_control": control,
        "stats": dict(stats),
        "deepest_live_frontier": deepest,
        "exact_closures": exact,
        "admitted_closures": [row for row in exact if row["mechanically_admitted"]],
        "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "material": "task-authored event graph and lexical alternatives; no catalogue text"},
        "reader_facing_next_operator": "Replace the full event topology using the recorded boundary contradiction; do not tune a single suffix or expose an unreviewed output.",
        "reader_status": "unreviewed; no programmatic result certifies readability",
    }


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--state-limit", type=int, default=100_000); args = parser.parse_args()
    if args.out.exists(): parser.error(f"refusing to overwrite {args.out}")
    result = run(state_limit=args.state_limit); args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps({"out": str(args.out), "states": result["stats"]["states"], "exact": len(result["exact_closures"])}, indent=2))


if __name__ == "__main__": main()
