"""Second center-out repair: a typed question--answer event.

The baker chart failed at its subject/recipient boundary.  This experiment
changes the whole topology to a question about an answer, and names the repair
operator ``endpoint_residual_gate``: the right object noun opens the residual;
the left auxiliary must match it before the subject is selected.  No suffix or
prebuilt palindrome is supplied.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

SLOTS = (
    ("auxiliary", ("can", "may", "will")),
    ("determiner_subject", ("the",)),
    ("subject_adjective", ("patient", "careful", "quiet")),
    ("subject_noun", ("tutor", "teacher", "editor")),
    ("predicate", ("answer", "explain", "clarify")),
    ("determiner_object", ("a", "the")),
    ("object_adjective", ("clear", "simple", "direct")),
    ("object_noun", ("question", "problem", "task")),
)
EVENT = {"kind": "question_answer", "meaning": "A tutor answers a clear question.", "roles": {"speaker": "tutor", "act": "answer", "object": "question"}}


def replay(ledger: list[dict[str, object]]) -> dict[str, object]:
    residual = ""; cancellations = 0
    for event in ledger:
        char = str(event["char"])
        if residual:
            if char != residual[0]:
                return {"ok": False, "events_replayed": len(ledger), "cancellations": cancellations, "residual": residual}
            residual = residual[1:]; cancellations += 1
        else: residual = char
    return {"ok": not residual, "events_replayed": len(ledger), "cancellations": cancellations, "residual": residual}


def render(words: tuple[str, ...]) -> str:
    return " ".join(words).capitalize() + "?"


def independent_parse(text: str) -> bool:
    words = tokenize(text)
    return len(words) == len(SLOTS) and words[0] in SLOTS[0][1] and words[1] == "the" and words[4] in SLOTS[4][1] and words[5] in {"a", "the"}


def audit(text: str, kind: str, provenance: tuple[str, ...]) -> dict[str, object]:
    tape = normalize_letters(text); gate = mechanical_admission_checks(text, min_letters=30, max_letters=220)
    codes = [key for key, value in gate.items() if not value]
    if not independent_parse(text): codes.append("independent_complete_reparse_failed")
    return {"record_kind": kind, "rendered": text, "provenance": provenance,
            "independent_exact_audit": {"exact": bool(tape) and tape == tape[::-1], "letters": len(tape), "normalized_sha256": sha256(tape.encode()).hexdigest()},
            "independent_parse": independent_parse(text), "central_admission": gate, "mechanically_admitted": not codes,
            "rejection_codes": codes, "reader_status": "unreviewed; programmatic checks do not certify readability"}


def run(*, state_limit: int = 100_000) -> dict[str, object]:
    stats = Counter(states=0, endpoint_pairs=0, endpoint_match_pairs=0); deepest = {"ledger": [], "rejection": None}; exact = []
    words = [""] * len(SLOTS); assigned: dict[int, str] = {}

    # Endpoint-residual gate: choose the answer object first, and test every
    # auxiliary against its first letter before touching subject slots.
    for right_word in SLOTS[-1][1]:
        for left_word in SLOTS[0][1]:
            stats["endpoint_pairs"] += 1
            residual = normalize_letters(right_word)[0]
            left_char = normalize_letters(left_word)[0]
            if left_char != residual:
                deepest = {"ledger": [{"side": "right", "slot": 7, "word": right_word, "char": residual, "action": "open", "residual_after": residual}], "rejection": {"side": "left", "slot": 0, "word": left_word, "char": left_char, "expected": residual, "action": "endpoint_residual_gate"}}
                continue
            stats["endpoint_match_pairs"] += 1
            words[0], words[7] = left_word, right_word; assigned[0], assigned[7] = left_word, right_word
            # Continue only through typed interior alternatives; endpoint
            # compatibility is never repaired by changing a character token.
            for dsub in SLOTS[1][1]:
                for adj in SLOTS[2][1]:
                    for noun in SLOTS[3][1]:
                        for pred in SLOTS[4][1]:
                            for dobj in SLOTS[5][1]:
                                for oadj in SLOTS[6][1]:
                                    if stats["states"] >= state_limit: break
                                    stats["states"] += 1
                                    candidate = (left_word, dsub, adj, noun, pred, dobj, oadj, right_word)
                                    row = audit(render(candidate), "question_answer_endpoint_gate_complete", candidate)
                                    if row["independent_exact_audit"]["exact"]: exact.append(row)
            words[0] = words[7] = ""; assigned.clear()
    return {"status": "centerout_question_answer_endpoint_residual_chart", "repair_operator": "endpoint_residual_gate",
            "config": {"event_fixed_before_search": True, "different_topology_from_baker": True, "two_boundary_chart": True, "right_endpoint_first": True, "subject_deferred_until_endpoint_gate": True, "constructed_suffix": False, "independent_reparse": True},
            "event_graph": EVENT, "seed_control": audit("Can the patient tutor answer a clear question?", "authored_question_answer_seed_control", tuple(tokenize("Can the patient tutor answer a clear question?"))),
            "stats": dict(stats), "deepest_live_frontier": {**deepest, "independent_replay": replay(deepest["ledger"]), "emissions_including_rejection": len(deepest["ledger"]) + bool(deepest["rejection"])},
            "exact_closures": exact, "admitted_closures": [row for row in exact if row["mechanically_admitted"]],
            "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "material": "authored typed question-answer event; no catalogue text"},
            "reader_facing_next_operator": "Replace the complete question-answer event with a new endpoint-compatible semantic topology; do not construct a suffix or alter a single boundary character.",
            "reader_status": "unreviewed; no programmatic result certifies readability"}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); parser.add_argument("--state-limit", type=int, default=100_000); args = parser.parse_args()
    if args.out.exists(): parser.error(f"refusing to overwrite {args.out}")
    result = run(state_limit=args.state_limit); args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps({"out": str(args.out), "states": result["stats"]["states"], "exact": len(result["exact_closures"])}, indent=2))


if __name__ == "__main__": main()
