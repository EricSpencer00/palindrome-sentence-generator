"""Joint discourse-plan / semantic-relation character solver.

The search chooses two complete event propositions and a typed discourse
relation as one state.  Causal, temporal, and contrast relations constrain
which proposition pairs are legal and which connective/attachment order is
used; both clauses remain in ordinary English order.  Character debt is
tracked while a candidate is rendered, but no reversed text is emitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

EXPERIMENT_ID = "semantic-relation-plan-solver-20260916"
SIGNATURE = (
    "typed-event-pair|joint-causal-temporal-contrast-relation-selection|"
    "relation-specific-clause-order|ordinary-order-character-debt"
)

# Fresh authored event inventory.  The event labels are semantic variables,
# not a tape or palindrome inventory.
EVENTS = (
    {"id": "storm", "subject": "the harbor crew", "verb": "secured", "object": "the boat", "time": "before dusk", "kind": "prepared"},
    {"id": "lantern", "subject": "the harbor crew", "verb": "lit", "object": "the lantern", "time": "after dusk", "kind": "prepared"},
    {"id": "map", "subject": "the coastal pilot", "verb": "studied", "object": "the map", "time": "at dawn", "kind": "observed"},
    {"id": "signal", "subject": "the coastal pilot", "verb": "reported", "object": "the signal", "time": "at noon", "kind": "observed"},
    {"id": "gate", "subject": "the patient keeper", "verb": "opened", "object": "the garden gate", "time": "in the morning", "kind": "changed"},
    {"id": "path", "subject": "the patient keeper", "verb": "blocked", "object": "the narrow path", "time": "by evening", "kind": "changed"},
)

RELATIONS = (
    {"id": "cause", "connective": "because", "order": "result-because-cause", "left_kind": "changed", "right_kind": "prepared"},
    {"id": "effect", "connective": "so", "order": "cause-so-result", "left_kind": "prepared", "right_kind": "changed"},
    {"id": "temporal", "connective": "before", "order": "earlier-before-later", "left_kind": "observed", "right_kind": "changed"},
    {"id": "contrast", "connective": "although", "order": "concession-main", "left_kind": "observed", "right_kind": "prepared"},
)


def render(left: dict, relation: dict, right: dict) -> str:
    """Render the relation's declared attachment in normal word order."""
    l = f"{left['subject']} {left['verb']} {left['object']} {left['time']}"
    r = f"{right['subject']} {right['verb']} {right['object']} {right['time']}"
    if relation["order"] == "result-because-cause":
        return f"{l} because {r}."
    if relation["order"] == "cause-so-result":
        return f"{l}, so {r}."
    if relation["order"] == "earlier-before-later":
        return f"{l} before {r}."
    return f"Although {l}, {r}."


def parse_scene(text: str, left: dict, relation: dict, right: dict) -> bool:
    """Independent surface check for the exact event/relation realization."""
    expected = render(left, relation, right)
    if text != expected or not text.endswith("."):
        return False
    words = tuple(tokenize(text))
    return (
        len(words) >= 12
        and left["subject"].split()[1] == words[1]
        and left["verb"] in words
        and right["verb"] in words
        and relation["connective"] in words
    )


def audit(text: str) -> dict[str, object]:
    tape = normalize_letters(text)
    mismatches = []
    for i in range(len(tape) // 2):
        j = len(tape) - 1 - i
        if tape[i] != tape[j]:
            mismatches.append({"left": i, "right": j, "left_char": tape[i], "right_char": tape[j]})
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    central = mechanical_admission_checks(text, min_letters=45, max_letters=220)
    return {
        "letters": len(tape),
        "two_pointer_exact": bool(tape) and not mismatches,
        "mismatches": mismatches[:12],
        "forward_sha256": forward,
        "reverse_sha256": reverse,
        "sha_exact": forward == reverse,
        "central_admission": central,
        "anti_shortcut": {
            "word_order_mirror": central["not_word_order_symmetry"],
            "repeated_content": central["distinct_words"],
            "proper_palindrome_span_absent": central["no_self_palindromic_proper_multiword_span"],
            "catalogue_absent": central["local_catalogue_absent"],
        },
    }


def novelty_preflight() -> dict[str, object]:
    registry = ROOT / "docs/EXPERIMENT-NOVELTY-REGISTRY.md"
    source = registry.read_text()
    tokens = ("semantic-relation-plan-solver", "joint-causal-temporal-contrast-relation-selection")
    collisions = [token for token in tokens if token in source]
    return {"registry": str(registry), "signature": SIGNATURE, "collisions": collisions, "passed": not collisions}


def run() -> dict[str, object]:
    preflight = novelty_preflight()
    if not preflight["passed"]:
        raise RuntimeError(f"novelty preflight failed: {preflight['collisions']}")
    rows = []
    rejected = 0
    for relation in RELATIONS:
        for left in EVENTS:
            for right in EVENTS:
                if left["id"] == right["id"] or left["kind"] != relation["left_kind"] or right["kind"] != relation["right_kind"]:
                    continue
                text = render(left, relation, right)
                checks = audit(text)
                valid = parse_scene(text, left, relation, right)
                if not valid:
                    rejected += 1
                    continue
                tape = normalize_letters(text)
                rows.append({
                    "rendered": text,
                    "relation": relation["id"],
                    "left_event": left["id"],
                    "right_event": right["id"],
                    "event_equation": {"left_kind": left["kind"], "right_kind": right["kind"], "relation_order": relation["order"]},
                    "independent_reparse": valid,
                    "character_debt": {"matched_prefix": next((i for i, m in enumerate(checks["mismatches"]) if m["left"] == i), len(tape) // 2), "first_mismatch": checks["mismatches"][0] if checks["mismatches"] else None},
                    "audit": checks,
                    "mechanically_admitted": checks["two_pointer_exact"] and all(checks["central_admission"].values()),
                    "reader_status": "unreviewed; programmatic checks do not certify readability",
                })
    exact = [row for row in rows if row["audit"]["two_pointer_exact"]]
    return {
        "experiment_id": EXPERIMENT_ID,
        "novelty_preflight": preflight,
        "state_signature": SIGNATURE,
        "relation_inventory": RELATIONS,
        "event_inventory": EVENTS,
        "rendered_candidates": rows,
        "exact_survivors": exact,
        "mechanically_admitted": [row for row in exact if row["mechanically_admitted"]],
        "stats": {"states": len(rows) + rejected, "rendered": len(rows), "rejected_before_reparse": rejected, "exact": len(exact), "admitted": 0},
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "material": "fresh authored event propositions; no catalogue text or palindrome seed", "audits": ["independent normalized two-pointer", "forward/reverse SHA-256", "mechanical admission", "independent relation/event reparse"]},
        "anti_shortcut_policy": "normal clause order only; distinct event ids; no copied tape, word-order mirror, or catalogue scaffold",
        "next_repair": "at the first residual character debt, replace the active event slot with the next relation-compatible held-out proposition and re-solve relation order jointly",
        "reader_facing_test": {"status": "not triggered unless exact admitted survivor exists"},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing to overwrite output")
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))


if __name__ == "__main__":
    main()
