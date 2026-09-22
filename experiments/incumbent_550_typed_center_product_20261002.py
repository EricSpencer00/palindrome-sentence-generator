"""Intersect a typed finite-clause grammar at the 550-letter midpoint."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.incumbent_498_deep_clause_transducer_20261002 import (
    audit,
    independent_tape,
)


PARENT = ROOT / "runs" / "incumbent-544-cross-boundary-seam-repair-20261002.json"
OUT = ROOT / "runs" / "incumbent-550-typed-center-product-20261002.json"
PARENT_SHA256 = "3040f0c4ac28002aa0edd7ce2fd920751b10e4a4f5430d82de3e46d09b3e7673"
WINNER_SHA256 = "29470b5ab408c402e8796530123357fea6a74aa4bdf14f7f1b2a601dbecc94fa"
OLD_WINDOW = "Leon, I saw diaper. Repaid was I, Noel; spam's"

NAMES = (
    "Aram", "Mara", "Aron", "Nora", "Aidan", "Nadia", "Tara", "Sara",
    "Noel", "Leon", "Liam", "Diana", "Eva", "Ada", "Iris", "Nina",
)
DETERMINERS = ("a", "the")
NOUNS = {
    "rat": "animal",
    "ram": "animal",
    "cat": "animal",
    "dog": "animal",
    "eel": "animal",
    "map": "artifact",
    "note": "text",
    "book": "text",
    "park": "place",
}
VERBS = {
    "stops": {"animal", "person"},
    "spots": {"person", "animal"},
    "sees": {"person", "animal"},
    "reads": {"text"},
    "notes": {"person", "text"},
    "maps": {"place"},
    "meets": {"person"},
    "calls": {"person"},
    "helps": {"person"},
    "draws": {"artifact"},
    "saves": {"person", "animal"},
    "serves": {"person"},
    "greets": {"person"},
    "finds": {"person", "animal", "artifact"},
    "watches": {"person", "animal"},
    "repaid": {"person"},
    "was": set(),
    "saw": {"person", "animal", "artifact"},
}


@dataclass(frozen=True)
class State:
    left_buffer: str = ""
    right_buffer: str = ""
    left_subject: str = ""
    right_object: str = ""
    left_verb: str = ""
    right_verb: str = ""
    determiner: str = ""
    right_subject: str = ""
    noun: str = ""
    trace: tuple[dict[str, object], ...] = ()


def advance(state: State, left: str, right: str, **updates: str) -> State | None:
    left_buffer = state.left_buffer + independent_tape(left)
    right_buffer = state.right_buffer + independent_tape(right)[::-1]
    limit = min(len(left_buffer), len(right_buffer))
    matched = 0
    while matched < limit and left_buffer[matched] == right_buffer[matched]:
        matched += 1
    if matched < limit:
        return None
    left_buffer = left_buffer[matched:]
    right_buffer = right_buffer[matched:]
    owner = "-" if not left_buffer and not right_buffer else ("L" if left_buffer else "R")
    residual = left_buffer or right_buffer
    event = {
        "left": left,
        "right_outside_in": right,
        "matched": matched,
        "owner": owner,
        "residual": residual,
    }
    values = {
        "left_buffer": left_buffer,
        "right_buffer": right_buffer,
        "left_subject": state.left_subject,
        "right_object": state.right_object,
        "left_verb": state.left_verb,
        "right_verb": state.right_verb,
        "determiner": state.determiner,
        "right_subject": state.right_subject,
        "noun": state.noun,
        "trace": (*state.trace, event),
    }
    values.update(updates)
    return State(**values)


def run_product() -> tuple[list[int], list[State]]:
    states = [State()]
    layer_counts = [len(states)]

    states = [next_state for state in states if (next_state := advance(state, "Leon", "Noel"))]
    layer_counts.append(len(states))

    states = [
        next_state
        for state in states
        for left_subject in NAMES
        for right_object in NAMES
        if (
            next_state := advance(
                state,
                left_subject,
                right_object,
                left_subject=left_subject,
                right_object=right_object,
            )
        )
    ]
    layer_counts.append(len(states))

    states = [
        next_state
        for state in states
        for left_verb in VERBS
        for right_verb in VERBS
        if (
            next_state := advance(
                state,
                left_verb,
                right_verb,
                left_verb=left_verb,
                right_verb=right_verb,
            )
        )
    ]
    layer_counts.append(len(states))

    states = [
        next_state
        for state in states
        for determiner in DETERMINERS
        for right_subject in NAMES
        if (
            next_state := advance(
                state,
                determiner,
                right_subject,
                determiner=determiner,
                right_subject=right_subject,
            )
        )
    ]
    layer_counts.append(len(states))

    accepted = []
    for state in states:
        for noun, noun_type in NOUNS.items():
            if noun_type not in VERBS[state.left_verb]:
                continue
            if "person" not in VERBS[state.right_verb]:
                continue
            next_state = advance(state, noun, "", noun=noun)
            if next_state and not next_state.left_buffer and not next_state.right_buffer:
                accepted.append(next_state)
    layer_counts.append(len(accepted))
    return layer_counts, accepted


def center_text(state: State) -> str:
    return (
        f"Leon, {state.left_subject} {state.left_verb} "
        f"{state.determiner} {state.noun}. "
        f"{state.right_subject} {state.right_verb} {state.right_object}, Noel."
    )


def selection_key(row: dict[str, object]) -> tuple[object, ...]:
    state = row["state"]
    distinct_penalty = state["left_verb"] == state["right_verb"]
    semantic_rank = 0 if (
        state["left_verb"], state["right_verb"]
    ) == ("stops", "spots") else 1
    noun_rank = 0 if state["noun"] == "rat" else 1
    return (-row["audit"]["letters"], distinct_penalty, semantic_rank, noun_rank, row["center"])


def build_payload() -> dict[str, object]:
    parent_payload = json.loads(PARENT.read_text())
    parent = parent_payload["rows"][0]
    assert parent["audit"]["letters"] == 550
    assert parent["audit"]["sha256_forward"] == PARENT_SHA256
    parent_rendered = str(parent["rendered"])
    assert parent_rendered.count(OLD_WINDOW) == 1

    layer_counts, accepted = run_product()
    assert layer_counts == [1, 1, 9, 45, 405, 54]
    rows = []
    for index, state in enumerate(accepted):
        center = center_text(state)
        rendered = parent_rendered.replace(OLD_WINDOW, f"{center} Spam's")
        result_audit = audit(rendered)
        assert result_audit["two_pointer_exact"]
        assert result_audit["byte_pointer_exact"]
        assert result_audit["project_validator_exact"]
        rows.append({
            "id": f"typed-center-{index:02d}",
            "center": center,
            "rendered": rendered,
            "audit": result_audit,
            "state": {
                "left_subject": state.left_subject,
                "right_object": state.right_object,
                "left_verb": state.left_verb,
                "right_verb": state.right_verb,
                "determiner": state.determiner,
                "right_subject": state.right_subject,
                "noun": state.noun,
            },
            "residual_trace": list(state.trace),
            "parent_artifact": str(PARENT.relative_to(ROOT)),
            "parent_sha256": PARENT_SHA256,
            "human_certified": False,
        })

    assert len({row["center"] for row in rows}) == 54
    rows.sort(key=selection_key)
    winner = rows[0]
    assert winner["center"] == "Leon, Aidan stops a rat. Tara spots Nadia, Noel."
    assert winner["audit"]["letters"] == 558
    assert winner["audit"]["sha256_forward"] == WINNER_SHA256

    return {
        "experiment_id": "incumbent-550-typed-center-product-20261002",
        "method": (
            "intersect two finite-clause feature grammars online while carrying "
            "the exact opposing-character residual"
        ),
        "parent": {
            "artifact": str(PARENT.relative_to(ROOT)),
            "letters": 550,
            "sha256": PARENT_SHA256,
            "source_498_artifact": "runs/overhang-growth-from-240-20261001.json",
            "source_498_sha256": "e809a2a05a414347615f68f27f6b4974aa00ede287fdb2a4b23f6ffbadec9032",
        },
        "grammar": {
            "left": "VOCATIVE PROPER.sg V_FIN.trans.sg DET NOUN",
            "right": "PROPER.sg V_FIN.trans.sg PROPER.object VOCATIVE",
            "names": list(NAMES),
            "verbs": {verb: sorted(types) for verb, types in VERBS.items()},
            "determiners": list(DETERMINERS),
            "nouns": NOUNS,
            "valency_checked_before_acceptance": True,
            "punctuation_added_after_acceptance": True,
        },
        "search": {
            "layer_counts": layer_counts,
            "terminal_unique_exact_closures": len(rows),
            "selection_rule": [
                "maximum full length",
                "distinct predicates",
                "prefer stops(animal) plus spots(person)",
                "prefer rat/Tara boundary shift",
                "lexicographic center",
            ],
        },
        "stats": {
            "maximum_live_states": max(layer_counts),
            "independently_exact_children": len(rows),
            "unique_centers": len(rows),
            "longest_letters": max(row["audit"]["letters"] for row in rows),
        },
        "active_method_winner": winner["id"],
        "rows": rows,
    }


def main() -> None:
    payload = build_payload()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))
    winner = next(row for row in payload["rows"] if row["id"] == payload["active_method_winner"])
    print(winner["rendered"])


if __name__ == "__main__":
    main()
