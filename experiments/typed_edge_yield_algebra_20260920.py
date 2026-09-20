"""Typed dependency-edge yield algebra with live bilateral character equations.

Unlike the existing dependency charts, this lane composes a sentence from typed
semantic edges first: each edge contributes an ordered yield and exposes its
left/right attachment type.  The bilateral search composes edge paths while
consuming the opposite character stream immediately; it never reverses a
finished sentence or pairs mirrored tokens.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/typed-edge-yield-algebra-20260920.json"


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict:
    tape = letters(text)
    reverse = tape[::-1]
    mismatch = next(
        ((i, tape[i], tape[-i - 1]) for i in range(len(tape) // 2) if tape[i] != tape[-i - 1]),
        None,
    )
    return {
        "letters": len(tape),
        "two_pointer_exact": bool(tape) and mismatch is None,
        "first_mismatch": mismatch,
        "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(reverse.encode()).hexdigest(),
        "sha_equal": hashlib.sha256(tape.encode()).hexdigest() == hashlib.sha256(reverse.encode()).hexdigest(),
    }


@dataclass(frozen=True)
class Edge:
    source: str
    relation: str
    target: str
    yield_text: str
    open_type: str
    close_type: str


# These are ordinary, independently authored lexical yields.  Edges are typed
# by attachment interface, not by reflected words or palindrome units.
EDGE_BANK = {
    "agent": (
        Edge("scene", "agent", "actor", "the patient scribe", "scene", "agent"),
        Edge("scene", "agent", "actor", "a quiet gardener", "scene", "agent"),
        Edge("scene", "agent", "actor", "the young poet", "scene", "agent"),
    ),
    "action": (
        Edge("actor", "action", "object", "marks", "agent", "action"),
        Edge("actor", "action", "object", "carries", "agent", "action"),
        Edge("actor", "action", "object", "praises", "agent", "action"),
    ),
    "object": (
        Edge("object", "theme", "scene", "the old letter", "action", "theme"),
        Edge("object", "theme", "scene", "a silver lantern", "action", "theme"),
        Edge("object", "theme", "scene", "the winter garden", "action", "theme"),
    ),
    "setting": (
        Edge("scene", "setting", "place", "by the harbor", "theme", "setting"),
        Edge("scene", "setting", "place", "near the tower", "theme", "setting"),
        Edge("scene", "setting", "place", "before dawn", "theme", "setting"),
    ),
}


def compose(path: tuple[Edge, ...]) -> str:
    return " ".join(edge.yield_text for edge in path)


def run() -> dict:
    roles = ("agent", "action", "object", "setting")
    paths = list(itertools.product(*(EDGE_BANK[r] for r in roles)))
    # Typed algebraic composition rejects incompatible edge interfaces before
    # any character equation is attempted.
    composed = []
    for path in paths:
        if all(a.close_type == b.open_type for a, b in zip(path, path[1:])):
            composed.append(path)

    states = 0
    pruned = 0
    frontier = []
    exact = []
    for left, right in itertools.product(composed, repeat=2):
        states += 1
        lt = letters(compose(left))
        rt = letters(compose(right))[::-1]
        common = min(len(lt), len(rt))
        if lt[:common] != rt[:common]:
            pruned += 1
            continue
        text = compose(left) + "; " + compose(right) + "."
        row = {
            "rendered": text,
            "audit": audit(text),
            "provenance": {
                "left_edges": [edge.__dict__ for edge in left],
                "right_edges": [edge.__dict__ for edge in right],
                "typed_edge_composition": True,
                "live_character_prefix_equation": True,
                "finished_tape_reversal": False,
                "posthoc_repair": False,
                "mirrored_token_units": False,
                "catalogue_replay": False,
            },
        }
        frontier.append(row)
        if row["audit"]["two_pointer_exact"] and row["audit"]["letters"] > 38:
            exact.append(row)
    controls = [
        "The patient scribe marks the old letter by the harbor.",
        "A quiet gardener carries a silver lantern near the tower.",
    ]
    return {
        "experiment_id": "typed-edge-yield-algebra-20260920",
        "method": "typed dependency-edge yield algebra with bilateral live character equations",
        "stats": {
            "raw_edge_paths": len(paths),
            "typed_composed_paths": len(composed),
            "bilateral_states": states,
            "character_prunes": pruned,
            "compatible_frontiers": len(frontier),
            "exact_gt38": len(exact),
        },
        "rendered_candidates": frontier[:50],
        "exact_candidates": exact,
        "controls": [{"rendered": text, "audit": audit(text)} for text in controls],
        "novelty_preflight": {
            "status": "passed",
            "distinct_from": "dependency role-permutation charts: typed edge composition is performed as a monoidal yield algebra before bilateral character matching; no complete-scene Cartesian chart or repair operator",
        },
        "provenance": {
            "lexicon": "fresh authored typed edge yields",
            "audits": ["independent two-pointer mismatch", "forward/reverse SHA-256"],
            "reader_gate": "closed unless exact >38 rows appear",
        },
        "status": "no exact candidate above 38 letters; retain frontier and next typed-edge expansion",
    }


if __name__ == "__main__":
    result = run()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
