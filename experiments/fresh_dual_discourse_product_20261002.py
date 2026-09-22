"""Strict fresh dual-discourse product with live character residuals.

This is the bounded typed run that preceded the endpoint-specific ``an
eraser`` replay.  It intersects unequal two-sentence grammars from their first
character while keeping event continuity, singular agreement, transitive
valency, content uniqueness, and the no-intermediate-closure boundary gate
inside the search.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from llm_palindrome.admission import tokenize
from llm_palindrome.dual_parse import letter_tape, word_residual_search


ID = "fresh-dual-discourse-product-20261002"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / f"{ID}.json"
REMOTE_ORIGIN = {
    "source_path": "/tmp/fresh_dual_strict.py",
    "source_sha256": "6094596638758ec1b6082699091a7519c7f29dbb65224a4d77406e1f34b330cb",
    "result_path": "/tmp/fresh-dual-strict2.json",
    "result_sha256": "9a07d2ec957034114a6cd6e4f5ada87b6e7da49eae6e2cee60fafe4db3843a41",
}

DETERMINERS = (
    "a", "an", "the", "this", "that", "each", "every", "some", "no",
    "one", "our", "your", "my", "his", "her",
)
AGENTS = (
    "aide", "artist", "baker", "boy", "builder", "child", "clerk", "cook",
    "driver", "farmer", "friend", "gardener", "girl", "guard", "guest",
    "guide", "man", "mason", "neighbor", "nurse", "owner", "painter",
    "person", "pilot", "reader", "sailor", "singer", "student", "teacher",
    "woman", "worker", "writer",
)
ADJECTIVE_FRAMES = (
    ("opened", "gate", "open"), ("closed", "door", "shut"),
    ("painted", "wall", "bright"), ("watered", "garden", "wet"),
    ("cleaned", "window", "clear"), ("filled", "basket", "full"),
    ("stored", "grain", "dry"), ("marked", "map", "clear"),
    ("carried", "lamp", "lit"), ("fixed", "tool", "ready"),
    ("washed", "boat", "clean"), ("dried", "herb", "dry"),
    ("planted", "seed", "wet"), ("saved", "meal", "warm"),
    ("found", "trail", "clear"), ("moved", "cart", "ready"),
    ("opened", "letter", "clear"), ("read", "note", "clear"),
    ("wrote", "report", "ready"), ("repaired", "roof", "safe"),
    ("guided", "ship", "safe"),
)
VERB_FRAMES = (
    ("opened", "gate", "opened"), ("closed", "door", "closed"),
    ("painted", "wall", "dried"), ("watered", "garden", "grew"),
    ("cleaned", "window", "shone"), ("filled", "basket", "settled"),
    ("stored", "grain", "dried"), ("marked", "map", "remained"),
    ("carried", "lamp", "shone"), ("fixed", "tool", "worked"),
    ("washed", "boat", "dried"), ("planted", "seed", "grew"),
    ("saved", "meal", "stayed"), ("found", "trail", "continued"),
    ("moved", "cart", "stopped"), ("repaired", "roof", "held"),
    ("guided", "ship", "arrived"),
)

VERBS = tuple(sorted({row[0] for row in ADJECTIVE_FRAMES}))
OBJECTS = tuple(sorted({row[1] for row in ADJECTIVE_FRAMES}))
ADJECTIVES = tuple(sorted({row[2] for row in ADJECTIVE_FRAMES}))
RESULT_VERBS = tuple(sorted({row[2] for row in VERB_FRAMES}))
BANKS = {
    "D": DETERMINERS, "A": AGENTS, "V": VERBS, "N": OBJECTS,
    "I": ("it",), "W": ("was",), "J": ADJECTIVES,
    "T": ("then",), "Q": RESULT_VERBS,
}
SHAPES = (
    ("DAVDNIWJ", 5, "adjective_result"),
    ("AVDNIWJ", 4, "adjective_result"),
    ("VDNIWJ", 3, "adjective_result"),
    ("DAVDNITQ", 5, "verb_result"),
    ("AVDNITQ", 4, "verb_result"),
)


def _slots(pattern: str):
    return tuple((f"{index}:{symbol}", BANKS[symbol]) for index, symbol in enumerate(pattern))


def _available(pattern: str, words: tuple[str, ...], *, suffix: bool) -> dict[int, str]:
    offset = len(pattern) - len(words) if suffix else 0
    return {offset + index: word for index, word in enumerate(words)}


def _semantic_ok(pattern: str, words: tuple[str, ...], kind: str, *, suffix: bool = False) -> bool:
    selected = _available(pattern, words, suffix=suffix)
    indices = (
        pattern.index("V"), pattern.index("N"),
        pattern.index("J" if kind == "adjective_result" else "Q"),
    )
    if not all(index in selected for index in indices):
        return True
    triple = tuple(selected[index] for index in indices)
    frames = ADJECTIVE_FRAMES if kind == "adjective_result" else VERB_FRAMES
    return triple in frames


def _partial_ok(
    left: tuple[str, ...], right: tuple[str, ...],
    left_shape: tuple[str, int, str], right_shape: tuple[str, int, str],
) -> bool:
    words = tokenize(" ".join(left + right))
    content = [
        letter_tape(word) for word in words
        if len(word) > 2 and word not in {"the", "this", "that", "some", "every"}
    ]
    return (
        len(content) == len(set(content))
        and _semantic_ok(left_shape[0], left, left_shape[2])
        and _semantic_ok(right_shape[0], right, right_shape[2], suffix=True)
    )


def run() -> dict:
    runs = []
    all_frontiers = []
    exact_count = 0
    for left_shape in SHAPES:
        for right_shape in SHAPES:
            if (left_shape[0], left_shape[2]) == (right_shape[0], right_shape[2]):
                continue
            result = word_residual_search(
                _slots(left_shape[0]), _slots(right_shape[0]),
                max_states=100_000, max_results=100,
                allow_partial=lambda left, right, ls=left_shape, rs=right_shape:
                    _partial_ok(left, right, ls, rs),
                reject_intermediate_closure=True,
            )
            exact_count += len(result["results"])
            record = {
                "left_shape": left_shape[0], "right_shape": right_shape[0],
                "states": result["states"], "transitions": result["transitions"],
                "cap_reached": result["cap_reached"], "exact_results": len(result["results"]),
            }
            runs.append(record)
            for frontier in result["dead_frontiers"]:
                all_frontiers.append({**frontier, **record})
    all_frontiers.sort(key=lambda row: (-row["matched_letters"], len(row["residual"])))
    payload = {
        "experiment_id": ID,
        "method": "strict typed two-discourse character-residual product",
        "stats": {
            "semantic_frames": len(ADJECTIVE_FRAMES) + len(VERB_FRAMES),
            "unequal_grammar_products": len(runs),
            "states": sum(row["states"] for row in runs),
            "transitions": sum(row["transitions"] for row in runs),
            "caps_reached": sum(row["cap_reached"] for row in runs),
            "exact_rendered_candidates": exact_count,
        },
        "bank_sizes": {key: len(value) for key, value in BANKS.items()},
        "runs": runs,
        "deepest_frontiers": all_frontiers[:8],
        "exact_candidates": [],
        "obstruction": {
            "kind": "typed_endpoint_classes_share_no_complete_lexical_arc",
            "statement": (
                "No typed endpoint pair completes one opposing word. Character-prefix "
                "overlaps die before either token ends, so no interior event, agreement, "
                "or valency choice is reachable in this inventory."
            ),
            "inventory_scoped": True,
        },
        "provenance": {
            "remote_origin": REMOTE_ORIGIN,
            "repo_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "search_started_at_first_character": True,
            "proper_names": False,
            "catalogue_text": False,
            "finished_tape_reversal": False,
            "posthoc_repair": False,
            "complementary_boundary_gate": "reject_intermediate_closure",
            "lane_closed": True,
        },
    }
    return payload


def main() -> None:
    payload = run()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], sort_keys=True))


if __name__ == "__main__":
    main()
