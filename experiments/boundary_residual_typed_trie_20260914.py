"""Joint boundary-residual trie over complete typed clause pairs.

Both sides are syntax-first: clauses are complete typed propositions before
their tapes enter the product.  The right side is indexed by reversed tape,
so each emitted character is simultaneously an exact palindrome constraint
and a live residual of a complete valid clause.  This is deliberately a new
operator, not a readability score applied after unconstrained generation.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.syntax_first_clause_pair_20260914 import (
    exact_audit, enumerate_clauses, normalize,
)


@dataclass
class Node:
    children: dict[str, int] = field(default_factory=dict)
    leaves: list[int] = field(default_factory=list)


def build_trie(rows: list[dict]) -> list[Node]:
    nodes = [Node()]
    for idx, row in enumerate(rows):
        node = 0
        for char in reversed(row["tape"]):
            nxt = nodes[node].children.get(char)
            if nxt is None:
                nxt = len(nodes)
                nodes[node].children[char] = nxt
                nodes.append(Node())
            node = nxt
        nodes[node].leaves.append(idx)
    return nodes


def run(max_left: int = 250_000, max_pairs: int = 100) -> dict:
    if max_left < 1 or max_pairs < 1:
        raise ValueError("bounds must be positive")
    rows = enumerate_clauses()
    trie = build_trie(rows)
    closures = []
    residual_histogram: dict[int, int] = {}
    traversed = 0
    for left in rows[:max_left]:
        node = 0
        consumed = 0
        for char in left["tape"]:
            nxt = trie[node].children.get(char)
            if nxt is None:
                break
            node = nxt
            consumed += 1
        residual_histogram[consumed] = residual_histogram.get(consumed, 0) + 1
        traversed += 1
        if consumed != len(left["tape"]):
            continue
        # Every leaf has a complete syntactic clause; no partial word or
        # fragment can be emitted as a closure.
        for right_idx in trie[node].leaves:
            right = rows[right_idx]
            rendered = left["text"].capitalize() + "; " + right["text"] + "."
            audit = exact_audit(rendered)
            if not audit["exact"]:
                raise AssertionError("trie closure failed independent audit")
            if len(closures) < max_pairs:
                closures.append({"left": left, "right": right,
                                 "rendered": rendered,
                                 "independent_exact_audit": audit,
                                 "reader_status": "human-unreviewed"})
    return {
        "status": "boundary_residual_typed_trie",
        "config": {"max_left_clauses": max_left, "max_reported_pairs": max_pairs,
                    "complete_typed_clause_on_both_sides": True,
                    "reverse_tape_trie": True, "character_residual_pruning": True,
                    "catalogue_text": False, "known_palindrome_units": False,
                    "readability_requires_blinded_humans": True},
        "inventory": {"typed_clauses": len(rows), "trie_nodes": len(trie),
                       "traversed_left_clauses": traversed},
        "residual_histogram": {str(k): v for k, v in sorted(residual_histogram.items())},
        "exact_closure_count_seen": sum(1 for _ in closures),
        "rendered_candidates": closures,
        "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                       "lexical_source": "authored ordinary pools in syntax_first_clause_pair_20260914.py",
                       "operator": "left complete clause against reversed right-clause character trie"},
        "next_constructive_operator": "Add a role-preserving boundary repair trie: at the first residual dead end, substitute one subject, verb, or object from a fresh authored valency-compatible pool and resume the joint traversal; retain only repairs whose entire clause reparses independently.",
        "scope": "Bounded joint syntax/tape search; programmatic exactness does not certify human readability.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-left", type=int, default=250_000)
    args = ap.parse_args()
    if args.out.exists():
        ap.error("output already exists")
    result = run(max_left=args.max_left)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"typed_clauses": result["inventory"]["typed_clauses"],
                      "trie_nodes": result["inventory"]["trie_nodes"],
                      "traversed": result["inventory"]["traversed_left_clauses"],
                      "closures": result["exact_closure_count_seen"]}, indent=2))


if __name__ == "__main__":
    main()
