"""Boundary-conditioned polar-question repair on the exact pair graph.

The latest token decoder paired identical grammatical slots and lost residual
ownership.  This lane instead reuses the incoming/outgoing character pair
graph: a forward edge from the question path is matched with an incoming edge
from the answer path, so the right hand is reconstructed in ordinary English
order.  Only a small set of complete, typed question and object-fronted answer
alternatives is added at the dead boundary; this is not a vocabulary sweep.

The 44-letter discourse line is retained as a diagnostic geometry control.  It
is not a success because its answer is fragmentary and it contains a proper
palindromic multiword island.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters


@dataclass
class Edge:
    source: int
    target: int
    char: str
    text: str
    provenance: str


class Graph:
    def __init__(self) -> None:
        self.size = 2
        self.start = 0
        self.end = 1
        self.out: dict[int, list[Edge]] = defaultdict(list)
        self.inc: dict[int, list[Edge]] = defaultdict(list)

    def node(self) -> int:
        node = self.size
        self.size += 1
        return node

    def phrase(self, source: int, target: int, text: str, provenance: str) -> None:
        letters = normalize_letters(text)
        if not letters:
            raise ValueError(f"phrase has no letters: {text!r}")
        current = source
        for index, char in enumerate(letters):
            nxt = target if index == len(letters) - 1 else self.node()
            edge = Edge(current, nxt, char, text if index == 0 else "", provenance)
            self.out[current].append(edge)
            self.inc[nxt].append(edge)
            current = nxt


QUESTION_NAMES = ("Noel", "Nora", "Mara", "Eva", "Ada", "Anna")
QUESTION_PREDICATES = (
    "an era?",
    "a gas?",
    "an item?",
    "a poet?",
    "a sailor?",
    "the heir?",
    "the actor?",
    "smart?",
    "stressed?",
    "mad?",
    "raw?",
    "calm?",
    "kind?",
    "wise?",
    "noble?",
    # The cross-boundary diagnostic is included as a control.  It is not a
    # complete answer and therefore cannot be promoted by this lane.
    "an era, a gas, an item?",
)

# Complete object-fronted declaratives.  The fronted object is the semantic
# complement of the final verb; these are ordinary marked English, not fragments.
FRONTED_ANSWERS = (
    "a note, Leon saw",
    "the map, Leon saw",
    "a poem, Leon read",
    "the letter, Leon read",
    "nine memos, some men inspired Diana",
    "some prose, the editor read",
    "a red rose, the poet held",
    "old books, the scholar found",
    "a message, the nurse carried",
    "the answer, the teacher wrote",
    "Diana, Noel saw",
    "Noel, Leon met",
    "Leon, Noel met",
)

# This is kept for geometry comparison only.  It is deliberately not in the
# complete-answer inventory above.
DIAGNOSTIC_ELLIPSIS = "met in a, saga, arena, Leon saw"


def audit(text: str) -> dict[str, object]:
    tape = normalize_letters(text)
    reverse = tape[::-1]
    mismatches = [(i, len(tape) - 1 - i) for i in range(len(tape) // 2)
                  if tape[i] != tape[-1 - i]]
    forward = hashlib.sha256(tape.encode()).hexdigest()
    backward = hashlib.sha256(reverse.encode()).hexdigest()
    return {
        "letters": len(tape),
        "two_pointer_exact": bool(tape) and not mismatches,
        "mismatch_count": len(mismatches),
        "first_mismatch": mismatches[0] if mismatches else None,
        "sha256_forward": forward,
        "sha256_reverse": backward,
        "sha_equal_under_reversal": forward == backward,
    }


def build_graph(*, include_diagnostic: bool = True) -> Graph:
    graph = Graph()
    q0, q1, q2 = graph.node(), graph.node(), graph.node()
    # The question path is shared, but every lexical alternative remains an
    # explicit provenance edge in the finite graph.
    graph.phrase(graph.start, q0, "Was", "question auxiliary")
    for name in QUESTION_NAMES:
        graph.phrase(q0, q1, name, f"question subject name: {name}")
    for predicate in QUESTION_PREDICATES:
        graph.phrase(q1, q2, predicate, f"complete polar predicate: {predicate}")
    for answer in FRONTED_ANSWERS:
        graph.phrase(q2, graph.end, answer, f"complete object-fronted answer: {answer}")
    if include_diagnostic:
        graph.phrase(q2, graph.end, DIAGNOSTIC_ELLIPSIS,
                     "diagnostic discourse ellipsis; not complete prose")
    return graph


def pair_reachability(graph: Graph) -> dict[str, object]:
    """Enumerate exact paths by matching a forward and incoming edge."""
    root = (graph.start, graph.end)
    queue = deque([root])
    parent: dict[tuple[int, int], tuple[tuple[int, int], Edge, Edge] | None] = {root: None}
    centers: dict[tuple[int, int], Edge | None] = {}
    arcs: dict[tuple[int, int], list[tuple[tuple[int, int], Edge, Edge]]] = defaultdict(list)
    while queue:
        left, right = queue.popleft()
        if left == right:
            centers[(left, right)] = None
        else:
            middle = next((edge for edge in graph.out[left] if edge.target == right), None)
            if middle is not None:
                centers[(left, right)] = middle
        incoming = defaultdict(list)
        for edge in graph.inc[right]:
            incoming[edge.char].append(edge)
        for left_edge in graph.out[left]:
            for right_edge in incoming[left_edge.char]:
                target = (left_edge.target, right_edge.source)
                arcs[(left, right)].append((target, left_edge, right_edge))
                if target not in parent:
                    parent[target] = ((left, right), left_edge, right_edge)
                    queue.append(target)

    exact_paths: list[dict[str, object]] = []
    for pair in centers:
        if pair not in parent:
            continue
        # A center pair itself is reachable from root; recover the paired paths.
        left_edges: list[Edge] = []
        right_edges: list[Edge] = []
        current = pair
        while parent[current] is not None:
            previous, left_edge, right_edge = parent[current]
            left_edges.append(left_edge)
            right_edges.append(right_edge)
            current = previous
        left_edges.reverse()
        # right edges were collected from the outside toward the center; their
        # text provenance must be rendered in ordinary forward answer order.
        right_edges.reverse()
        chosen = left_edges + ([centers[pair]] if centers[pair] is not None else []) + right_edges
        phrases = [edge.text for edge in chosen if edge.text]
        rendered = " ".join(phrases).replace("Was ", "Was ", 1)
        rendered = re.sub(r"\s+([?.!,;:])", r"\1", rendered).strip()
        if rendered and not rendered.endswith("."):
            rendered += "."
        exact_paths.append({
            "rendered": rendered,
            "edge_provenance": [edge.provenance for edge in chosen if edge.text],
            "audit": audit(rendered),
        })
    unique: dict[str, dict[str, object]] = {}
    for row in exact_paths:
        unique.setdefault(str(row["rendered"]), row)
    return {
        "pair_states": len(parent),
        "pair_edges": sum(len(items) for items in arcs.values()),
        "center_states": len(centers),
        "exact_paths": list(unique.values()),
    }


def run() -> dict[str, object]:
    graph = build_graph()
    reach = pair_reachability(graph)
    rows: list[dict[str, object]] = []
    for item in reach["exact_paths"]:
        text = str(item["rendered"])
        checks = mechanical_admission_checks(text, min_letters=30, max_letters=240)
        rows.append({**item, "mechanical_checks": checks,
                     "mechanically_admitted": all(checks.values()),
                     "reader_status": "not_run; human readability is unmeasured"})
    exact = [row for row in rows if row["audit"]["two_pointer_exact"]]
    admitted = [row for row in rows if row["mechanically_admitted"]]
    diagnostic = {
        "rendered": f"Was Noel an era, a gas, an item? {DIAGNOSTIC_ELLIPSIS}.",
        "audit": audit(f"Was Noel an era, a gas, an item? {DIAGNOSTIC_ELLIPSIS}."),
        "mechanical_checks": mechanical_admission_checks(
            f"Was Noel an era, a gas, an item? {DIAGNOSTIC_ELLIPSIS}.",
            min_letters=30, max_letters=240),
        "reader_status": "diagnostic only; fragmentary answer and proper-palindromic span",
    }
    return {
        "experiment_id": "polar-question-boundary-repair-20260918",
        "signature": "typed-polar-question|object-fronted-answer|incoming-outgoing-pair-graph|boundary-conditioned-repair",
        "config": {"question_names": len(QUESTION_NAMES),
                   "question_predicates": len(QUESTION_PREDICATES),
                   "complete_answers": len(FRONTED_ANSWERS),
                   "diagnostic_ellipsis_included": True},
        "graph": {"nodes": graph.size, **reach},
        "rendered_candidates": rows,
        "diagnostic_control": diagnostic,
        "stats": {"exact": len(exact), "mechanically_admitted": len(admitted),
                   "reader_eligible": 0, "longest_exact_letters": max(
                       (row["audit"]["letters"] for row in exact), default=0)},
        "provenance": {"finished_tape_reversed": False,
                       "catalogue_text_imported": False,
                       "word_order_mirror": False,
                       "construction": "small boundary-conditioned complete phrase repair on pair graph",
                       "independent_validator": "two-pointer normalized tape plus forward/reverse SHA-256",
                       "human_readability_certified": False},
        "next_repair": "Use the first dead pair frontier to author one additional complete object-fronted answer whose outer letters satisfy the observed question boundary; retain lexical history and reject every proper palindromic subspan before readers.",
        "reader_gate": "closed until a new exact row clears strict mechanical admission and then passes randomized blinded intact-versus-shuffled reading",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], indent=2))


if __name__ == "__main__":
    main()
