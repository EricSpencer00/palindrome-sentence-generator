"""Token-boundary CSP following the polar-question graph repair.

This lane keeps the incoming/outgoing character-pair state but makes the
answer's subject, verb, and object independent transitions.  The previous
boundary repair committed complete answer phrases too early; this version lets
the residual cross those lexical boundaries while preserving a typed answer
grammar.  It includes the 44-letter discourse path only as a diagnostic
control, never as an admitted result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters


class Graph:
    def __init__(self) -> None:
        self.size = 2
        self.start, self.end = 0, 1
        self.out = defaultdict(list)
        self.inc = defaultdict(list)

    def node(self):
        node = self.size
        self.size += 1
        return node

    def phrase(self, source, target, text, provenance):
        letters = normalize_letters(text)
        current = source
        for i, char in enumerate(letters):
            nxt = target if i == len(letters) - 1 else self.node()
            edge = (current, nxt, char, text if i == 0 else "", provenance)
            self.out[current].append(edge)
            self.inc[nxt].append(edge)
            current = nxt


NAMES = ("Noel", "Nora", "Mara", "Eva", "Ada", "Anna")
PREDICATES = (
    "an era?", "a gas?", "an item?", "a poet?", "a sailor?", "the heir?",
    "the actor?", "smart?", "stressed?", "mad?", "raw?", "calm?", "kind?",
    "wise?", "noble?", "an era, a gas, an item?",
)
SUBJECTS = ("Leon", "Aron", "Aram", "Ave", "Ada", "Anna", "a poet",
            "a sailor", "the nurse", "the editor", "an aide", "some men",
            "the actor", "the heir", "our friend", "the scholar", "the teacher",
            "Mara", "Nora")
VERBS = ("saw", "read", "wrote", "found", "heard", "held", "made", "sent",
         "met", "helped", "guided", "admired", "inspired", "marked", "carried",
         "taught", "knew", "opened", "closed")
OBJECTS = (
    "a note", "the map", "a poem", "the letter", "nine memos", "some prose",
    "a red rose", "old books", "new notes", "the archive", "a message",
    "the answer", "Diana", "Noel", "Leon", "some men", "a saga", "an arena",
    "an era", "a gas", "an item", "the garden", "the harbor", "a story",
    "the book", "a small boat",
)
DIAGNOSTIC = "met in a, saga, arena, Leon saw"


def audit(text):
    tape = normalize_letters(text)
    reverse = tape[::-1]
    bad = [(i, len(tape) - 1 - i) for i in range(len(tape) // 2)
           if tape[i] != tape[-1 - i]]
    forward = hashlib.sha256(tape.encode()).hexdigest()
    backward = hashlib.sha256(reverse.encode()).hexdigest()
    return {"letters": len(tape), "two_pointer_exact": bool(tape) and not bad,
            "mismatch_count": len(bad), "first_mismatch": bad[0] if bad else None,
            "sha256_forward": forward, "sha256_reverse": backward,
            "sha_equal_under_reversal": forward == backward}


def build_graph():
    g = Graph()
    q_name, q_pred = g.node(), g.node()
    g.phrase(g.start, q_name, "Was", "question auxiliary")
    for name in NAMES:
        g.phrase(q_name, q_pred, name, f"question subject: {name}")
    for predicate in PREDICATES:
        g.phrase(q_pred, _answer_start := g.node(), predicate,
                 f"question predicate: {predicate}")
        # Each predicate receives an independent answer grammar start so the
        # lexical history remains explicit in the reconstructed path.
        answer_start = _answer_start
        front_subject, front_verb = g.node(), g.node()
        normal_verb, normal_object = g.node(), g.node()
        for obj in OBJECTS:
            g.phrase(answer_start, front_subject, obj, f"fronted object: {obj}")
        for subj in SUBJECTS:
            g.phrase(front_subject, front_verb, subj, f"fronted subject: {subj}")
        for verb in VERBS:
            g.phrase(front_verb, g.end, verb, f"fronted verb: {verb}")
        for subj in SUBJECTS:
            g.phrase(answer_start, normal_verb, subj, f"normal subject: {subj}")
        for verb in VERBS:
            g.phrase(normal_verb, normal_object, verb, f"normal verb: {verb}")
        for obj in OBJECTS:
            g.phrase(normal_object, g.end, obj, f"normal object: {obj}")
        g.phrase(answer_start, g.end, DIAGNOSTIC, "diagnostic discourse ellipsis")
    return g


def reachability(g):
    root = (g.start, g.end)
    queue = deque([root])
    parent = {root: None}
    centers = {}
    arcs = defaultdict(list)
    while queue:
        left, right = queue.popleft()
        if left == right:
            centers[(left, right)] = None
        else:
            edge = next((e for e in g.out[left] if e[1] == right), None)
            if edge is not None:
                centers[(left, right)] = edge
        incoming = defaultdict(list)
        for edge in g.inc[right]:
            incoming[edge[2]].append(edge)
        for left_edge in g.out[left]:
            for right_edge in incoming[left_edge[2]]:
                target = (left_edge[1], right_edge[0])
                arcs[(left, right)].append((target, left_edge, right_edge))
                if target not in parent:
                    parent[target] = ((left, right), left_edge, right_edge)
                    queue.append(target)
    rows = []
    for center in centers:
        if center not in parent:
            continue
        left_edges, right_edges = [], []
        current = center
        while parent[current] is not None:
            previous, left_edge, right_edge = parent[current]
            left_edges.append(left_edge)
            right_edges.append(right_edge)
            current = previous
        left_edges.reverse(); right_edges.reverse()
        edges = left_edges + ([centers[center]] if centers[center] else []) + right_edges
        text = " ".join(e[3] for e in edges if e[3]).strip()
        text = re.sub(r"\s+([?.!,;:])", r"\1", text)
        if text and not text.endswith("."):
            text += "."
        rows.append({"rendered": text,
                     "edge_provenance": [e[4] for e in edges if e[3]],
                     "audit": audit(text)})
    unique = {}
    for row in rows:
        unique.setdefault(row["rendered"], row)
    return {"pair_states": len(parent), "pair_edges": sum(map(len, arcs.values())),
            "center_states": len(centers), "exact_paths": list(unique.values())}


def run():
    graph = build_graph(); result = reachability(graph); rows = []
    non_exact_paths = []
    for row in result["exact_paths"]:
        if not row["audit"]["two_pointer_exact"]:
            non_exact_paths.append(row)
            continue
        checks = mechanical_admission_checks(row["rendered"], min_letters=30, max_letters=240)
        rows.append({**row, "mechanical_checks": checks,
                     "mechanically_admitted": all(checks.values()),
                     "reader_status": "not_run; human readability is unmeasured"})
    exact = [row for row in rows if row["audit"]["two_pointer_exact"]]
    admitted = [row for row in rows if row["mechanically_admitted"]]
    return {"experiment_id": "polar-question-token-boundary-csp-20260918",
            "signature": "typed-polar-question|token-boundary-answer-csp|incoming-outgoing-pair-graph|independent-audit",
            "config": {"names": len(NAMES), "predicates": len(PREDICATES),
                       "subjects": len(SUBJECTS), "verbs": len(VERBS), "objects": len(OBJECTS)},
            "graph": {"nodes": graph.size, **result},
            "rendered_candidates": rows,
            "non_exact_reconstruction_paths": non_exact_paths,
            "stats": {"exact": len(exact), "mechanically_admitted": len(admitted),
                       "reader_eligible": 0, "longest_exact_letters": max(
                           (row["audit"]["letters"] for row in exact), default=0)},
            "provenance": {"finished_tape_reversed": False, "catalogue_text_imported": False,
                           "word_order_mirror": False,
                           "independent_validator": "two-pointer normalized tape plus forward/reverse SHA-256",
                           "human_readability_certified": False},
            "next_repair": "Use the first dead token-boundary pair to add one complete finite-predicate complement, preserving subject/verb/object state and rejecting proper palindromic subspans.",
            "reader_gate": "closed until a new exact row clears strict mechanical admission and randomized blinded readers prefer intact prose to shuffled controls"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True, type=Path); args = ap.parse_args()
    if args.out.exists(): ap.error(f"refusing to overwrite {args.out}")
    result = run(); args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result["stats"], indent=2))


if __name__ == "__main__":
    main()
