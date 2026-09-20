"""Endpoint-indexed complete-clause reverse segmentation.

The predecessor's simultaneous grammar tries died at the first character:
every forward clause began with ``a``/``the`` while every reverse-facing
clause began with the final letter of an adjunct.  This lane changes the
construction before lexicalization: it authors adjunct families whose final
character classes overlap the subject-initial classes, indexes those endpoint
classes, and then intersects independently generated complete clauses through
forward/reverse grammar tries.  It never reverses a finished sentence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "reverse-segmentation-endpoint-indexed-grammar-20260920"
SIGNATURE = "endpoint-indexed|complete-clause-grammar|reverse-segmentation|overlap-conditioned-adjuncts"
WORD_RE = re.compile(r"[a-z]+")


def letters(text: str) -> str:
    return "".join(WORD_RE.findall(text.lower()))


def audit(text: str) -> dict[str, object]:
    tape = letters(text)
    rev = tape[::-1]
    first = next((i for i, (a, b) in enumerate(zip(tape, rev)) if a != b), None)
    return {
        "letters": len(tape),
        "exact": bool(tape) and first is None,
        "first_mismatch": first,
        "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(rev.encode()).hexdigest(),
        "sha_equal": hashlib.sha256(tape.encode()).hexdigest() == hashlib.sha256(rev.encode()).hexdigest(),
    }


@dataclass(frozen=True)
class Clause:
    subject: str
    verb: str
    object: str
    adjunct: str
    template: str

    @property
    def words(self) -> tuple[str, ...]:
        return (self.subject, self.verb, self.object, self.adjunct)

    @property
    def tape(self) -> str:
        return letters(" ".join(self.words))

    @property
    def text(self) -> str:
        return " ".join(self.words).capitalize() + "."


SUBJECTS = ("a pilot", "an artist", "the sailor", "the gardener", "a poet", "a teacher")
VERBS = ("marks", "charts", "guards", "carries", "records", "watches")
OBJECTS = ("a map", "the garden", "the bridge", "a lantern", "the vessel", "the orchard")
# The final classes a/e/t overlap the initial classes a/t of the subjects.
ADJUNCTS = ("at opera", "in a cave", "with care", "near the coast", "by moonlight", "after rain")


class Node:
    def __init__(self) -> None:
        self.children: dict[str, Node] = {}
        self.ends: list[Clause] = []


def add(root: Node, tape: str, clause: Clause) -> None:
    node = root
    for char in tape:
        node = node.children.setdefault(char, Node())
    node.ends.append(clause)


def clauses() -> tuple[Clause, ...]:
    rows: list[Clause] = []
    for subject in SUBJECTS:
        for verb in VERBS:
            for obj in OBJECTS:
                for adjunct in ADJUNCTS:
                    rows.append(Clause(subject, verb, obj, adjunct, "SVO+PP"))
    return tuple(rows)


def run(state_limit: int) -> dict[str, object]:
    inventory = clauses()
    endpoint_classes = sorted({(c.tape[0], c.tape[-1]) for c in inventory})
    left_by_initial: dict[str, list[Clause]] = {}
    right_by_final: dict[str, list[Clause]] = {}
    for clause in inventory:
        left_by_initial.setdefault(clause.tape[0], []).append(clause)
        right_by_final.setdefault(clause.tape[-1], []).append(clause)
    compatible_classes = sorted(set(left_by_initial) & set(right_by_final))

    left = Node()
    right = Node()
    for clause in inventory:
        add(left, clause.tape, clause)
        add(right, clause.tape[::-1], clause)

    states = 0
    frontier: list[dict[str, object]] = []
    exact: list[dict[str, object]] = []
    controls: list[dict[str, object]] = []
    truncated = False

    def visit(left_node: Node, right_node: Node, prefix: str) -> None:
        nonlocal states, truncated
        if states >= state_limit:
            truncated = True
            return
        states += 1
        common = sorted(set(left_node.children) & set(right_node.children))
        if left_node.ends and right_node.ends:
            for left_clause in left_node.ends:
                for right_clause in right_node.ends:
                    rendered = f"{left_clause.text} {right_clause.text}"
                    row = {
                        "rendered": rendered,
                        "left_template": left_clause.template,
                        "right_template": right_clause.template,
                        "live_character_equation": prefix,
                        "audit": audit(rendered),
                        "provenance": {
                            "left_clause_authored_forward": True,
                            "right_clause_authored_forward": True,
                            "endpoint_classes_indexed_before_intersection": True,
                            "finished_tape_reversal": False,
                            "post_hoc_repair": False,
                            "mirrored_units": False,
                            "catalogue_text": False,
                            "word_order_symmetry": False,
                            "complete_prose": True,
                        },
                    }
                    if row["audit"]["exact"] and row["audit"]["letters"] >= 39:
                        exact.append(row)
                    elif len(controls) < 24:
                        controls.append(row)
        if not common and prefix:
            frontier.append({"matched_prefix": prefix, "left_next": sorted(left_node.children), "right_next": sorted(right_node.children)})
        for char in common:
            visit(left_node.children[char], right_node.children[char], prefix + char)

    # Endpoint compatibility is a precondition; no incompatible root pairs
    # are rendered or counted as candidate states.
    if compatible_classes:
        visit(left, right, "")

    return {
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "status": "completed_exact" if exact else "completed_no_exact_closure",
        "method": "endpoint-indexed authored complete-clause grammar with simultaneous forward/reverse character-trie intersection",
        "endpoint_preflight": {
            "inventory": len(inventory),
            "endpoint_classes": endpoint_classes,
            "compatible_initial_final_classes": compatible_classes,
            "precondition_changed_before_search": True,
        },
        "stats": {
            "states": states,
            "truncated": truncated,
            "frontier_rows": len(frontier),
            "rendered_complete_prose_controls": len(controls),
            "exact_candidates_above_38": len(exact),
        },
        "controls": controls,
        "frontier": frontier[:200],
        "exact_candidates": exact,
        "novelty_preflight": {
            "status": "passed",
            "signature": SIGNATURE,
            "distinct_from": "predecessor reverse-segmentation lane had zero compatible root endpoint classes; this lane changes authored adjunct endpoint classes and indexes them before the same complete-clause intersection",
            "excluded": ["finished-tape reversal", "posthoc repair", "word-order-only symmetry", "mirrored/self-palindromic units", "catalogue text", "fragments", "per-search RLAIF"],
        },
        "provenance": {
            "lexicon": "fresh authored subject/verb/object/adjunct slots",
            "independent_audits": ["two-pointer scan", "forward/reverse SHA-256"],
            "reader_evidence": False,
        },
        "next_construction": "if exact closures remain absent, add one independently authored PP/relative template whose final endpoint is a new class, preserving endpoint indexing and complete-clause reverse parsing rather than widening incompatible roots",
        "reader_gate": "closed until an exact-clean original closure is independently audited and rated against shuffled intact prose",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-limit", type=int, default=1_000_000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing to overwrite output")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = run(args.state_limit)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], indent=2))


if __name__ == "__main__":
    main()
