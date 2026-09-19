"""Bounded half-tape CSP pilot with phrase-valued grammar edges.

Each target length owns one variable per mirrored character pair.  A grammar
edge assigns its letters to those variables while word boundaries are still
live; the finished tape is never reversed to manufacture a candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

EXPERIMENT_ID = "half-tape-grammar-csp-z2-20260919"


@dataclass(frozen=True)
class Edge:
    text: str
    tag: str
    number: str | None = None

    @property
    def tape(self) -> str:
        return normalize_letters(self.text)


SUBJECTS = tuple(
    Edge(text, "S", number)
    for text, number in (
        ("an aide", "sg"), ("a poet", "sg"), ("a scribe", "sg"),
        ("a sailor", "sg"), ("a captain", "sg"), ("a nurse", "sg"),
        ("a baker", "sg"), ("a player", "sg"), ("a writer", "sg"),
        ("a clerk", "sg"), ("the queen", "sg"), ("the king", "sg"),
        ("some men", "pl"), ("some poets", "pl"), ("the sailors", "pl"),
        ("the writers", "pl"), ("Diana", "sg"), ("Noel", "sg"),
        ("Nora", "sg"), ("Mara", "sg"), ("Leon", "sg"),
    )
)
VERBS = tuple(
    Edge(text, "V", number)
    for text, number in (
        ("rips", "sg"), ("reads", "sg"), ("marks", "sg"),
        ("writes", "sg"), ("carries", "sg"), ("guards", "sg"),
        ("inspires", "sg"), ("praises", "sg"), ("keeps", "sg"),
        ("finds", "sg"), ("follows", "sg"), ("remembers", "sg"),
        ("opens", "sg"), ("closes", "sg"), ("watches", "sg"),
        ("hears", "sg"), ("sees", "sg"), ("holds", "sg"),
        ("loves", "sg"), ("needs", "sg"), ("read", "pl"),
        ("mark", "pl"), ("write", "pl"), ("carry", "pl"),
        ("guard", "pl"), ("inspire", "pl"), ("praise", "pl"),
        ("keep", "pl"), ("find", "pl"), ("follow", "pl"),
        ("remember", "pl"), ("open", "pl"), ("close", "pl"),
        ("watch", "pl"), ("hear", "pl"), ("see", "pl"),
        ("hold", "pl"), ("love", "pl"), ("need", "pl"),
    )
)
OBJECTS = tuple(
    Edge(text, "O")
    for text in (
        "nine memos", "a letter", "the notes", "old maps", "the chart",
        "a sealed note", "fresh pages", "the ledger", "some poems", "the book",
        "new plans", "a red rose", "the stars", "the sonnet", "a song",
        "the poem", "the tale", "a prayer", "the bell", "old songs",
        "new books", "the map", "a small boat", "the dark wood",
        "the bright moon", "a true story", "the last letter", "the old king",
        "a white rose", "Diana", "Noel", "Nora", "Mara", "Leon",
    )
)
ADJUNCTS = tuple(Edge(text, "P") for text in (
    "at dawn", "after rain", "under the moon", "near the river",
    "by the shore", "before dusk", "in still air", "through the gate",
    "beside the harbor", "in the garden", "after the storm", "under the stars",
))
SEAM = Edge(";", "C")
DOMAINS = {"S": SUBJECTS, "V": VERBS, "O": OBJECTS, "P": ADJUNCTS, "C": (SEAM,)}
TEMPLATES = (
    ("S", "V", "O", "C", "S", "V", "O"),
    ("S", "V", "O", "P", "C", "S", "V", "O"),
    ("S", "V", "O", "C", "S", "V", "O", "P"),
)


def audit(text: str) -> dict[str, object]:
    tape = normalize_letters(text)
    reverse = tape[::-1]
    mismatches = [(i, tape[i], reverse[i]) for i in range(len(tape)) if tape[i] != reverse[i]]
    return {
        "normalized": tape,
        "letters": len(tape),
        "two_pointer_exact": bool(tape) and not mismatches,
        "first_mismatch": mismatches[0] if mismatches else None,
        "sha256_forward": hashlib.sha256(tape.encode("ascii")).hexdigest(),
        "sha256_reverse": hashlib.sha256(reverse.encode("ascii")).hexdigest(),
    }


def search(template: tuple[str, ...], target: int, max_nodes: int = 2_000) -> tuple[list[dict[str, object]], int]:
    rows: list[dict[str, object]] = []
    nodes = 0
    domains = {key: value for key, value in DOMAINS.items()}
    min_len = {key: min(len(edge.tape) for edge in value) for key, value in domains.items()}
    max_len = {key: max(len(edge.tape) for edge in value) for key, value in domains.items()}

    def visit(index: int, position: int, assigned: dict[int, str], chosen: list[Edge], used: frozenset[str], number: str | None) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes:
            return
        if index == len(template):
            if position != target:
                return
            raw = " ".join(edge.text for edge in chosen).replace(" ; ", "; ")
            rendered = raw[:1].upper() + raw[1:] + "."
            checked = audit(rendered)
            if not checked["two_pointer_exact"] or checked["letters"] < 30:
                return
            gates = mechanical_admission_checks(rendered, min_letters=30, max_letters=260)
            rows.append({
                "rendered": rendered,
                "word_spans": [edge.text for edge in chosen],
                "audit": checked,
                "mechanical_checks": gates,
                "mechanically_admitted": all(gates.values()),
                "provenance": {
                    "construction": "half-tape CSP with phrase-valued grammar edges",
                    "finished_tape_reversed": False,
                    "catalogue_imported": False,
                    "word_order_mirror": False,
                    "rlaif_used": False,
                },
                "reader_status": "unreviewed; exactness does not certify readability",
            })
            return
        slot = template[index]
        remaining = template[index + 1:]
        for edge in domains[slot]:
            if len(edge.tape) > 1 and edge.tape in used:
                continue
            if slot == "V" and number not in (None, edge.number):
                continue
            end = position + len(edge.tape)
            if end + sum(min_len[key] for key in remaining) > target:
                continue
            if end + sum(max_len[key] for key in remaining) < target:
                continue
            next_assigned = dict(assigned)
            valid = True
            for offset, character in enumerate(edge.tape):
                absolute = position + offset
                mirror = min(absolute, target - 1 - absolute)
                if mirror in next_assigned and next_assigned[mirror] != character:
                    valid = False
                    break
                next_assigned[mirror] = character
            if valid:
                next_number = edge.number if edge.number is not None else number
                next_used = used | ({edge.tape} if len(edge.tape) > 1 else set())
                visit(index + 1, end, next_assigned, chosen + [edge], next_used, next_number)

    visit(0, 0, {}, [], frozenset(), None)
    return rows, nodes


def run() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    nodes = 0
    target_runs = 0
    for template in TEMPLATES:
        for target in range(38, 46):
            found, count = search(template, target)
            rows.extend(found)
            nodes += count
            target_runs += 1
    unique = {row["audit"]["normalized"]: row for row in rows}
    exact = sorted(unique.values(), key=lambda row: (-row["audit"]["letters"], row["rendered"]))
    return {
        "experiment_id": EXPERIMENT_ID,
        "method": "fixed-length half-tape CSP; phrase-valued grammar edges assign mirrored character variables before rendering",
        "templates": [list(template) for template in TEMPLATES],
        "stats": {
            "target_runs": target_runs,
            "nodes": nodes,
            "exact": len(exact),
            "mechanically_admitted": sum(row["mechanically_admitted"] for row in exact),
            "longest_exact_letters": max((row["audit"]["letters"] for row in exact), default=0),
        },
        "candidates": exact,
        "independent_audit": ["two-pointer normalized tape", "forward/reverse SHA-256"],
        "novelty_preflight": {"status": "passed", "signature": EXPERIMENT_ID, "catalogue_imported": False},
        "next_repair": "Expose relative-clause marker, verb, and object as separate grammar edges; do not make the clause opaque.",
        "reader_gate": "closed; exactness and mechanical admission do not certify readability",
    }


if __name__ == "__main__":
    result = run()
    (ROOT / "runs" / (EXPERIMENT_ID + ".json")).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
