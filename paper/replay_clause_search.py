"""Replay the saved 672-letter clause search from its pinned small inputs.

This module does not build a new archive index and does not scan repository
history. It replays the deterministic DFS using a caller-supplied parent and
relation-count mapping.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping


ROOT = Path(__file__).resolve().parents[1]
PARENT_PATH = ROOT / "runs" / "incumbent-560-outer-causal-scene-20261002.json"
INDEX_PATH = ROOT / "runs" / "incumbent-672-global-novelty-snapshot-20260922.json"
SAVED_RESULT_PATH = ROOT / "runs" / "incumbent-672-discourse-linked-reverse-chain-20260922.json"
GENERATOR_PATH = ROOT / "experiments" / "incumbent_672_discourse_linked_reverse_chain_20260922.py"
RESULT_PATH = ROOT / "paper" / "clause_search_replay.json"

PARENT_ID = "outer-causal-scene-568-working-incumbent"
PARENT_SHA256 = "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
EXPECTED_CHILD_SHA256 = "ed6290e2459a142ace98a5d20219eff252c6c89d1dae47115289c11211ec1bee"
LEFT_CUT, RIGHT_CUT = 48, 520
CHAIN_LENGTH = 4
ENTITIES = tuple(
    "Nora Leon Aron Noel Mara Aram Nadia Aidan Liam Ira".split()
)
PREDICATES = ("stops", "spots", "sees")


def normalize(text: str) -> str:
    return "".join(char.lower() for char in text if char.isascii() and char.isalpha())


def raw_after_letters(text: str, count: int) -> int:
    seen = 0
    for index, char in enumerate(text):
        if char.isascii() and char.isalpha():
            seen += 1
            if seen == count:
                return index + 1
    raise ValueError(count)


def outside_in(tape: str) -> dict[str, object]:
    left, right = 0, len(tape) - 1
    while left < right and tape[left] == tape[right]:
        left += 1
        right -= 1
    exact = left >= right
    return {
        "exact": bool(tape) and exact,
        "letters": len(tape),
        "first_mismatch": None if exact else {
            "offset": left,
            "left": tape[left],
            "right": tape[right],
        },
    }


@dataclass(frozen=True)
class Clause:
    subject: str
    predicate: str
    object: str

    @property
    def surface(self) -> str:
        return f"{self.subject} {self.predicate} {self.object}."

    @property
    def tape(self) -> str:
        return normalize(self.surface)

    @property
    def relation(self) -> str:
        return f"{self.subject.casefold()} {self.predicate} {self.object.casefold()}"

    @property
    def frame(self) -> str:
        return f"{self.subject.casefold()}|{self.predicate}|{self.object.casefold()}"


def iter_clauses(subject: str | None = None,
                 object_: str | None = None) -> Iterator[Clause]:
    subjects = (subject,) if subject is not None else ENTITIES
    objects = (object_,) if object_ is not None else ENTITIES
    for subj in subjects:
        for predicate in PREDICATES:
            for obj in objects:
                yield Clause(subj, predicate, obj)


class ReverseResidualGrammar:
    def __init__(self) -> None:
        self.states_examined = 0

    def consume_clause(self, left_tape: str,
                       object_constraint: str | None) -> list[Clause]:
        frontier = list(iter_clauses(object_=object_constraint))
        for offset, emitted in enumerate(left_tape):
            before = len(frontier)
            frontier = [
                candidate for candidate in frontier
                if offset < len(candidate.tape)
                and candidate.tape[::-1][offset] == emitted
            ]
            self.states_examined += before
            if not frontier:
                return []
        return [candidate for candidate in frontier
                if len(candidate.tape) == len(left_tape)]


def replay(parent_surface: str,
           prior_relations: Mapping[str, int]) -> dict[str, object]:
    """Run the pinned first-success DFS without reading files or Git state."""
    parent_tape = normalize(parent_surface)
    parent_audit = outside_in(parent_tape)
    if len(parent_tape) != 568 or not parent_audit["exact"]:
        raise ValueError("parent_surface must be the exact 568-letter parent")

    left_raw = raw_after_letters(parent_surface, LEFT_CUT) + 1
    right_raw = raw_after_letters(parent_surface, RIGHT_CUT) + 1
    if (left_raw, right_raw) != (62, 722):
        raise ValueError("pinned sentence-boundary offsets changed")
    if parent_surface[left_raw - 1] != "." or parent_surface[right_raw - 1] != ".":
        raise ValueError("pinned insertion points no longer follow periods")

    decoder = ReverseResidualGrammar()
    rejected: list[str] = []
    accepted: tuple[list[Clause], list[Clause]] | None = None

    def search(left_chain: list[Clause], right_reverse: list[Clause]) -> None:
        nonlocal accepted
        if accepted is not None:
            return
        if len(left_chain) == CHAIN_LENGTH:
            right_chain = list(reversed(right_reverse))
            if all(
                right_chain[index].object == right_chain[index + 1].subject
                for index in range(CHAIN_LENGTH - 1)
            ):
                relations = [clause.relation for clause in left_chain + right_chain]
                unique = len(set(relations)) == len(relations)
                known_absent = all(prior_relations.get(relation, 0) == 0
                                   for relation in relations)
                predicate_variety = len({clause.predicate for clause in left_chain}) >= 2
                complete = all(
                    clause.surface.endswith(".") and bool(clause.tape)
                    for clause in left_chain + right_chain
                )
                if unique and known_absent and predicate_variety and complete:
                    accepted = (left_chain[:], right_chain)
                else:
                    rejected.append("rejected_chain_gate")
            return

        subject = left_chain[-1].object if left_chain else None
        for left_clause in iter_clauses(subject=subject):
            if any(left_clause.surface == prior.surface for prior in left_chain):
                continue
            if left_clause.subject == left_clause.object:
                continue
            if left_clause.tape == left_clause.tape[::-1]:
                continue
            object_constraint = right_reverse[-1].subject if right_reverse else None
            decoded = decoder.consume_clause(left_clause.tape, object_constraint)
            if not decoded:
                rejected.append("rejected_residual")
                continue
            for right_clause in decoded:
                if right_clause.subject == right_clause.object:
                    continue
                if right_clause.tape == right_clause.tape[::-1]:
                    continue
                if right_clause.surface in [prior.surface for prior in right_reverse]:
                    continue
                search(left_chain + [left_clause], right_reverse + [right_clause])
                if accepted is not None:
                    return

    search([], [])
    if accepted is None:
        raise AssertionError("fixed relation index did not reproduce an accepted path")

    left_chain, right_chain = accepted
    left_block = " ".join(clause.surface for clause in left_chain)
    right_block = " ".join(clause.surface for clause in right_chain)
    if normalize(left_block) != normalize(right_block)[::-1]:
        raise AssertionError("paired blocks are not character reversals")
    rendered = (
        parent_surface[:left_raw] + " " + left_block
        + parent_surface[left_raw:right_raw] + " " + right_block
        + parent_surface[right_raw:]
    )
    tape = normalize(rendered)
    audit = outside_in(tape)
    digest = hashlib.sha256(tape.encode("ascii")).hexdigest()
    if len(tape) != 672 or not audit["exact"] or digest != EXPECTED_CHILD_SHA256:
        raise AssertionError("replayed full rendering failed exactness or pinned digest")
    return {
        "rendered": rendered,
        "letters": len(tape),
        "sha256": digest,
        "exact": audit["exact"],
        "first_mismatch": audit["first_mismatch"],
        "states_examined": decoder.states_examined,
        "rejected_attempts": len(rejected),
        "rejections_by_type": {
            key: rejected.count(key) for key in sorted(set(rejected))
        },
        "accepted_paths": 1,
        "left_chain": [clause.relation for clause in left_chain],
        "right_chain": [clause.relation for clause in right_chain],
        "inserted_relations": [
            clause.relation for clause in left_chain + right_chain
        ],
        "residual_scope": "one complete clause at a time; no cross-clause buffer",
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> dict[str, object]:
    parent_payload = json.loads(PARENT_PATH.read_text())
    parent_row = next(
        row for row in parent_payload["rows"] if row.get("id") == PARENT_ID
    )
    parent_surface = str(parent_row["rendered"])
    parent_tape = normalize(parent_surface)
    parent_sha = hashlib.sha256(parent_tape.encode("ascii")).hexdigest()
    if parent_sha != PARENT_SHA256:
        raise AssertionError("pinned parent digest changed")

    snapshot = json.loads(INDEX_PATH.read_text())
    prior_relations = snapshot["relation_counts"]
    saved_run = json.loads(SAVED_RESULT_PATH.read_text())
    saved_row = next(
        row for row in saved_run["rows"]
        if row.get("id") == "discourse-linked-reverse-chain-672"
    )

    result = replay(parent_surface, prior_relations)
    if result["rendered"] != saved_row["rendered"]:
        raise AssertionError("replay surface differs from frozen saved result")
    if (result["states_examined"] != 9273
            or result["rejected_attempts"] != 35
            or result["accepted_paths"] != 1):
        raise AssertionError("replay counters differ from the saved run")

    compact = {
        "replay_id": "incumbent-672-fixed-index-replay",
        "method_scope": "deterministic replay of the saved bounded DFS using a fixed relation-count index; no repository history scan and no new novelty claim",
        "sources": {
            "parent_artifact": str(PARENT_PATH.relative_to(ROOT)),
            "parent_artifact_sha256": sha256_file(PARENT_PATH),
            "parent_normalized_sha256": parent_sha,
            "relation_index_artifact": str(INDEX_PATH.relative_to(ROOT)),
            "relation_index_file_sha256": sha256_file(INDEX_PATH),
            "relation_index_key_count": len(prior_relations),
            "relation_index_snapshot_id": snapshot["snapshot_id"],
            "generator_source": str(GENERATOR_PATH.relative_to(ROOT)),
            "generator_source_sha256": sha256_file(GENERATOR_PATH),
            "saved_result_artifact": str(SAVED_RESULT_PATH.relative_to(ROOT)),
            "saved_result_file_sha256": sha256_file(SAVED_RESULT_PATH),
        },
        "configuration": {
            "entities_in_order": list(ENTITIES),
            "predicates_in_order": list(PREDICATES),
            "normalized_seam_cuts": [LEFT_CUT, RIGHT_CUT],
            "chain_length": CHAIN_LENGTH,
            "first_success_stop": True,
            "novelty_gate": "relation counts from the supplied fixed index only",
        },
        "result": {key: value for key, value in result.items() if key != "rendered"},
        "rendered": result["rendered"],
        "saved_run_comparison": {
            "rendered_text_matches": True,
            "normalized_length_matches": result["letters"] == saved_row["audit"]["normalized_letters"],
            "saved_sha256": saved_row["audit"]["sha256_forward"],
        },
        "limitations": [
            "This is a deterministic replay on the saved parent and relation index, not a fresh novelty audit.",
            "The small proper-name and predicate inventory is engineered for reverse compatibility.",
            "Exactness and relation-chain continuity do not establish readability or coherent discourse.",
        ],
    }
    RESULT_PATH.write_text(json.dumps(compact, indent=2, ensure_ascii=False) + "\n")
    return compact


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, ensure_ascii=False))
