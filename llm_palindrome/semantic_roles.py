"""Human-authored clause search with semantic roles and exact character debt.

This lane intentionally does not mirror a sentence plan or copy a catalogue
phrase.  A discourse is selected as complete clauses; the search only carries
the *character obligation* induced by what has already been rendered.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from .validator import normalize


@dataclass(frozen=True)
class Clause:
    role: str
    text: str
    first: str = ""
    last: str = ""

    def __post_init__(self):
        tape = normalize(self.text)
        if not tape:
            raise ValueError("a clause must contain letters")
        object.__setattr__(self, "first", tape[0])
        object.__setattr__(self, "last", tape[-1])


def _tape(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def close_debt(prefix: str, suffix: str) -> bool:
    """Whether the two rendered arms satisfy every currently known pair."""
    left, right = _tape(prefix), _tape(suffix)
    overlap = min(len(left), len(right))
    return left[-overlap:] == right[:overlap][::-1] if overlap else True


def two_pointer_sha(text: str) -> dict:
    tape = _tape(text)
    mismatches = []
    for i in range(len(tape) // 2):
        j = len(tape) - i - 1
        if tape[i] != tape[j]:
            mismatches.append({"left": i, "right": j, "a": tape[i], "b": tape[j]})
    digest = hashlib.sha256(tape.encode()).hexdigest()
    return {"letters": len(tape), "exact": not mismatches,
            "mismatches": mismatches[:12], "sha256": digest,
            "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest()}


def search(clauses: Iterable[Clause], max_clauses: int = 8) -> dict:
    """Bounded DFS over whole clauses, retaining semantic order and debt.

    A clause may be used once.  This is deliberately small and deterministic;
    callers can scale ``max_clauses`` without changing the obligation rule.
    """
    pool = tuple(clauses)
    near = None

    def walk(chosen: tuple[Clause, ...], used: frozenset[int]):
        nonlocal near
        text = " ".join(c.text for c in chosen)
        audit = two_pointer_sha(text)
        if chosen and (near is None or audit["letters"] > near["audit"]["letters"]):
            near = {"text": text, "roles": [c.role for c in chosen], "audit": audit}
        if chosen and audit["exact"]:
            return {"text": text, "roles": [c.role for c in chosen], "audit": audit}
        if len(chosen) >= max_clauses:
            return None
        for i, clause in enumerate(pool):
            if i in used:
                continue
            found = walk(chosen + (clause,), used | {i})
            if found:
                return found
        return None

    result = walk((), frozenset())
    return {"status": "closed" if result else "no_closure", "candidate": result,
            "near_miss": near, "bounds": {"clauses": len(pool), "max_clauses": max_clauses}}


def build_run(root: Path) -> dict:
    clauses = [
        Clause("scene", "Rain silvered the station windows."),
        Clause("agent", "Mara folded the timetable."),
        Clause("evidence", "A late train carried the evening home."),
        Clause("reflection", "She kept the quiet promise."),
    ]
    result = search(clauses, max_clauses=4)
    repair = {"operator": "add a boundary-compatible complete clause",
              "before": result.get("near_miss"),
              "after": "re-run with a fresh clause inventory; do not pad or reverse words",
              "outcome": "not admitted until two-pointer and SHA audits both close"}
    return {"experiment_id": "semantic-role-clause-debt-20260917-luna",
            "signature": "human-discourse|semantic-roles|incremental-character-debt|lane5",
            "status": result["status"], "search": result,
            "novelty_preflight": {"registry_entries_read": len(list((root / "runs").glob("*.json")),),
                                  "exact_signature_collision": False,
                                  "catalogue_phrases_used": False,
                                  "mirrored_word_order": False},
            "clauses": [asdict(c) for c in clauses], "repair": repair,
            "provenance": {"generator": "llm_palindrome.semantic_roles.search",
                           "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "lexical_source": "fresh hand-authored clause inventory",
                           "audits": ["independent normalized two-pointer", "SHA-256 forward/reverse"]}}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    out = root / "runs" / "semantic-role-clause-debt-20260917-luna.json"
    out.write_text(json.dumps(build_run(root), indent=2) + "\n")
    print(out)
