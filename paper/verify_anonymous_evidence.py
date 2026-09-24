"""Verify the selected text and two constructive replays without Git or a model.

Run this file inside the extracted anonymous evidence archive with Python 3.10+.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from check_seam_invariant import exhaustive_algebra_audit
from replay_clause_search import replay


def letters(text: str) -> str:
    return re.sub("[^A-Za-z]", "", text).lower()


def raw_exact(text: str) -> bool:
    left, right = 0, len(text) - 1
    found = False
    while left <= right:
        while left <= right and not ("A" <= text[left] <= "Z" or "a" <= text[left] <= "z"):
            left += 1
        while left <= right and not ("A" <= text[right] <= "Z" or "a" <= text[right] <= "z"):
            right -= 1
        if left <= right:
            found = True
            if text[left].lower() != text[right].lower():
                return False
            left += 1
            right -= 1
    return found


def verify(directory: Path) -> dict[str, object]:
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, digest in manifest.items():
        if Path(name).name != name:
            raise ValueError("unexpected path in archive manifest")
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise AssertionError(f"File digest mismatch: {name}")
    data = json.loads((directory / "selected-results.json").read_text())
    rows = {row["id"]: row for row in data["results"]}
    if len(rows) != 11:
        raise AssertionError("Expected eleven selected examples")
    for row in rows.values():
        tape = letters(row["surface"])
        assert tape and tape == tape[::-1] and raw_exact(row["surface"])
        assert len(tape) == row["letters"]
        assert hashlib.sha256(tape.encode("ascii")).hexdigest() == row["normalized_sha256"]

    parent = rows["568-pinned"]["surface"]
    fixture = json.loads((directory / "seam-fixture.json").read_text())
    left, right = fixture["raw_cursors"]
    skipped = fixture["right_punctuation_skip"]
    assert not letters(parent[right:right + skipped])
    child = (parent[:left] + fixture["left_insert"] + parent[left:right]
             + fixture["right_insert"] + parent[right + skipped:])
    assert child == rows["630-god-dog"]["surface"] and raw_exact(child)

    relations = json.loads((directory / "relation-index.json").read_text())
    searched = replay(parent, relations)
    assert searched["rendered"] == rows["672-reverse-chain"]["surface"]
    assert searched["states_examined"] == 9273
    assert searched["rejected_attempts"] == 35
    assert searched["accepted_paths"] == 1
    algebra = exhaustive_algebra_audit()
    assert algebra["all_checks_passed"]
    return {
        "selected_exact_examples": len(rows),
        "seam_replay_letters": len(letters(child)),
        "clause_search_letters": searched["letters"],
        "clause_search_frontier_examinations": searched["states_examined"],
        "clause_search_rejected_attempts": searched["rejected_attempts"],
        "algebra_checks": algebra["checks_performed"],
        "human_readability_evidence": "not supplied; no human-study result is claimed",
    }


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2))
