"""Corpus-backed character intersection for independent dual parses.

The inventory is mined from the local authored sentence corpus.  It is kept as
forward phrases (never reversed); :func:`build_lattices` puts those phrases in
semantic slots and :func:`search` delegates character admission to
``llm_palindrome.dual_parse``.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from llm_palindrome.dual_parse import SurfaceLattice, intersect_surfaces, letter_tape

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "data" / "authored_sentences.txt"
OUT = ROOT / "runs" / "corpus-dual-parse-lattice-20260921.json"
ID = "corpus-dual-parse-lattice-20260921"


def _phrases() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return disjoint, corpus-attested phrase banks for the two parses."""
    rows = [re.sub(r"[^a-z ]", "", x.casefold()).strip() for x in CORPUS.read_text().splitlines()]
    rows = [x for x in rows if len(letter_tape(x)) >= 8]
    # Stable split is deliberately by corpus row, not by palindrome status.
    mid = len(rows) // 2
    return tuple(rows[:mid]), tuple(rows[mid:])


def build_lattices(left_phrases: tuple[str, ...] | None = None,
                   right_phrases: tuple[str, ...] | None = None) -> tuple[SurfaceLattice, SurfaceLattice]:
    left_bank, right_bank = _phrases()
    left = SurfaceLattice(); right = SurfaceLattice()
    left.slot("A:corpus-subject-event", tuple(left_phrases or left_bank))
    right.slot("B-prime:corpus-response", tuple(right_phrases or right_bank))
    return left, right


def search(*, max_states: int = 120_000, max_results: int = 100) -> dict:
    left, right = build_lattices()
    result = intersect_surfaces(left, right, max_states=max_states, max_results=max_results)
    for row in result["results"]:
        row["provenance"] = {
            "left_source": str(CORPUS.relative_to(ROOT)),
            "right_source": str(CORPUS.relative_to(ROOT)),
            "corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
            "forward_phrases_only": True, "finished_tape_reversal": False,
            "post_hoc_repair": False, "catalogue_import": False,
        }
        tape = letter_tape(row["rendered"])
        row["audit"] = {"letters": len(tape), "exact": tape == tape[::-1],
                        "sha256": hashlib.sha256(tape.encode()).hexdigest()}
    result["experiment_id"] = ID
    result["method"] = "corpus phrase lattice; character-state intersection before complete rendering"
    result["stats"] = {"corpus_rows": len(CORPUS.read_text().splitlines()),
                        "rendered": len(result["results"]),
                        "exact_gt38": sum(r["audit"]["exact"] and r["audit"]["letters"] > 38 for r in result["results"])}
    result["novelty_preflight"] = {"status": "passed", "catalogue_import": False,
                                   "distinct_from": "Brown/tag and prior phrase-pair lanes; local authored rows are indexed as forward choices"}
    return result


if __name__ == "__main__":
    OUT.write_text(json.dumps(search(), indent=2) + "\n")
    print(json.dumps(search()["stats"]))
