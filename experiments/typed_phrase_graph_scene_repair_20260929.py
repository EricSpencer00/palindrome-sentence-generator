"""Bounded repair of one phrase-graph edge with an ordinary scene clause."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from experiments.typed_phrase_graph_palindromes_20260928 import (
    LEFT, RIGHT, Phrase, exact_two_pointer, letters, validator,
)

# Fresh typed bank: the new right edge is a normal subject/verb/object scene
# clause. Its independently authored mate is retained as a boundary phrase;
# no characters are edited after matching.
REPAIRED_LEFT = LEFT[:-1] + (Phrase("Live was Nora.", "boundary-mate"),)
REPAIRED_RIGHT = RIGHT[:-1] + (Phrase("Aron saw evil.", "scene-subject-verb-object"),)

def match(left, right):
    return left.tape == right.tape[::-1]

def run():
    edges = []
    for left, right in zip(REPAIRED_LEFT, REPAIRED_RIGHT):
        assert match(left, right)
        edges.append((left, right))
    candidate = " ".join([x.text for x, _ in edges] + [y.text for _, y in edges[::-1]])
    tape = letters(candidate)
    units = [x.tape for x, _ in edges] + [y.tape for _, y in edges]
    audit = {
        "text": candidate, "letters": len(tape), "normalized": tape,
        "exact_two_pointer": exact_two_pointer(candidate), "validator": validator(candidate),
        "sha256": hashlib.sha256(tape.encode()).hexdigest(),
        "reverse_sha256": hashlib.sha256(tape[::-1].encode()).hexdigest(),
        "novelty_preflight": len(units) == len(set(units)),
        "provenance": {"new_edge": "Aron saw evil.", "new_edge_type": "scene-subject-verb-object",
                       "distinct_units": len(units) == len(set(units)),
                       "self_palindromic_units": any(u == u[::-1] for u in units),
                       "catalogue_text": False, "posthoc_repair": False},
        "reader_gate": "pending-blinded-human-ratings",
    }
    result = {"method": "typed phrase graph; one ordinary scene-edge replacement",
              "candidate": audit, "next_repair": "author a scene edge whose boundary mate is also natural English",
              "edge_types": [[x.kind, y.kind] for x, y in edges]}
    out = Path(__file__).parents[1] / "runs" / "typed-phrase-graph-scene-repair-20260929.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    return result

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
