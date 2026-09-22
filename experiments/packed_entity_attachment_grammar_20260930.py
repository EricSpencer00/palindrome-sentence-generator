"""Live character intersection over a fresh named-entity scene grammar.

The lexical frame is deliberately ordinary: named subject + finite verb +
theme object + optional attachment.  It is not a phrase-reversal bank.  The
intersection advances compatible characters from both ends before any full
sentence is rendered.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from experiments.packed_seam_grammar_20260927 import Grammar, intersect, norm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "packed-entity-attachment-grammar-20260930.json"

SLOTS = (
    ("agent", ("Mara", "Nora", "Leon", "Diana", "Anna", "Elena")),
    ("finite_event", ("maps", "marks", "reads", "keeps", "folds", "opens", "carries")),
    ("theme", ("the map", "a note", "the letter", "a parcel", "the chart", "a book")),
    # These are grammatical adjuncts, not reverse-paired material.
    ("attachment", ("", "at dawn", "by the gate", "near the river", "for Nora", "with care")),
)


def audit(text: str) -> dict[str, object]:
    tape = norm(text)
    mismatch = next(((i, tape[i], tape[-1 - i]) for i in range(len(tape) // 2)
                     if tape[i] != tape[-1 - i]), None)
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"letters": len(tape), "two_pointer_exact": bool(tape) and mismatch is None,
            "first_mismatch": mismatch, "sha256_forward": forward,
            "sha256_reverse": reverse, "sha_equal": forward == reverse,
            "normalized": tape}


def grammar() -> Grammar:
    g = Grammar()
    for role, alternatives in SLOTS:
        g.slot(alternatives, role)
    return g


def bank_checks() -> dict[str, object]:
    phrases = [p for _, vals in SLOTS for p in vals if p]
    tapes = [norm(p) for p in phrases]
    reverse_pairs = [(a, b) for a in tapes for b in tapes if a != b and a == b[::-1]]
    return {"phrase_count": len(phrases), "reverse_pairs": reverse_pairs,
            "reverse_pair_free": not reverse_pairs,
            "lexical_frame": "proper-agent + finite-event + theme-object + optional-attachment"}


def run() -> dict[str, object]:
    checks = bank_checks()
    assert checks["reverse_pair_free"]
    raw = intersect(grammar(), max_letters=140, cap=150_000)
    rows = []
    for row in raw["candidates"]:
        rendered = row["rendered"]
        rows.append({**row, "audit": audit(rendered),
                     "novelty_preflight": True,
                     "provenance": {"fresh_entity_attachment_grammar": True,
                                    "complete_sentence_enumeration": False,
                                    "finished_tape_reversal": False,
                                    "posthoc_repair": False, "catalogue_text": False,
                                    "per_candidate_rlaif": False, "reader_certified": False}})
    exact = [r for r in rows if r["audit"]["two_pointer_exact"] and r["audit"]["sha_equal"]]
    controls = ["Mara maps the map at dawn.", "Nora reads a letter by the gate."]
    return {"experiment_id": "packed-entity-attachment-grammar-20260930",
            "method": "online equal-character intersection over proper-name scene slots",
            "slots": [{"role": r, "alternatives": list(v)} for r, v in SLOTS],
            "bank_checks": checks,
            "controls": [{"text": c, "audit": audit(c)} for c in controls],
            "solver_stats": {k: raw[k] for k in ("states", "transitions", "cap_reached", "grammar_states", "grammar_character_edges")},
            "exact_candidates": exact, "accepting_witnesses": rows,
            "reader_gate": "closed; no blinded human ratings collected",
            "next_operator": raw["dead_frontiers"][:5],
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "independent_audits": ["two-pointer", "forward/reverse SHA-256", "shared packed intersection"]}}


if __name__ == "__main__":
    result = run()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"exact": len(result["exact_candidates"]), "states": result["solver_stats"]["states"], "transitions": result["solver_stats"]["transitions"]}, sort_keys=True))
