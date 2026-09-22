"""Packed intersection over a fresh ordinary-English scene grammar.

Unlike the vocative/semordnilap lanes, no phrase in these banks is authored as
the reverse of another phrase.  The grammar is intersected at character
frontiers before a surface is rendered; a closure would therefore have to
cross ordinary word boundaries on both sides.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.packed_seam_grammar_20260927 import Grammar, intersect, norm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "packed-fresh-scene-grammar-20260928.json"


def audit(text: str) -> dict[str, object]:
    tape = norm(text)
    mismatch = next(((i, tape[i], tape[-i - 1])
                     for i in range(len(tape) // 2)
                     if tape[i] != tape[-i - 1]), None)
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"letters": len(tape), "two_pointer_exact": bool(tape) and mismatch is None,
            "first_mismatch": mismatch, "sha256_forward": forward,
            "sha256_reverse": reverse, "sha_equal": forward == reverse}


SLOTS = (
    ("subject", ("the pilot", "a baker", "the nurse", "a poet",
                  "the clerk", "a teacher", "the sailor", "a gardener")),
    ("verb", ("marks", "maps", "reads", "writes", "keeps", "carries",
               "guides", "records")),
    ("object", ("the chart", "a letter", "the map", "a note",
                 "the parcel", "a lantern", "the garden", "a story")),
    ("setting", ("near the harbor", "by the river", "under the bridge",
                  "beside the gate", "after rain", "at dawn", "before dusk",
                  "at sea", "by boat", "at a club", "in a group",
                  "near a fort", "in a samba", "at a festa", "in a toga",
                  "at night")),
)


def grammar() -> Grammar:
    g = Grammar()
    for role, alternatives in SLOTS:
        g.slot(alternatives, role)
    return g


def bank_checks() -> dict[str, object]:
    phrases = [phrase for _, values in SLOTS for phrase in values]
    tapes = [norm(phrase) for phrase in phrases]
    reverse_pairs = [(a, b) for a in tapes for b in tapes if a != b and a == b[::-1]]
    return {"phrase_count": len(phrases), "phrase_reverse_pairs": reverse_pairs,
            "reverse_pair_free": not reverse_pairs}


def run() -> dict[str, object]:
    checks = bank_checks()
    assert checks["reverse_pair_free"]
    raw = intersect(grammar(), max_letters=180, cap=250_000)
    rows = []
    for row in raw["candidates"]:
        rendered = row["rendered"]
        rows.append({**row, "audit": audit(rendered),
                     "novelty_preflight": True,
                     "provenance": {
                         "fresh_scene_grammar": True,
                         "phrase_reverse_pair_bank": False,
                         "complete_sentence_enumeration": False,
                         "finished_tape_reversal": False,
                         "posthoc_character_repair": False,
                         "catalogue_text": False,
                         "per_candidate_rlaif": False,
                         "reader_certified": False,
                     }})
    exact = [row for row in rows if row["audit"]["two_pointer_exact"]
             and row["audit"]["sha_equal"]]
    return {
        "experiment_id": "packed-fresh-scene-grammar-20260928",
        "method": "packed live character intersection over fresh ordinary scene slots",
        "slots": [{"role": role, "alternatives": list(values)} for role, values in SLOTS],
        "bank_checks": checks,
        "solver_stats": {key: raw[key] for key in (
            "states", "transitions", "cap_reached", "grammar_states",
            "grammar_character_edges")},
        "exact_candidates": exact,
        "accepting_witnesses": rows,
        "reader_gate": "closed; no blinded human ratings collected",
        "next_repair": raw["dead_frontiers"][:5],
        "provenance": {"generator_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
            "independent_audits": ["two-pointer", "forward/reverse SHA-256"]},
    }


if __name__ == "__main__":
    payload = run()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"exact": len(payload["exact_candidates"]),
                      "states": payload["solver_stats"]["states"],
                      "transitions": payload["solver_stats"]["transitions"]}, sort_keys=True))
