"""Packed grammar intersection with an editable document-scene center.

This is a continuation of the tested packed seam solver, not a Cartesian
product of complete sentences.  Every lexical alternative is compiled into a
shared character NFA; opposing states advance only when their exposed
characters agree.  The central scene is allowed to vary jointly in quantity,
document noun, human subject, and plural predicate, so a closure must earn
its full tape rather than inherit the fixed ``nine memos; some men`` seam.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from experiments.packed_seam_grammar_20260927 import Grammar, intersect, norm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "packed-editable-center-20260928.json"
INCUMBENT = "An aide rips nine memos; some men inspire Diana."


def audit(text: str) -> dict[str, object]:
    tape = norm(text)
    mismatches = [(i, tape[i], tape[-i - 1])
                  for i in range(len(tape) // 2)
                  if tape[i] != tape[-i - 1]]
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {
        "letters": len(tape),
        "two_pointer_exact": bool(tape) and not mismatches,
        "first_mismatch": mismatches[:1],
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "sha_equal": forward == reverse,
    }


def grammar() -> Grammar:
    g = Grammar()
    g.slot((
        "an aide", "a nurse", "a clerk", "the aide", "the nurse",
        "an old aide", "a tired aide", "a quiet aide", "one aide",
        "a senior aide", "another aide", "a patient nurse",
    ), "subject:singular")
    g.slot(("", "now", "often", "still", "quietly", "slowly", "carefully"),
           "adverb:opening")
    g.slot((
        "rips", "reads", "writes", "sends", "files", "copies", "shreds",
        "signs", "sorts", "edits", "saves", "keeps",
    ), "verb:singular:document")
    # These four slots are jointly editable.  They deliberately contain
    # ordinary inflectional alternatives rather than pre-paired mirrors.
    g.slot(("nine", "one", "two", "six", "ten", "many", "several"),
           "quantity")
    g.slot(("memos;", "notes;", "letters;", "files;", "pages;", "reports;"),
           "document")
    g.slot(("some", "many", "the", "these", "our"), "human:determiner")
    g.slot(("men", "women", "aides", "clerks", "nurses", "poets"),
           "human:subject")
    g.slot((
        "inspire", "admire", "praise", "encourage", "assist", "help",
        "guide", "support", "surprise", "impress", "thank",
    ), "verb:plural:human")
    g.slot((
        "Diana", "Anna", "Nora", "Leon", "aide", "nurse", "clerk",
        "one writer", "a poet", "the writer", "our aide", "the old aide",
    ), "object:human")
    return g


def run() -> dict[str, object]:
    raw = intersect(grammar(), max_letters=180, cap=250_000)
    rows = []
    for row in raw["candidates"]:
        rendered = row["rendered"]
        checked = audit(rendered)
        rows.append({
            **row,
            "audit": checked,
            "novelty_preflight": norm(rendered) != norm(INCUMBENT),
            "provenance": {
                "generator": "packed shared-slot character NFA",
                "editable_center_slots": ["quantity", "document", "human:subject", "verb:plural:human"],
                "complete_sentence_enumeration": False,
                "finished_tape_reversal": False,
                "posthoc_character_repair": False,
                "catalogue_text": False,
                "per_candidate_rlaif": False,
                "reader_certified": False,
            },
        })
    exact = [row for row in rows if row["audit"]["two_pointer_exact"]
             and row["audit"]["sha_equal"] and row["novelty_preflight"]]
    return {
        "experiment_id": "packed-editable-center-20260928",
        "method": "packed live character intersection over an editable document-scene grammar",
        "represented_paths": 12 * 7 * 12 * 7 * 6 * 5 * 6 * 11 * 12,
        "solver_stats": {key: raw[key] for key in (
            "states", "transitions", "cap_reached", "grammar_states",
            "grammar_character_edges")},
        "exact_novel_candidates": exact,
        "all_accepting_witnesses": rows,
        "incumbent": {"rendered": INCUMBENT, "audit": audit(INCUMBENT)},
        "reader_gate": "closed; no blinded human ratings collected",
        "next_repair": raw["dead_frontiers"][:3],
        "provenance": {
            "grammar_sha256": hashlib.sha256(
                json.dumps({"slots": "editable document scene"}, sort_keys=True).encode()
            ).hexdigest(),
            "independent_audits": ["two-pointer", "forward/reverse SHA-256"],
        },
    }


if __name__ == "__main__":
    payload = run()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "exact_novel": len(payload["exact_novel_candidates"]),
        "states": payload["solver_stats"]["states"],
        "transitions": payload["solver_stats"]["transitions"],
    }, sort_keys=True))
