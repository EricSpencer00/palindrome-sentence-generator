"""Human-authored scene lattice: semantic roles constrain the live tape product.

Each scene is written as a small valency frame before search.  The product then
matches characters while expanding the frame; it never reverses a completed
sentence.  A scene's role banks are deliberately tiny and semantically typed,
so this is a construction test, not a larger vocabulary sweep.
"""
from __future__ import annotations

import hashlib, json
from pathlib import Path
from experiments.full_sequence_grammar_product_20260917 import (
    CATALOGUE_FIXTURE, FUNCTION_WORDS, exact_audit, letters, search_pattern,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "scene-lattice-semantic-product-20260917.json"
EXPERIMENT_ID = "scene-lattice-semantic-product-20260917"
SIGNATURE = "human-authored-scene-lattice|typed-valency-banks|live-character-equations|independent-audit"

# Roles are chosen for one coherent scene, not selected by a language score.
# The same scene has a subject, transitive event, object, and response clause.
SCENES = {
    "letter_desk": {
        "description": "A writer sends a letter; a reader answers and keeps the page.",
        "slots": ("DET", "AGENT", "VERB_S", "DET", "OBJECT", "CONJ", "PRON", "VERB_BASE", "DET", "OBJECT", "PREP", "DET", "OBJECT"),
        "banks": {
            "AGENT": ("writer", "reader", "editor", "teacher", "artist"),
            "OBJECT": ("letter", "page", "note", "book", "message"),
        },
    },
    "garden_work": {
        "description": "A gardener tends a garden; we water the young plant by the wall.",
        "slots": ("DET", "AGENT", "VERB_S", "DET", "OBJECT", "CONJ", "PRON", "VERB_BASE", "DET", "OBJECT", "PREP", "DET", "OBJECT"),
        "banks": {
            "AGENT": ("gardener", "farmer", "worker", "teacher"),
            "OBJECT": ("garden", "plant", "flower", "tree", "seed", "wall"),
        },
    },
    "quiet_meal": {
        "description": "A baker makes a meal; she eats the bread at the table.",
        "slots": ("DET", "AGENT", "VERB_S", "DET", "OBJECT", "CONJ", "PRON", "VERB_BASE", "DET", "OBJECT", "PREP", "DET", "OBJECT"),
        "banks": {
            "AGENT": ("baker", "artist", "reader", "teacher"),
            "OBJECT": ("meal", "bread", "apple", "food", "table", "cake"),
        },
    },
}

BASE = {
    "DET": ("a", "the", "no", "one"), "VERB_S": ("makes", "marks", "keeps", "tends", "bakes", "teaches"),
    "CONJ": ("and", "but"), "PRON": ("i", "we", "she", "he", "they"),
    "VERB_BASE": ("read", "keep", "eat", "water", "tend", "mark", "teach", "bake"),
    "PREP": ("at", "by", "in", "on", "near", "with"),
}

def audit_candidate(rendered: str, words: tuple[str, ...]) -> dict:
    tape = letters(rendered)
    independent = tape == tape[::-1] and len(tape) > 0
    return {"rendered": rendered, "words": words, "audit": exact_audit(rendered),
            "independent_recheck": independent,
            "anti_shortcut": {"repeated_content": len([w for w in words if w not in FUNCTION_WORDS]) != len(set(w for w in words if w not in FUNCTION_WORDS)),
                              "self_palindromic_words": [w for w in words if len(w)>1 and w==w[::-1]],
                              "catalogue_overlap": sorted(set(words) & (set(CATALOGUE_FIXTURE)-FUNCTION_WORDS))},
            "provenance": "seedless authored scene frame; role bank selected before character search"}

def run() -> dict:
    reports = {}
    for name, scene in SCENES.items():
        banks = dict(BASE)
        banks.update(scene["banks"])
        result = search_pattern(scene["slots"], state_budget=120_000, banks=banks,
                                forbidden_words=frozenset(CATALOGUE_FIXTURE))
        reports[name] = {"description": scene["description"], "slots": scene["slots"],
                         "states": result.states, "mismatch_edges": result.mismatch_edges,
                         "budget_exhausted": result.budget_exhausted,
                         "exact_candidates": [audit_candidate(p["rendered"], p["words"]) for p in result.paths],
                         "first_mismatch_frontiers": result.mismatch_frontiers[:8],
                         "next_repair": "replace the first mismatching semantic role with a new valency-compatible role bank, preserving the matched prefix/suffix and resuming inward"}
    return {"experiment_id": EXPERIMENT_ID, "signature": SIGNATURE,
            "method": "scene authored first; typed valency banks and live outer-character equations searched together",
            "scenes": reports, "reader_gate": "closed unless an exact candidate survives independent audit",
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "anti_shortcut_policy": ["no finished-tape reversal", "no repeated content", "no catalogue admission", "programmatic scores do not certify readability"]}

if __name__ == "__main__":
    OUT.write_text(json.dumps(run(), indent=2) + "\n")
    data = json.loads(OUT.read_text())
    print(json.dumps({"status": "exact" if any(s["exact_candidates"] for s in data["scenes"].values()) else "no_exact_closure",
                      "scenes": {k: (v["states"], len(v["exact_candidates"])) for k,v in data["scenes"].items()}}))
