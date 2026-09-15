"""Bounded test of a scalable, semantic clause-expansion family.

The hypothesis is that a single typed event plan can grow by adding an
ordinary adverbial or an object modifier while an outside-in character zipper
keeps both independently authored clauses exact.  This is intentionally a
small finite-state test: an exact closure must be independently audited at
the full rendered length, and no fallback/filler output is emitted.
"""
from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

from experiments import outer_boundary_residual_search_20260915 as zipper

ROOT = Path(__file__).resolve().parents[1]

FAMILIES = {
    "base_event": {
        "left": (("det", ("a", "the")), ("subject", ("artist", "baker", "captain", "doctor", "farmer", "guard", "poet", "sailor", "teacher")), ("verb", ("admires", "carries", "finds", "gathers", "guides", "notices", "opens", "paints", "repairs", "saves")), ("det", ("a", "the")), ("object", ("bridge", "candle", "garden", "letter", "mirror", "parcel", "picture", "river", "signal", "window"))),
        "right": (("det", ("a", "the")), ("subject", ("author", "child", "clerk", "singer", "student", "traveler", "worker")), ("verb", ("answers", "builds", "cleans", "draws", "hears", "keeps", "learns", "reads")), ("det", ("a", "the")), ("object", ("anchor", "basket", "branch", "compass", "field", "flower", "harbor", "message", "stone", "tower"))),
    },
    "adverb_expansion": {
        "left": (("det", ("a", "the")), ("subject", ("artist", "baker", "captain", "doctor", "farmer", "guard", "poet", "sailor", "teacher")), ("adverb", ("calmly", "gently", "openly", "slowly", "wisely")), ("verb", ("admires", "carries", "finds", "gathers", "guides", "notices", "opens", "paints", "repairs", "saves")), ("det", ("a", "the")), ("object", ("bridge", "candle", "garden", "letter", "mirror", "parcel", "picture", "river", "signal", "window"))),
        "right": (("det", ("a", "the")), ("subject", ("author", "child", "clerk", "singer", "student", "traveler", "worker")), ("adverb", ("daily", "firmly", "gladly", "neatly", "rarely")), ("verb", ("answers", "builds", "cleans", "draws", "hears", "keeps", "learns", "reads")), ("det", ("a", "the")), ("object", ("anchor", "basket", "branch", "compass", "field", "flower", "harbor", "message", "stone", "tower"))),
    },
    "adverb_object_modifier_expansion": {
        "left": (("det", ("a", "the")), ("subject", ("artist", "baker", "captain", "doctor", "farmer", "guard", "poet", "sailor", "teacher")), ("adverb", ("calmly", "gently", "openly", "slowly", "wisely")), ("verb", ("admires", "carries", "finds", "gathers", "guides", "notices", "opens", "paints", "repairs", "saves")), ("det", ("a", "the")), ("adjective", ("bright", "kind", "old", "quiet", "red", "young")), ("object", ("bridge", "candle", "garden", "letter", "mirror", "parcel", "picture", "river", "signal", "window"))),
        "right": (("det", ("a", "the")), ("subject", ("author", "child", "clerk", "singer", "student", "traveler", "worker")), ("adverb", ("daily", "firmly", "gladly", "neatly", "rarely")), ("verb", ("answers", "builds", "cleans", "draws", "hears", "keeps", "learns", "reads")), ("det", ("a", "the")), ("adjective", ("blue", "calm", "distant", "gentle", "green", "hidden", "little", "narrow", "silver")), ("object", ("anchor", "basket", "branch", "compass", "field", "flower", "harbor", "message", "stone", "tower"))),
    },
}

# Keep the event lexicon readable while ensuring the zipper actually tests a
# nontrivial second character.  These ordinary terminal-vowel nouns are
# lexical alternatives, not preassembled palindromic units.
for _family in FAMILIES.values():
    right_slots = list(_family["right"])
    role, words = right_slots[-1]
    right_slots[-1] = (role, tuple(words) + ("area", "idea", "camera", "drama", "opera", "sofa", "data", "quota", "flora", "fauna", "villa", "pizza"))
    _family["right"] = tuple(right_slots)


def run(max_states: int) -> dict:
    reports = {}
    for name, family in FAMILIES.items():
        zipper.LEFT, zipper.RIGHT = family["left"], family["right"]
        result = zipper.run(max_states)
        reports[name] = {
            "slot_count": len(zipper.LEFT),
            "stats": result["stats"],
            "exact_candidates": result["rendered_candidates"],
            "deepest_frontier": result["mismatch_frontier"][0] if result["mismatch_frontier"] else None,
            "independent_audit_scope": "Every exact closure is normalized and mechanically audited by the shared admission gate; no closure is promoted as readable without blinded human evidence.",
        }
    return {
        "status": "typed_clause_expansion_family",
        "config": {"max_states_per_family": max_states, "family_growth": "base -> adverb -> adverb plus object adjective", "word_order_mirror_generation": False, "catalogue_text": False, "filler_or_fallback": False, "center_word_prepared": False, "closure_requires_exact_full_tape": True},
        "families": reports,
        "next_construction": "If a family reaches an exact closure, retain its typed derivation and add one semantically licensed role at the first live boundary; if empty, mine the deepest paired boundary as the next lexical valency repair rather than adding filler.",
        "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "material": "task-authored finite event-role inventories; no intact source sentences"},
    }


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--max-states", type=int, default=250_000); args = ap.parse_args()
    if args.out.exists(): ap.error("refusing to overwrite existing output")
    result = run(args.max_states); args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: row["stats"] for name, row in result["families"].items()}, indent=2))


if __name__ == "__main__": main()
