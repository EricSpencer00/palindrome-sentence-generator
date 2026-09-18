"""Frame-first, hashed class-centre successor.

This successor starts from an ordinary connected sentence frame, independently
types every lexical candidate, and only then invokes the one-character full
tree scheduler.  The prior artifacts are protected by an exact base-source
hash; no multiword center or catalogue text is used.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from experiments import fresh_internal_center_inventory_20260913 as base

BASE_SOURCE_SHA256 = "6614ae113e86fda1472b3f7cefbfdcddfebc56262c1e4c9d5a874ac7472c287b"


def isolated_bindings():
    base_path = Path(base.__file__).resolve()
    actual = sha256(base_path.read_bytes()).hexdigest()
    if actual != BASE_SOURCE_SHA256:
        raise RuntimeError(f"base source changed: expected {BASE_SOURCE_SHA256}, found {actual}")
    old_by_kind, old_frame, old_pivots = base.BY_KIND, base.FRAME, base.PIVOTS
    by_kind = {kind: tuple(items) for kind, items in old_by_kind.items()}
    by_kind["noun_center"] = by_kind["noun_center"] + (base.Lexeme("class", "noun_center", "class"),)
    by_kind["adj_center"] = by_kind["adj_center"] + (base.Lexeme("small", "adj_center", "class"),)
    by_kind["verb_intro"] = by_kind["verb_intro"] + (base.Lexeme("held", "verb_intro", "teaching", "person", "class"),)
    by_kind["prep"] = by_kind["prep"] + (base.Lexeme("along", "prep", "route_relation"),)
    by_kind["noun_event"] = by_kind["noun_event"] + (base.Lexeme("route", "noun_event", "place"),)
    base.BY_KIND = by_kind
    base.FRAME = {"class": {"verb": "held", "adjectives": {"small"}, "relations": {"along"}}}
    base.PIVOTS = {"class": (4,)}
    return old_by_kind, old_frame, old_pivots


def run(*, state_limit: int = 100_000, closure_limit: int = 100) -> dict[str, object]:
    old_by_kind, old_frame, old_pivots = isolated_bindings()
    try:
        grammar = base.Grammar(); leaf_slots = base.slots(grammar.expand(base.Symbol("S")))
        stats = Counter(state_count=0, search_states=0, lexical_assignments_considered=0,
                        character_emissions=0, residual_cancellations=0,
                        residual_contradictions=0, complete_tree_states=0,
                        exact_closures=0, admitted_closures=0)
        sentence_frames = [{"center": "class", "pivot": "clas|s", "verb": "held", "adjective": "small", "relation": "along", "surface": "held a small class along the western route", "semantic_status": "ordinary classroom/event frame"}]
        inventory = base.centre_inventory(stats); boundaries = base.boundary_inventory(stats)
        exact, admitted = base.search(grammar, leaf_slots, state_limit=state_limit,
                                       closure_limit=closure_limit, stats=stats)
        controls = [
            base.audit(grammar, "The careful researcher held a small class along the western route. The careful researcher documented the annual report after the public study today.", "shared_referent_two_sentence_control", tuple(re.findall(r"[a-z]+", "the careful researcher held a small class along the western route the careful researcher documented the annual report after the public study today"))),
            base.audit(grammar, "The patient analyst held a small class along the public route. The patient analyst reviewed the public record after the western study today.", "shared_referent_two_sentence_control", tuple(re.findall(r"[a-z]+", "the patient analyst held a small class along the public route the patient analyst reviewed the public record after the western study today"))),
        ]
        deepest = {"ledger_before_rejection": stats.get("deepest_live_ledger", []), "next_literal_rejection": stats.get("deepest_next_literal_rejection"), "independent_replay": stats.get("deepest_independent_replay"), "emissions_including_rejection": stats.get("deepest_ledger_length", 0)}
        return {"status": "class_frame_hashed_variant_typed_tree_search", "config": {"state_limit": state_limit, "closure_limit": closure_limit, "one_connected_tree": True, "grammar_owns_every_leaf": True, "frame_first_inventory": True, "word_internal_center_inventory": True, "prepared_multiword_center": False, "joint_boundary_enumeration_before_search": True, "ordinary_class_frame": True, "prior_sources_hash_enforced": True, "one_character_emission_states": True, "replayed_ledger": True, "independent_complete_reparse": True, "reject_every_self_palindromic_contiguous_multiword_span": True, "corpus_or_catalogue_generation": False}, "grammar_leaf_count": len(leaf_slots), "sentence_frame_inventory": sentence_frames, "centre_inventory": inventory, "boundary_inventory": boundaries, "best_boundary_trace": boundaries[0] if boundaries else None, "deepest_full_scheduler_replay": deepest, "stats": dict(stats), "exact_closures": exact, "admitted_closures": admitted, "complete_grammar_controls": controls, "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "base_generator": str(Path(base.__file__).resolve()), "base_generator_sha256": BASE_SOURCE_SHA256, "grammar_sha256": grammar.digest(), "material": "task-authored ordinary class sentence frame and shared-referent discourse grammar; no catalogue text"}, "reader_facing_next_operator": "Derive any successor from this variant's persisted deepest scheduler ledger; do not mutate the hashed base or prepare a center phrase.", "scope": "This is a bounded construction run; exactness and feature parsing do not certify human readability."}
    finally:
        base.BY_KIND, base.FRAME, base.PIVOTS = old_by_kind, old_frame, old_pivots


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, required=True); p.add_argument("--state-limit", type=int, default=100_000); p.add_argument("--closure-limit", type=int, default=100); args = p.parse_args()
    if args.out.exists(): p.error(f"refusing to overwrite {args.out}")
    result = run(state_limit=args.state_limit, closure_limit=args.closure_limit); args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps({"out": str(args.out), "states": result["stats"]["state_count"], "exact": len(result["exact_closures"]), "admitted": len(result["admitted_closures"])}, indent=2))


if __name__ == "__main__": main()
