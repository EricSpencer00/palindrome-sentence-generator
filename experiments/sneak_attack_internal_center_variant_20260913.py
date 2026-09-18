"""Immutable sneak-attack variant of the non-career internal-centre search.

The preceding non-career artifact remains unchanged.  This method variant
isolates a stronger ordinary collocation (``described a sneak attack``) and
executes the existing connected-tree character scheduler with a fresh
lexical/frame binding.  It does not install a multiword center.
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


def isolated_bindings() -> None:
    """Bind the stronger frame only in this process; prior source is untouched."""
    base.BY_KIND = {kind: tuple(items) for kind, items in base.BY_KIND.items()}
    base.BY_KIND["adj_center"] = base.BY_KIND["adj_center"] + (base.Lexeme("sneak", "adj_center", "attack"),)
    base.FRAME = {
        "attack": {"verb": "described", "adjectives": {"sneak"}, "relations": {"during"}},
        "effect": {"verb": "measured", "adjectives": {"visible"}, "relations": {"during"}},
    }


def run(*, state_limit: int = 100_000, closure_limit: int = 100) -> dict[str, object]:
    old_by_kind, old_frame = base.BY_KIND, base.FRAME
    isolated_bindings()
    grammar = base.Grammar(); leaf_slots = base.slots(grammar.expand(base.Symbol("S")))
    stats = Counter(state_count=0, search_states=0, lexical_assignments_considered=0,
                    character_emissions=0, residual_cancellations=0,
                    residual_contradictions=0, complete_tree_states=0,
                    exact_closures=0, admitted_closures=0)
    inventory = base.centre_inventory(stats); boundaries = base.boundary_inventory(stats)
    best = boundaries[0] if boundaries else None
    exact, admitted = base.search(grammar, leaf_slots, state_limit=state_limit,
                                   closure_limit=closure_limit, stats=stats)
    controls = [
        base.audit(grammar, "The careful researcher described a sneak attack during the annual trial. The careful researcher documented the annual report after the public study today.", "shared_referent_two_sentence_control", tuple(re.findall(r"[a-z]+", "the careful researcher described a sneak attack during the annual trial the careful researcher documented the annual report after the public study today"))),
        base.audit(grammar, "The patient analyst measured a visible effect during the annual trial. The patient analyst reviewed the public record after the western study today.", "shared_referent_two_sentence_control", tuple(re.findall(r"[a-z]+", "the patient analyst measured a visible effect during the annual trial the patient analyst reviewed the public record after the western study today"))),
    ]
    deepest = {"ledger_before_rejection": stats.get("deepest_live_ledger", []),
               "next_literal_rejection": stats.get("deepest_next_literal_rejection"),
               "independent_replay": stats.get("deepest_independent_replay"),
               "emissions_including_rejection": stats.get("deepest_ledger_length", 0)}
    result = {
        "status": "sneak_attack_internal_center_variant_typed_tree_search",
        "config": {"state_limit": state_limit, "closure_limit": closure_limit,
                    "one_connected_tree": True, "grammar_owns_every_leaf": True,
                    "word_internal_center_inventory": True, "prepared_multiword_center": False,
                    "joint_boundary_enumeration_before_search": True,
                    "ordinary_sneak_attack_frame": True, "prior_career_artifact_untouched": True,
                    "one_character_emission_states": True, "replayed_ledger": True,
                    "independent_complete_reparse": True,
                    "reject_every_self_palindromic_contiguous_multiword_span": True,
                    "corpus_or_catalogue_generation": False},
        "grammar_leaf_count": len(leaf_slots), "centre_inventory": inventory,
        "boundary_inventory": boundaries, "best_boundary_trace": best,
        "deepest_full_scheduler_replay": deepest, "stats": dict(stats),
        "exact_closures": exact, "admitted_closures": admitted,
        "complete_grammar_controls": controls,
        "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                       "base_generator": str(Path(base.__file__).resolve()),
                       "grammar_sha256": grammar.digest(),
                       "material": "task-authored sneak-attack frame variant over connected grammar; no catalogue text"},
        "reader_facing_next_operator": "Derive any successor from this variant's persisted full-scheduler mismatch; do not weaken the ordinary collocation or revive retired centers.",
        "scope": "This is a bounded construction run; exactness and feature parsing do not certify human readability.",
    }
    base.BY_KIND, base.FRAME = old_by_kind, old_frame
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--state-limit", type=int, default=100_000); parser.add_argument("--closure-limit", type=int, default=100)
    args = parser.parse_args()
    if args.out.exists(): parser.error(f"refusing to overwrite {args.out}")
    result = run(state_limit=args.state_limit, closure_limit=args.closure_limit); args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "states": result["stats"]["state_count"], "exact": len(result["exact_closures"]), "admitted": len(result["admitted_closures"])}, indent=2))


if __name__ == "__main__": main()
