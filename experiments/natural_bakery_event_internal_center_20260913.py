"""Natural bakery-event-first internal-centre search.

The complete discourse frame is chosen before character search: a baker bakes
fresh challah during a festival and records the family recipe afterward.  The
single-word centre is then selected from that frame and its best internal
boundary (``chal|lah``) is derived mechanically.  The connected grammar,
one-character scheduler, independent replay, and admission audit are inherited
only through a source-hash-locked module whose bindings are restored after use.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import fresh_internal_center_inventory_20260913 as base

BASE_SOURCE_SHA256 = "6614ae113e86fda1472b3f7cefbfdcddfebc56262c1e4c9d5a874ac7472c287b"


def derived_pivots(word: str) -> tuple[int, ...]:
    scores = {}
    for pivot in range(1, len(word)):
        left, right = word[:pivot][::-1], word[pivot:]
        common = 0
        while common < min(len(left), len(right)) and left[common] == right[common]:
            common += 1
        scores[pivot] = common
    best = max(scores.values())
    return tuple(pivot for pivot, score in scores.items() if score == best and score > 0)


def isolated_bindings():
    source = Path(base.__file__).resolve()
    actual = sha256(source.read_bytes()).hexdigest()
    if actual != BASE_SOURCE_SHA256:
        raise RuntimeError(f"base source changed: expected {BASE_SOURCE_SHA256}, found {actual}")
    old_by_kind, old_frame, old_pivots = base.BY_KIND, base.FRAME, base.PIVOTS
    by_kind = {kind: tuple(items) for kind, items in old_by_kind.items()}
    by_kind["noun_person"] += (base.Lexeme("baker", "noun_person", "person"),)
    by_kind["noun_center"] += (base.Lexeme("challah", "noun_center", "challah"),)
    by_kind["adj_center"] += (base.Lexeme("fresh", "adj_center", "challah"),)
    by_kind["verb_intro"] += (base.Lexeme("baked", "verb_intro", "baking", "person", "challah"),)
    by_kind["noun_event"] += (base.Lexeme("festival", "noun_event", "event"),)
    by_kind["adj_event"] += (
        base.Lexeme("family", "adj_event", "event"),
        base.Lexeme("local", "adj_event", "event"),
    )
    by_kind["noun_artifact"] += (base.Lexeme("recipe", "noun_artifact", "record"),)
    by_kind["verb_record"] += (base.Lexeme("recorded", "verb_record", "documentation", "person", "record"),)
    base.BY_KIND = by_kind
    base.FRAME = {"challah": {"verb": "baked", "adjectives": {"fresh"}, "relations": {"during"}}}
    pivots = derived_pivots("challah")
    if pivots != (4,):
        raise RuntimeError(f"unexpected derived challah pivot: {pivots}")
    base.PIVOTS = {"challah": pivots}
    return old_by_kind, old_frame, old_pivots


def run(*, state_limit: int = 100_000, closure_limit: int = 100) -> dict[str, object]:
    old_by_kind, old_frame, old_pivots = isolated_bindings()
    try:
        grammar = base.Grammar()
        leaf_slots = base.slots(grammar.expand(base.Symbol("S")))
        stats = Counter(state_count=0, search_states=0, lexical_assignments_considered=0,
                        character_emissions=0, residual_cancellations=0,
                        residual_contradictions=0, complete_tree_states=0,
                        exact_closures=0, admitted_closures=0)
        frames = [
            {
                "first_sentence": "The careful baker baked a fresh challah during the annual festival.",
                "second_sentence": "The careful baker recorded the family recipe after the local festival today.",
                "center": "challah", "pivot": "chal|lah", "verb": "baked",
                "adjective": "fresh", "relation": "during", "event_noun": "festival",
                "semantic_status": "complete ordinary bakery event/discourse pair fixed before center search",
            },
            {
                "first_sentence": "The patient baker baked a fresh challah during the public festival.",
                "second_sentence": "The patient baker recorded the family recipe after the annual festival today.",
                "center": "challah", "pivot": "chal|lah", "verb": "baked",
                "adjective": "fresh", "relation": "during", "event_noun": "festival",
                "semantic_status": "complete ordinary bakery event/discourse pair fixed before center search",
            },
        ]
        inventory = base.centre_inventory(stats)
        boundaries = base.boundary_inventory(stats)
        exact, admitted = base.search(grammar, leaf_slots, state_limit=state_limit,
                                      closure_limit=closure_limit, stats=stats)
        controls = []
        for frame in frames:
            text = frame["first_sentence"] + " " + frame["second_sentence"]
            controls.append(base.audit(grammar, text, "complete_natural_bakery_event_pair_control",
                                       tuple(re.findall(r"[a-z]+", text.lower()))))
        deepest = {
            "ledger_before_rejection": stats.get("deepest_live_ledger", []),
            "next_literal_rejection": stats.get("deepest_next_literal_rejection"),
            "independent_replay": stats.get("deepest_independent_replay"),
            "emissions_including_rejection": stats.get("deepest_ledger_length", 0),
        }
        return {
            "status": "natural_bakery_event_internal_center_typed_tree_search",
            "config": {
                "state_limit": state_limit, "closure_limit": closure_limit,
                "one_connected_tree": True, "grammar_owns_every_leaf": True,
                "complete_event_pair_before_center_search": True,
                "frame_first_inventory": True, "word_internal_center_inventory": True,
                "pivot_derived_from_center_boundary": True,
                "prepared_multiword_center": False, "one_character_emission_states": True,
                "replayed_ledger": True, "independent_complete_reparse": True,
                "reject_every_self_palindromic_contiguous_multiword_span": True,
                "corpus_or_catalogue_generation": False,
            },
            "grammar_leaf_count": len(leaf_slots), "sentence_frame_inventory": frames,
            "centre_inventory": inventory, "boundary_inventory": boundaries,
            "best_boundary_trace": boundaries[0] if boundaries else None,
            "deepest_full_scheduler_replay": deepest, "stats": dict(stats),
            "exact_closures": exact, "admitted_closures": admitted,
            "complete_grammar_controls": controls,
            "provenance": {
                "generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                "base_generator": str(Path(base.__file__).resolve()),
                "base_generator_sha256": BASE_SOURCE_SHA256,
                "grammar_sha256": grammar.digest(),
                "material": "task-authored complete natural bakery event pairs; no catalogue text",
            },
            "reader_facing_next_operator": "Derive the successor from this complete-frame scheduler ledger; do not patch the bakery relation or revive retired center families.",
            "scope": "This bounded construction run records exact diagnostics; programmatic controls do not certify human readability.",
        }
    finally:
        base.BY_KIND, base.FRAME, base.PIVOTS = old_by_kind, old_frame, old_pivots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--state-limit", type=int, default=100_000)
    parser.add_argument("--closure-limit", type=int, default=100)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    result = run(state_limit=args.state_limit, closure_limit=args.closure_limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "states": result["stats"]["state_count"],
                      "exact": len(result["exact_closures"]),
                      "admitted": len(result["admitted_closures"])}, indent=2))


if __name__ == "__main__":
    main()
