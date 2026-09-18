"""Cross-word tail segmentation with an eight-pair free-centre product.

Opening predicate/NP residuals and terminal purpose-attachment tails are
indexed as continuous character streams across token boundaries.  Semantic
attachment changes remain within an action/object/location family.  A path
must reach eight actual outer pairs before the connected grammar is seeded for
free even/odd centre expansion; shorter seams are retained only as diagnostics.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.cross_attachment_inside_product_20260913 import inside_product, independent_parse
from experiments.residual_aware_endpoint_expansion_20260913 import CORES, Core
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

MIN_LETTERS, MAX_LETTERS = 30, 160


def family_key(core: Core) -> tuple[str, str, str, str, str]:
    return core.action, core.object, core.object_adj, core.place, core.place_adj


def tail_words(core: Core) -> tuple[str, ...]:
    return core.words[-4:]


def compose_words(source: Core, selected: Core) -> tuple[str, ...]:
    """Compose one typed imperative from a shared frame and a selected tail.

    The opening/location comes from ``source`` and the purpose attachment from
    ``selected``.  ``family_key`` guarantees that the action, object, and
    location roles remain compatible; the independent parser still reparses
    this composite surface from its own role inventory at closure.
    """
    if family_key(source) != family_key(selected):
        raise ValueError("cross-family attachment is not semantically licensed")
    return tuple(source.words[:9]) + tuple(selected.words[-4:])


def tail_segmentation_index(cores=CORES) -> dict:
    """Index every reverse prefix, including prefixes crossing word seams."""
    reverse_index = defaultdict(list)
    for core in cores:
        tail = "".join(tail_words(core)); reversed_tail = tail[::-1]
        for depth in range(1, len(reversed_tail) + 1):
            reverse_index[(core.action, depth, reversed_tail[:depth])].append({
                "core": core.identifier, "family": family_key(core), "tail_words": list(tail_words(core)),
                "reversed_prefix": reversed_tail[:depth]})
    return {"index": {f"{a}:{n}:{s}": rows for (a, n, s), rows in reverse_index.items()},
            "depths": sorted({n for _, n, _ in reverse_index}), "max_depth": max((n for _, n, _ in reverse_index), default=0)}


def endpoint_joins(cores=CORES) -> tuple[dict, ...]:
    index = tail_segmentation_index(cores)["index"]; joins = []
    for source in cores:
        opening = "".join(source.words[:4])
        for selected in cores:
            if selected.identifier == source.identifier or family_key(selected) != family_key(source): continue
            tail = "".join(tail_words(selected))[::-1]; depth = 0
            for left, right in zip(opening, tail):
                if left != right: break
                depth += 1
            if depth < 3: continue
            key = f"{source.action}:3:{tail[:3]}"
            indexed = any(row["core"] == selected.identifier for row in index.get(key, ()))
            if not indexed: continue
            joins.append({"source_core": source.identifier, "selected_core": selected.identifier,
                          "family": list(family_key(source)), "opening_words": list(source.words[:4]),
                          "middle_words": list(source.words[4:9]),
                          "terminal_words": list(tail_words(selected)),
                          "composite_words": list(compose_words(source, selected)),
                          "opening_letters": opening,
                          "reversed_terminal_letters": tail, "matched_pairs": depth,
                          "cross_word_tail_segmentation": True, "index_depth": 3})
    return tuple(joins)


def render(words):
    text = " ".join(words); return text[:1].upper() + text[1:] + "."


def outside_ledger(text):
    tape = normalize_letters(text); events = []
    for i in range(len(tape) // 2):
        j = len(tape) - 1 - i; e = {"pair": i + 1, "left": tape[i], "right": tape[j], "equal": tape[i] == tape[j]}; events.append(e)
        if not e["equal"]: return {"exact": False, "letters": len(tape), "events": events, "first_mismatch": e, "normalized_sha256": sha256(tape.encode()).hexdigest()}
    return {"exact": bool(tape), "letters": len(tape), "events": events, "first_mismatch": None, "normalized_sha256": sha256(tape.encode()).hexdigest()}


def run(max_paths_per_channel=100_000, max_states=100_000):
    joins = endpoint_joins(); by_id = {core.identifier: core for core in CORES}; channels = defaultdict(lambda: {"eligible": 0, "scheduled": 0, "truncated": False, "deepest": 0})
    residual_diagnostics, free_runs, exact = [], [], []
    # Visit deepest real endpoint seams first so a bounded diagnostic ledger
    # retains the strongest rejection rather than only the first shallow rows.
    for join in sorted(joins, key=lambda row: row["matched_pairs"], reverse=True):
        channel = join["opening_letters"][:3]; channels[channel]["eligible"] += 1
        source = by_id[join["source_core"]]; selected = by_id[join["selected_core"]]
        words = tuple(join["composite_words"]); text = render(words)
        if join["matched_pairs"] < 8:
            if len(residual_diagnostics) < 200:
                residual_diagnostics.append({"source_core": join["source_core"], "selected_core": join["selected_core"],
                                             "rendered": text, "endpoint_join": join, "outside_in_ledger": outside_ledger(text),
                                             "rejection": "free_center_requires_eight_outer_pairs"})
            continue
        if channels[channel]["scheduled"] >= max_paths_per_channel:
            channels[channel]["truncated"] = True; continue
        grammar = __import__("experiments.whole_text_palindrome_product_20260913", fromlist=["compile_slots"]).compile_slots(tuple((word,) for word in words))
        product = inside_product(grammar, words, join["matched_pairs"], max_states=max_states)
        channels[channel]["scheduled"] += 1; channels[channel]["deepest"] = max(channels[channel]["deepest"], product["deepest_pairs"])
        free_runs.append({"source_core": join["source_core"], "selected_core": selected.identifier, "endpoint_join": join,
                          "product": {key: value for key, value in product.items() if key != "records"}})
        for record in product["records"]:
            central = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
            exact.append({"record_kind": "cross_word_tail_free_center_closure", "rendered": text, "endpoint_join": join,
                          "inside_product": record, "outside_in_ledger": outside_ledger(text),
                          "independent_parse": independent_parse(text), "central_admission": central,
                          "mechanically_admitted": all(central.values()),
                          "reader_status": "human-unreviewed; programmatic checks do not certify readability"})
    admitted = [row for row in exact if row["mechanically_admitted"]]
    return {"status": "cross_word_tail_segmentation_product", "config": {"min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS,
        "minimum_endpoint_pairs": 3, "minimum_pairs_before_free_center": 8, "cross_word_tail_segmentation": True,
        "partial_word_suffix_index": True, "free_even_odd_center": True, "online_span_prune": True,
        "independent_complete_reparse": True, "corpus_generation": False, "human_readability_required_after_admission": True},
        "authored_core_count": len(CORES), "endpoint_join_count": len(joins),
        "endpoint_depth_histogram": {str(depth): sum(row["matched_pairs"] == depth for row in joins)
                                     for depth in sorted({row["matched_pairs"] for row in joins})},
        "maximum_endpoint_pairs": max((row["matched_pairs"] for row in joins), default=0),
        "endpoint_scope_exhausted": True,
        "free_center_search_entered": bool(free_runs),
        "channel_coverage": dict(channels),
        "residual_diagnostics": residual_diagnostics, "free_center_runs": free_runs, "exact_candidates": exact,
        "admitted_candidates": admitted, "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "material": "authored modular imperative purpose/location inventory; no catalogue or fixed output",
        "known_catalogue_check": "central admission gates applied"},
        "next_construction_operator": "Add a semantically licensed terminal attachment whose continuous reverse suffix extends a distinct opening residual beyond eight pairs; do not patch a token.",
        "reader_facing_next_test": "Only an admitted exact surface may enter randomized blinded intact-prose versus shuffled-control reading; programmatic checks do not certify readability."}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--out", required=True, type=Path); parser.add_argument("--max-paths-per-channel", type=int, default=100_000); parser.add_argument("--max-states", type=int, default=100_000); args = parser.parse_args()
    if args.out.exists(): parser.error(f"output already exists: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True); result = run(args.max_paths_per_channel, args.max_states); args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "joins": result["endpoint_join_count"], "exact": len(result["exact_candidates"]), "admitted": len(result["admitted_candidates"])}, indent=2))


if __name__ == "__main__": main()
