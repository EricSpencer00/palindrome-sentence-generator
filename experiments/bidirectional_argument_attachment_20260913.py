"""Two-sided segmentation intersection with an explicit continuation gate.

The source and target share an identifying-event core, but attach possession
to different arguments.  Their attachment graphs are *not* claimed equivalent.
Only target words are rendered or admitted.  Prefix/suffix boundaries can
disagree in either direction; no tape is copied into a reflected shell.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.credential_attachment_source_20260913 import Source, source_steps, source_meaning
from experiments.credential_attachment_target_20260913 import Target, target_steps, target_meaning
from experiments.credential_attachment_validator_20260913 import parse_credential_sentence, render_credential_sentence
from experiments.boundary_sensitive_semantic_intersection_20260913 import intersection_statistics
from experiments.boundary_shifting_sentence_lattice_20260913 import Edge, Lattice, Node, compile_lattice, search_lattice
from llm_palindrome.admission import mechanical_admission_checks


@dataclass(frozen=True)
class BothEnds:
    source: int
    target: int
    prefix: str = ""
    source_left: int = 0
    target_left: int = 0
    source_tail: int = 0
    target_tail: int = 0
    length_capped: int = 0


def coaccessible(lattice, accepting):
    incoming = defaultdict(list)
    for edge in lattice.edges:
        incoming[edge.target].append(edge)
    live, pending = set(accepting), list(accepting)
    while pending:
        node = pending.pop()
        for edge in incoming[node]:
            if edge.source not in live:
                live.add(edge.source)
                pending.append(edge.source)
    return Lattice(lattice.nodes, tuple(e for e in lattice.edges if e.source in live and e.target in live), lattice.start, frozenset(accepting)), len(live)


def compile_bidirectional(source, target, source_key, target_key, pairs=6):
    if pairs < 6:
        raise ValueError("six real outer pairs are required")
    aout, bout = defaultdict(list), defaultdict(list)
    for edge in source.edges:
        aout[edge.source].append(edge)
    for edge in target.edges:
        bout[edge.source].append(edge)
    nodes, ids, edges, ends, raw_ends = [], {}, set(), set(), set()
    def index(state):
        if state not in ids:
            ids[state] = len(nodes)
            nodes.append(Node(state, target.nodes[state.target].lexical_prefix))
        return ids[state]
    initial = BothEnds(source.start, target.start)
    start = index(initial)
    todo, seen = [initial], set()
    tail_mask = (1 << pairs) - 1
    interior_mask = tail_mask ^ 1
    while todo:
        state = todo.pop()
        if state in seen:
            continue
        seen.add(state)
        if state.source in source.accepting and state.target in target.accepting and source_key(source.nodes[state.source].parser_state) == target_key(target.nodes[state.target].parser_state):
            raw_ends.add(index(state))
            left_shift = (state.source_left ^ state.target_left) & interior_mask
            right_shift = (state.source_tail ^ state.target_tail) & interior_mask
            if state.length_capped >= 2 * pairs and left_shift and right_shift:
                ends.add(index(state))
        for a in aout[state.source]:
            for b in bout[state.target]:
                if a.char != b.char:
                    continue
                before = len(state.prefix) < pairs
                prefix = state.prefix + a.char if before else state.prefix
                cut = len(prefix)
                ab, bb = source.nodes[a.target].boundary, target.nodes[b.target].boundary
                following = BothEnds(a.target, b.target, prefix,
                    state.source_left | ((1 << cut) if before and cut < pairs and ab else 0),
                    state.target_left | ((1 << cut) if before and cut < pairs and bb else 0),
                    ((state.source_tail << 1) | int(ab)) & tail_mask,
                    ((state.target_tail << 1) | int(bb)) & tail_mask,
                    min(2 * pairs, state.length_capped + 1))
                edges.add(Edge(index(state), index(following), a.char, b.completed_word))
                todo.append(following)
    graph = Lattice(tuple(nodes), tuple(sorted(edges, key=lambda e: (e.source, e.target, e.char, e.completed_word))), start, frozenset(raw_ends))
    filtered, live = coaccessible(graph, ends)
    return filtered, {"unrestricted": intersection_statistics(graph), "two_sided_boundary_filtered": intersection_statistics(filtered),
                      "constructed_states": len(nodes), "coaccessible_two_sided_states": live}


def frontier_certificate(lattice, pairs=6, required_letters=6):
    """Count only six-pair, coaccessible matched continuations, not vocabulary."""
    out, inc = defaultdict(list), defaultdict(list)
    for edge in lattice.edges:
        out[edge.source].append(edge)
        inc[edge.target].append(edge)
    @lru_cache(maxsize=None)
    def reaches(a, b):
        return a == b or any(reaches(e.target, b) for e in out[a])
    @lru_cache(maxsize=None)
    def minimum_words(a, b):
        if a == b:
            return 0
        return min((int(bool(e.completed_word)) + minimum_words(e.target, b) for e in out[a] if reaches(e.target, b)), default=10**9)
    todo = [(lattice.start, end, 0, "", end) for end in lattice.accepting]
    rows, stats, deepest = defaultdict(list), defaultdict(int), {"depth": -1}
    while todo:
        left, right, depth, tape, end = todo.pop()
        if not reaches(left, right):
            continue
        stats["states"] += 1
        if depth > deepest["depth"]:
            deepest = {"depth": depth, "left_next_letters": sorted({e.char for e in out[left] if reaches(e.target, right)}),
                       "right_next_letters_inward": sorted({e.char for e in inc[right] if reaches(left, e.source)})}
        if depth and lattice.nodes[left].boundary and lattice.nodes[right].boundary and minimum_words(left, right) >= 2:
            stats["proper_island_prunes"] += 1
            continue
        options = [(a, b) for a in out[left] for b in inc[right] if a.char == b.char and reaches(a.target, b.source)]
        if depth == pairs:
            left_letters = {e.char for e in out[left] if reaches(e.target, right)}
            right_letters = {e.char for e in inc[right] if reaches(left, e.source)}
            state = lattice.nodes[end].parser_state
            left_cross = state.source_left ^ state.target_left
            right_cross = state.source_tail ^ state.target_tail
            rows[tape].append({"left_node": left, "right_node": right, "accepting_node": end,
                               "actual_pairs": depth, "left_crossed_cuts": [i for i in range(1, pairs) if left_cross & (1 << i)],
                               "right_crossed_cuts": [i for i in range(1, pairs) if right_cross & (1 << i)],
                               "left_next_letters": sorted(left_letters), "right_next_letters": sorted(right_letters),
                               "paired_next_letters": sorted({a.char for a, b in options})})
            continue
        if not options:
            stats["dead_frontiers"] += 1
        for a, b in options:
            todo.append((a.target, b.source, depth + 1, tape + a.char, end))
    channels = []
    for tape, witnesses in sorted(rows.items()):
        left = set().union(*(set(w["left_next_letters"]) for w in witnesses))
        right = set().union(*(set(w["right_next_letters"]) for w in witnesses))
        paired = set().union(*(set(w["paired_next_letters"]) for w in witnesses))
        two_sided = all(w["left_crossed_cuts"] and w["right_crossed_cuts"] for w in witnesses)
        channels.append({"tape": tape, "actual_pairs": pairs, "frontier_states": len(witnesses),
                         "left_next_letters": sorted(left), "right_next_letters": sorted(right), "paired_next_letters": sorted(paired),
                         "qualified": two_sided and min(len(left), len(right), len(paired)) >= required_letters,
                         "witnesses": witnesses})
    qualified = {row["tape"] for row in channels if row["qualified"]}
    accepting = {end for end in lattice.accepting if lattice.nodes[end].parser_state.prefix in qualified}
    selected, _ = coaccessible(lattice, accepting)
    return selected, {"required_actual_pairs": pairs, "required_paired_next_letters": required_letters,
                      "stats": dict(stats), "deepest": deepest, "channels": channels,
                      "qualified_channels": len(qualified), "six_letter_continuation_requirement_met": bool(qualified)}


def review_closures(records, min_letters=100, max_letters=240):
    reviews = []
    for words, row in {tuple(r["words"]): r for r in records}.items():
        rendering = render_credential_sentence(words)
        checks = mechanical_admission_checks(rendering, min_letters=min_letters, max_letters=max_letters)
        parses = parse_credential_sentence(words)
        failures = [name for name, passed in checks.items() if not passed]
        if not any(p["semantic_relation_valid"] for p in parses):
            failures.append("independent_final_semantics")
        if row["matched_depth"] < 6:
            failures.append("fewer_than_six_actual_pairs")
        reviews.append({"rendered_diagnostic": rendering, "tokens": words, "normalized": row["tape"], "letters": len(row["tape"]),
                        "exact": row["tape"] == row["tape"][::-1], "central_admission": checks, "independent_parses": parses,
                        "failures": failures, "external_provenance": "required; unchecked", "promoted": False})
    return reviews


def run():
    source = compile_lattice(Source("start"), source_steps, lambda s: source_meaning(s) is not None)
    target = compile_lattice(Target("initial"), target_steps, lambda s: target_meaning(s) is not None)
    lattice, counts = compile_bidirectional(source, target, lambda s: source_meaning(s)["shared_core"], lambda s: target_meaning(s)["shared_core"])
    selected, certificate = frontier_certificate(lattice)
    result = search_lattice(selected, probe_pairs=6)
    reviews = review_closures(result["records"])
    paths = [Path(__file__), ROOT / "experiments/credential_attachment_source_20260913.py", ROOT / "experiments/credential_attachment_target_20260913.py", ROOT / "experiments/credential_attachment_validator_20260913.py"]
    return {"status": "qualified_run" if certificate["qualified_channels"] else "construction_regime_not_qualified",
            "config": {"two_sided_word_boundary_disagreement": True, "min_actual_pairs": 6, "min_coaccessible_paired_next_letters": 6,
                       "independent_source_target_final_grammars": True, "source_plus_reverse_source": False,
                       "fixed_center": False, "source_target_attachment_graphs_claimed_equivalent": False},
            "semantic_scope": "Shared credential-to-attendant identifying core; source ownership attaches to a duty domain, target ownership to an attendant. Those additional attachment meanings are deliberately not asserted equivalent or reader-certified.",
            "construction": counts, "production_frontier_certificate": certificate,
            "product": {k: v for k, v in result.items() if k != "records"}, "exact_closures": len(result["records"]),
            "closure_reviews": reviews, "pending_external_review": [r for r in reviews if not r["failures"]], "promoted_candidates": [],
            "requirements_unmet": [] if certificate["qualified_channels"] else ["No production frontier simultaneously provides two-sided disagreement at six actual pairs and six compatible next-letter classes."],
            "next_non_wrapper_repair": "Search an independently parsed relative-clause/coordination attachment ambiguity, not another role-compound replacement; permit variable-position two-sided token changes before freezing a shared endpoint key. Keep the six-continuation gate unchanged.",
            "source_hashes": {p.name: sha256(p.read_bytes()).hexdigest() for p in paths},
            "scope": "Failed qualification is not a successful production construction. Toy fixtures establish mechanics only. No model wrapper, reflected full-tape shell, or catalogue seed is used."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run()
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"status": result["status"], "construction": result["construction"],
                          "frontier": result["production_frontier_certificate"], "exact_closures": result["exact_closures"]}))
    else:
        print(json.dumps(result, indent=2))
