"""Intersect two independent forward languages before whole-text symmetry.

Only source/target paths with a genuine token-boundary disagreement inside the
five-character prefix are coaccessible.  The palindrome product must then
actually match five outer pairs.  Source cuts are provisional; only target
cuts define rendered words and the online island guard.  No source tape is
reflected, and no source or target clause is installed as a center.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.acoustic_source_grammar_20260913 import SourceState, source_transitions, source_signature
from experiments.acoustic_target_grammar_20260913 import TargetState, target_transitions, target_signature
from experiments.acoustic_whole_sentence_validator_20260913 import parse_acoustic_sentence
from experiments.boundary_shifting_sentence_lattice_20260913 import (
    Edge, Lattice, Node, compile_lattice, lattice_statistics, search_lattice,
)
from llm_palindrome.admission import mechanical_admission_checks


@dataclass(frozen=True)
class IntersectionState:
    source_node: int
    target_node: int
    prefix: str = ""
    provisional_cuts: tuple[int, ...] = ()
    final_cuts: tuple[int, ...] = ()
    provisional_words: tuple[str, ...] = ()
    final_words: tuple[str, ...] = ()

    @property
    def crossed_cuts(self):
        return tuple(sorted(set(self.provisional_cuts) - set(self.final_cuts)))


def distinct_target_surfaces(lattice):
    """Exact target-word projection, determinized before finite counting.

Different source segmentations can induce several character paths for the
same rendered target words.  A subset state after each complete target word
removes that ambiguity without confusing word surfaces with letter tapes.
"""
    outgoing = defaultdict(list)
    for edge in lattice.edges:
        outgoing[edge.source].append(edge)

    @lru_cache(maxsize=None)
    def word_successors(node):
        options = defaultdict(set)
        for edge in outgoing[node]:
            if edge.completed_word:
                options[edge.completed_word].add(edge.target)
            else:
                for word, destinations in word_successors(edge.target):
                    options[word].update(destinations)
        return tuple((word, frozenset(destinations)) for word, destinations in sorted(options.items()))

    active = set()
    @lru_cache(maxsize=None)
    def count(subset):
        if subset in active:
            raise ValueError("target-word projection must be acyclic")
        active.add(subset)
        successors = defaultdict(set)
        for node in subset:
            for word, destinations in word_successors(node):
                successors[word].update(destinations)
        total = int(bool(subset & lattice.accepting))
        total += sum(count(frozenset(destinations)) for destinations in successors.values())
        active.remove(subset)
        return total

    surfaces = count(frozenset({lattice.start}))
    return {"distinct_rendered_target_surfaces": surfaces, "determinized_target_word_states": count.cache_info().currsize}


def intersection_statistics(lattice):
    stats = lattice_statistics(lattice)
    pairs = stats.pop("accepted_lexical_paths")
    return {**stats, "accepted_source_target_derivation_pairs": pairs, **distinct_target_surfaces(lattice)}


def semantic_intersection(source, target, source_key, target_key, probe_pairs=5, require_shift=True):
    if probe_pairs < 1:
        raise ValueError("a positive actual-pair frontier is required")
    out_source, out_target = defaultdict(list), defaultdict(list)
    for edge in source.edges:
        out_source[edge.source].append(edge)
    for edge in target.edges:
        out_target[edge.source].append(edge)
    nodes, ids, edges, accepting, unrestricted_accepting = [], {}, set(), set(), set()
    def node(state):
        if state not in ids:
            ids[state] = len(nodes)
            nodes.append(Node(state, target.nodes[state.target_node].lexical_prefix))
        return ids[state]
    initial = IntersectionState(source.start, target.start)
    start = node(initial)
    pending, visited = [initial], set()
    while pending:
        state = pending.pop()
        if state in visited:
            continue
        visited.add(state)
        left, right = state.source_node, state.target_node
        if left in source.accepting and right in target.accepting and source_key(source.nodes[left].parser_state) == target_key(target.nodes[right].parser_state):
            unrestricted_accepting.add(node(state))
            if not require_shift or (len(state.prefix) == probe_pairs and state.crossed_cuts):
                accepting.add(node(state))
        for a in out_source[left]:
            for b in out_target[right]:
                if a.char != b.char:
                    continue
                before_frontier = len(state.prefix) < probe_pairs
                prefix = state.prefix + a.char if before_frontier else state.prefix
                cut = len(prefix)
                source_cut = before_frontier and cut < probe_pairs and source.nodes[a.target].boundary
                target_cut = before_frontier and cut < probe_pairs and target.nodes[b.target].boundary
                following = IntersectionState(a.target, b.target, prefix,
                    state.provisional_cuts + ((cut,) if source_cut else ()),
                    state.final_cuts + ((cut,) if target_cut else ()),
                    state.provisional_words + ((a.completed_word,) if before_frontier and a.completed_word else ()),
                    state.final_words + ((b.completed_word,) if before_frontier and b.completed_word else ()))
                edges.add(Edge(node(state), node(following), a.char, b.completed_word))
                pending.append(following)
    all_edges = tuple(sorted(edges, key=lambda e: (e.source, e.target, e.char, e.completed_word)))
    unrestricted = Lattice(tuple(nodes), all_edges, start, frozenset(unrestricted_accepting))
    before = intersection_statistics(unrestricted)
    incoming = defaultdict(list)
    for edge in all_edges:
        incoming[edge.target].append(edge)
    live, todo = set(accepting), list(accepting)
    while todo:
        node_id = todo.pop()
        for edge in incoming[node_id]:
            if edge.source not in live:
                live.add(edge.source)
                todo.append(edge.source)
    # Preserve stable node ids for auditable source/target prefix witnesses.
    filtered = Lattice(tuple(nodes), tuple(e for e in all_edges if e.source in live and e.target in live), start, frozenset(accepting))
    return filtered, {"count_units": {"derivation_pairs": "synchronized source/target lexical analyses", "target_surfaces": "distinct target word sequences, counted by determinizing the word-level projection; not distinct normalized tapes"},
                      "unrestricted_intersection": before, "shift_filtered_intersection": intersection_statistics(filtered),
                      "constructed_states": len(nodes), "coaccessible_states": len(live),
                      "semantic_accepting_states_before_shift": len(unrestricted_accepting), "semantic_accepting_states_after_shift": len(accepting)}


def production_grammars():
    source = compile_lattice(SourceState("start"), source_transitions, lambda state: source_signature(state) is not None)
    target = compile_lattice(TargetState("initial"), target_transitions, lambda state: target_signature(state) is not None)
    return source, target


def run(min_letters=100, max_letters=240):
    source, target = production_grammars()
    lattice, construction = semantic_intersection(source, target, source_signature, target_signature)
    result = search_lattice(lattice, probe_pairs=5)
    result["language"] = construction["shift_filtered_intersection"]
    states_by_repr = {repr(node.parser_state): node.parser_state for node in lattice.nodes}
    ids_by_repr = {repr(node.parser_state): index for index, node in enumerate(lattice.nodes)}
    outgoing, incoming, source_edges = defaultdict(list), defaultdict(list), defaultdict(list)
    for edge in lattice.edges:
        outgoing[edge.source].append(edge)
        incoming[edge.target].append(edge)
    for edge in source.edges:
        source_edges[(edge.source, edge.target, edge.char)].append(edge)

    @lru_cache(maxsize=None)
    def path_between(first, last):
        if first == last:
            return ()
        for edge in outgoing[first]:
            tail = path_between(edge.target, last)
            if tail is not None:
                return (edge,) + tail
        return None

    def complete_witness(left, right):
        beginning, middle = path_between(lattice.start, left), path_between(left, right)
        ending = next((path_between(right, end) for end in sorted(lattice.accepting)
                       if path_between(right, end) is not None), None)
        if beginning is None or middle is None or ending is None:
            raise AssertionError("production witness is not a complete coaccessible path")
        path = beginning + middle + ending
        source_words, target_words = [], []
        for edge in path:
            if edge.completed_word:
                target_words.append(edge.completed_word)
            a = lattice.nodes[edge.source].parser_state.source_node
            b = lattice.nodes[edge.target].parser_state.source_node
            source_edge, = source_edges[(a, b, edge.char)]
            if source_edge.completed_word:
                source_words.append(source_edge.completed_word)
        tape = "".join(edge.char for edge in path)
        if tape != "".join(source_words) or tape != "".join(target_words):
            raise AssertionError("source and target whole-path tapes differ")
        parses = parse_acoustic_sentence(target_words)
        if not any(row["semantic_relation_valid"] for row in parses):
            raise AssertionError("production witness lacks independent whole-sentence semantics")
        return {"source_tokens": source_words, "target_tokens": target_words, "normalized": tape,
                "letters": len(tape), "exact_palindrome": tape == tape[::-1],
                "independent_target_semantics": parses, "purpose": "noncandidate completion proving production coaccessibility"}

    witnesses = []
    for channel in result["channels"]:
        for sampled in channel["sample_states"]:
            state = states_by_repr[sampled["left_parser_state"]]
            if not state.crossed_cuts or len(state.prefix) != 5:
                raise AssertionError("an eligible production channel lacks actual relexicalization")
            witnesses.append({"matched_tape": channel["tape"], "actual_outer_pairs": 5,
                              "provisional_complete_words": state.provisional_words,
                              "provisional_unfinished_word": source.nodes[state.source_node].lexical_prefix,
                              "final_complete_words": state.final_words,
                              "final_unfinished_word": target.nodes[state.target_node].lexical_prefix,
                              "provisional_boundaries": state.provisional_cuts,
                              "final_boundaries": state.final_cuts, "crossed_provisional_cuts": state.crossed_cuts,
                              "source_semantic_state": repr(source.nodes[state.source_node].parser_state),
                              "target_semantic_state": repr(target.nodes[state.target_node].parser_state),
                              "whole_path_completion": complete_witness(ids_by_repr[sampled["left_parser_state"]], ids_by_repr[sampled["right_parser_state"]]),
                              "production_not_fixture": True})
    rejection_codes, pending, closure_reviews = Counter(), [], []
    exact = {tuple(record["words"]): record for record in result["records"]}
    for words, record in exact.items():
        parses = parse_acoustic_sentence(words)
        checks = mechanical_admission_checks(" ".join(words), min_letters=min_letters, max_letters=max_letters)
        failures = [name for name, passed in checks.items() if not passed]
        if not any(row["semantic_relation_valid"] for row in parses):
            failures.append("independent_whole_sentence_semantics")
        boundary, rendered_cuts = 0, set()
        for word in words:
            boundary += len(word)
            rendered_cuts.add(boundary)
        matching_witnesses = [w for w in witnesses if w["matched_tape"] == record["tape"][:5]
                              and any(cut not in rendered_cuts for cut in w["crossed_provisional_cuts"])]
        if record["matched_depth"] < 5 or not matching_witnesses:
            failures.append("production_relexicalization_at_five_actual_pairs")
        closure_reviews.append({"rendered_diagnostic": " ".join(words).capitalize() + ".", "tokens": words,
                                "normalized": record["tape"], "letters": record["letters"],
                                "exact": record["tape"] == record["tape"][::-1],
                                "central_admission": checks, "independent_semantic_parses": parses,
                                "failures": failures, "candidate_accepted": False,
                                "external_provenance": "not checked", "purpose": "closure audit, not promotion"})
        if failures:
            rejection_codes.update(failures)
        else:
            pending.append({"tokens": words, "normalized": record["tape"], "letters": record["letters"],
                            "central_checks": checks, "independent_semantic_parses": parses,
                            "external_provenance": "required; unchecked", "promoted": False})
    paths = [Path(__file__), ROOT / "experiments/acoustic_source_grammar_20260913.py",
             ROOT / "experiments/acoustic_target_grammar_20260913.py", ROOT / "experiments/acoustic_whole_sentence_validator_20260913.py"]
    deepest = result["deepest"]
    mismatch = None
    if deepest["depth"] >= 0:
        left_id, right_id = ids_by_repr[deepest["left_parser_state"]], ids_by_repr[deepest["right_parser_state"]]
        first_chars = sorted({edge.char for edge in outgoing[left_id]})
        last_chars = sorted({edge.char for edge in incoming[right_id]})
        mismatch = {"next_pair": deepest["depth"] + 1, "left_next_letters": first_chars,
                    "right_next_letters_read_inward": last_chars,
                    "disjoint": not set(first_chars) & set(last_chars),
                    "left_final_lexical_prefix": deepest["left_lexical_prefix"],
                    "right_final_lexical_prefix": deepest["right_lexical_prefix"]}
    return {"status": "boundary_sensitive_semantic_intersection", "config": {
        "min_letters": min_letters, "max_letters": max_letters, "required_actual_pairs": 5,
        "independently_declared_source_and_target_grammars": True, "independent_final_semantic_parser": True,
        "same_tape_forward_semantic_intersection": True, "source_plus_reverse_source": False,
        "source_cuts_provisional_target_cuts_rendered": True, "production_relexicalization_required": True,
        "shift_window": "strictly inside the first five letters; absence removes the whole path before palindrome search",
        "free_even_odd_center": True, "target_online_island_guard": True,
        "source_is_auxiliary_not_a_promotable_sentence": True, "central_and_external_provenance_required": True},
        "grammar_independence_scope": "Distinct source/target state classes, inventories, transition functions, and files; intentionally the same acoustic-duration relational schema and nearly the same syntactic skeleton. This is a narrow orthographic-semantic intersection, not broad grammar diversity.",
        "source_language": lattice_statistics(source), "target_language": lattice_statistics(target),
        "construction": construction, **{k: v for k, v in result.items() if k != "records"},
        "production_relexicalization_witnesses": witnesses, "exact_closures": len(result["records"]),
        "unique_exact_surfaces": len(exact), "independent_final_parse_calls": len(exact),
        "closure_reviews": closure_reviews, "exact_rejected_surfaces": sum(bool(row["failures"]) for row in closure_reviews),
        "next_endpoint_mismatch": mismatch,
        "distinct_repair_direction": "Replace the single left-prefix compound seam with bidirectional, variable-position resegmentation across different argument-attachment grammars. Require a genuine right-side cut change as well as a left one, so the right residual can change instead of remaining the fixed elk/emit boundary. Preserve whole-sentence event identity and target island pruning.",
        "rejections": dict(rejection_codes), "pending_external_review": pending, "promoted_candidates": [],
        "source_hashes": {path.name: sha256(path.read_bytes()).hexdigest() for path in paths},
        "scope": "Finite acoustic-duration grammar. Production shift witnesses are real matched paths, not palindrome or readability successes. No external novelty claim follows from this finite exhaustion."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-letters", type=int, default=100)
    parser.add_argument("--max-letters", type=int, default=240)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.min_letters, args.max_letters)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: report[key] for key in ("construction", "stats", "deepest", "exact_closures")}))
        print(json.dumps({"output": str(args.output), "production_shift_witnesses": len(report["production_relexicalization_witnesses"]),
                          "pending_external_review": len(report["pending_external_review"]), "promoted_candidates": 0}))
    else:
        print(json.dumps(report, indent=2))
