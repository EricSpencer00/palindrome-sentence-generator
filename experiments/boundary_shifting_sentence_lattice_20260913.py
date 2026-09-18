"""Variable-token whole-sentence lattice intersected with letter symmetry.

Word completions and unfinished prefixes are competing character edges.  No
slot number, provisional endpoint boundary, source sentence, reversed source,
or fixed midpoint constrains the output.  A semantic prefix state accompanies
every node; a separately authored grammar reparses each exact whole path once.
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
from experiments.art_conservation_sentence_validator_20260913 import parse_sentence
from llm_palindrome.admission import (REPEATABLE_FUNCTION_WORDS,
    has_forbidden_catalogue_endpoint_scaffold, mechanical_admission_checks)


@dataclass(frozen=True)
class ParseState:
    phase: str
    verb: str = ""
    defect: str = ""
    medium: str = ""


ACTORS = tuple("artists painters sculptors framers restorers conservators curators dealers traders collectors printmakers woodcarvers".split())
OBJECTS = {"art": "mixed", "artwork": "mixed", "paintings": "painted", "prints": "paper",
           "drawings": "paper", "posters": "paper", "woodcuts": "wood", "sculptures": "solid", "frames": "solid"}
OBSERVED = {"torn": ("tear", 2), "chipped": ("chip", 2), "faded": ("fading", 1), "damaged": ("damage", 2)}
OPERATIONS = {"restore": {"tear": -1, "chip": -1, "fading": -1, "damage": -1},
              "repair": {"tear": -1, "chip": -1, "damage": -1}, "mend": {"tear": -1}}
SUPPORTS = {"tear": {"paper", "painted", "mixed"}, "chip": {"solid", "wood", "painted", "mixed"},
            "fading": {"paper", "wood", "painted", "mixed"}, "damage": set(OBJECTS.values())}


def transitions(state):
    """Finite incremental semantic parser; transitions consume whole lexemes."""
    phase, verb, defect, medium = state.phase, state.verb, state.defect, state.medium
    def nxt(target, word, **features):
        return word, ParseState(target, features.get("verb", verb), features.get("defect", defect), features.get("medium", medium))
    if phase == "start":
        return [nxt("actor", w) for w in ACTORS] + [nxt("print_actor", "print"), nxt("wood_actor", "wood")]
    if phase in {"print_actor", "wood_actor"}:
        return [nxt("actor", "makers" if phase == "print_actor" else "carvers")]
    if phase == "actor":
        return [nxt("relative_verb", "who")] + [nxt("after_manner", w) for w in ("carefully", "patiently", "slowly")] + [nxt("defect", w, verb=w) for w in OPERATIONS]
    if phase == "relative_verb":
        return [nxt("relative_object", w) for w in ("study", "consult", "compare")]
    if phase == "relative_object":
        return [nxt("relative_head", w) for w in ("old", "detailed", "archival")] + [nxt("relative_done", w) for w in ("records", "photographs", "catalogues")]
    if phase == "relative_head":
        return [nxt("relative_done", w) for w in ("records", "photographs", "catalogues")]
    if phase == "relative_done":
        return [nxt("source", "from")] + [nxt("after_manner", w) for w in ("carefully", "patiently", "slowly")] + [nxt("defect", w, verb=w) for w in OPERATIONS]
    if phase == "source":
        return [nxt("source_head", w) for w in ("local", "national", "private")] + [nxt("after_relative", w) for w in ("museums", "archives", "collections")]
    if phase == "source_head":
        return [nxt("after_relative", w) for w in ("museums", "archives", "collections")]
    if phase == "after_relative":
        return [nxt("after_manner", w) for w in ("carefully", "patiently", "slowly")] + [nxt("defect", w, verb=w) for w in OPERATIONS]
    if phase == "after_manner":
        return [nxt("defect", w, verb=w) for w in OPERATIONS]
    if phase == "defect":
        return [nxt("patient", word, defect=word) for word, (prop, initial) in OBSERVED.items()
                if prop in OPERATIONS[verb] and max(0, initial + OPERATIONS[verb][prop]) < initial]
    if phase in {"patient", "patient_head"}:
        prop = OBSERVED[defect][0]
        options = [nxt("patient_done", w, medium=m) for w, m in OBJECTS.items() if m in SUPPORTS[prop]]
        if "mixed" in SUPPORTS[prop]:
            options.append(nxt("art_compound", "art", medium="mixed"))
        if "wood" in SUPPORTS[prop]:
            options.append(nxt("wood_compound", "wood", medium="wood"))
        if phase == "patient":
            options.extend(nxt("patient_head", w) for w in ("red", "blue", "green", "black"))
        return options
    if phase in {"art_compound", "wood_compound"}:
        return [nxt("patient_done", "work" if phase == "art_compound" else "cuts")]
    if phase == "patient_done":
        return [nxt("location_modifier", w) for w in ("in", "inside", "near")]
    if phase == "location_modifier":
        return [nxt("location_head", w) for w in ("local", "quiet", "private")]
    if phase == "location_head":
        return [nxt("done", w) for w in ("museums", "galleries", "studios")]
    return []


@dataclass(frozen=True)
class Node:
    parser_state: object
    lexical_prefix: str

    @property
    def boundary(self):
        return not self.lexical_prefix


@dataclass(frozen=True)
class Edge:
    source: int
    target: int
    char: str
    completed_word: str = ""


@dataclass
class Lattice:
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    start: int
    accepting: frozenset[int]


def compile_lattice(initial, lexical_transitions, accepting):
    """Compile a lexical-prefix NFA, sharing prefixes but never erasing parses."""
    nodes, ids, edges = [], {}, set()
    def node(state, prefix=""):
        key = Node(state, prefix)
        if key not in ids:
            ids[key] = len(nodes)
            nodes.append(key)
        return ids[key]
    start = node(initial)
    pending, seen, ends = [initial], set(), set()
    while pending:
        state = pending.pop()
        if state in seen:
            continue
        seen.add(state)
        if accepting(state):
            ends.add(node(state))
        for word, target in lexical_transitions(state):
            if not word or not word.isascii() or not word.isalpha() or not word.islower():
                raise ValueError("lexical entries must be lowercase ASCII words")
            pending.append(target)
            current = node(state)
            for offset, char in enumerate(word, 1):
                last = offset == len(word)
                following = node(target) if last else node(state, word[:offset])
                edges.add(Edge(current, following, char, word if last else ""))
                current = following
    return Lattice(tuple(nodes), tuple(sorted(edges, key=lambda e: (e.source, e.target, e.char, e.completed_word))), start, frozenset(ends))


def toy_lattice(sentences):
    """Mechanism fixture only; never part of the production lexical inventory."""
    links = defaultdict(set)
    accepting = set()
    for sentence in sentences:
        prefix = ()
        for word in sentence:
            following = prefix + (word,)
            links[prefix].add((word, following))
            prefix = following
        accepting.add(prefix)
    return compile_lattice((), lambda state: sorted(links[state]), accepting.__contains__)


def lattice_statistics(lattice):
    """Exact finite-language counts/ranges, with an explicit acyclicity check."""
    outgoing = defaultdict(list)
    for edge in lattice.edges:
        outgoing[edge.source].append(edge)
    active = set()

    @lru_cache(maxsize=None)
    def summarize(node):
        if node in active:
            raise ValueError("a bounded acyclic semantic grammar is required")
        active.add(node)
        rows = [(1, 0, 0, 0, 0)] if node in lattice.accepting else []
        for edge in outgoing[node]:
            count, min_letters, max_letters, min_words, max_words = summarize(edge.target)
            if count:
                rows.append((count, min_letters + 1, max_letters + 1,
                             min_words + bool(edge.completed_word), max_words + bool(edge.completed_word)))
        active.remove(node)
        return ((sum(row[0] for row in rows), min(row[1] for row in rows), max(row[2] for row in rows),
                 min(row[3] for row in rows), max(row[4] for row in rows)) if rows else (0, 0, 0, 0, 0))

    paths, low, high, fewest, most = summarize(lattice.start)
    return {"acyclic": True, "accepted_lexical_paths": paths,
            "letter_length_range": [low, high], "word_count_range": [fewest, most]}


def complete_words(path, initial_boundary=True):
    words = tuple(edge.completed_word for edge in path if edge.completed_word)
    return words if initial_boundary else words[1:]


def known_island(words):
    for start in range(len(words)):
        for end in range(start + 2, len(words) + 1):
            tape = "".join(words[start:end])
            if tape == tape[::-1]:
                return {"start_word": start, "end_word": end, "tokens": words[start:end]}
    return None


def search_lattice(lattice, probe_pairs=5):
    """Outside-in chart; current nodes carry both lexical and semantic states."""
    if probe_pairs < 1:
        raise ValueError("positive endpoint probe required")
    language = lattice_statistics(lattice)
    outgoing, incoming = defaultdict(list), defaultdict(list)
    for edge in lattice.edges:
        outgoing[edge.source].append(edge)
        incoming[edge.target].append(edge)

    @lru_cache(maxsize=None)
    def reachable(left, right):
        return left == right or any(reachable(edge.target, right) for edge in outgoing[left])

    @lru_cache(maxsize=None)
    def word_range(left, right):
        if left == right:
            return 0, 0
        options = []
        for edge in outgoing[left]:
            if reachable(edge.target, right):
                lo, hi = word_range(edge.target, right)
                options.append((lo + bool(edge.completed_word), hi + bool(edge.completed_word)))
        return (min(row[0] for row in options), max(row[1] for row in options)) if options else (10**9, -1)

    stack = [(lattice.start, end, (), (), 0, (), ()) for end in sorted(lattice.accepting)]
    stats, depths, boundaries, channels = Counter(), Counter(), Counter(), defaultdict(list)
    records, witnesses = [], []
    deepest = {"depth": -1}
    while stack:
        left, right, prefix, suffix, depth, left_cuts, right_cuts = stack.pop()
        if not reachable(left, right):
            continue
        stats["states"] += 1
        depths[depth] += 1
        ln, rn = lattice.nodes[left], lattice.nodes[right]
        if depth:
            boundaries[f"left:{depth}"] += ln.boundary
            boundaries[f"right:{depth}"] += rn.boundary
            boundaries[f"joint:{depth}"] += ln.boundary and rn.boundary
        if depth > deepest["depth"]:
            deepest = {"depth": depth, "left_parser_state": repr(ln.parser_state), "right_parser_state": repr(rn.parser_state),
                       "left_lexical_prefix": ln.lexical_prefix, "right_lexical_prefix": rn.lexical_prefix,
                       "left_boundary_depths": left_cuts, "right_boundary_depths": right_cuts}
        # At a proper pair of token boundaries, any multiword completion would
        # itself be a forbidden palindrome.  Only an empty center or a single
        # palindromic function word can survive; do not discard that exception.
        if depth and ln.boundary and rn.boundary and left != right:
            lo, hi = word_range(left, right)
            function_centers = []
            if lo <= 1:
                todo = [(left, ())]
                while todo:
                    node_id, path = todo.pop()
                    for edge in outgoing[node_id]:
                        next_path = path + (edge,)
                        if edge.completed_word:
                            w = edge.completed_word
                            if edge.target == right and w in REPEATABLE_FUNCTION_WORDS and w == w[::-1]:
                                function_centers.append(next_path)
                        elif reachable(edge.target, right):
                            todo.append((edge.target, next_path))
            stats["proper_island_prunes" if lo >= 2 else "single_or_multiword_center_prunes"] += 1
            if len(witnesses) < 8:
                witnesses.append({"kind": "forced_inner_span", "depth": depth, "remaining_word_range": [lo, hi],
                                  "left_parser_state": repr(ln.parser_state), "right_parser_state": repr(rn.parser_state),
                                  "allowed_function_centers": len(function_centers)})
            for middle in function_centers:
                path = prefix + middle + suffix
                words = complete_words(path)
                tape = "".join(edge.char for edge in path)
                records.append({"words": words, "tape": tape, "letters": len(tape), "matched_depth": depth,
                                "center_characters": len(middle), "center_kind": "function_word_exception",
                                "left_boundary_depths": left_cuts, "right_boundary_depths": right_cuts})
            continue
        left_words = complete_words(prefix)
        right_words = complete_words(suffix, rn.boundary)
        if known_island(left_words) or known_island(right_words):
            stats["observed_island_prunes"] += 1
            continue
        # A reversed completion edge already identifies its whole final lexeme,
        # even while its initial letters remain unconsumed.  That commitment is
        # sufficient for the narrow catalogue endpoint exclusion.
        if has_forbidden_catalogue_endpoint_scaffold(left_words + complete_words(suffix)):
            stats["catalogue_scaffold_prunes"] += 1
            continue
        if depth == probe_pairs:
            key = "".join(edge.char for edge in prefix)
            channels[key].append({"left_boundary_depths": left_cuts, "right_boundary_depths": right_cuts,
                                  "left_lexical_prefix": ln.lexical_prefix, "right_lexical_prefix": rn.lexical_prefix,
                                  "left_parser_state": repr(ln.parser_state), "right_parser_state": repr(rn.parser_state)})
            stats["actual_endpoint_eligible_states"] += 1
        middles = [()] if left == right else []
        middles.extend((edge,) for edge in outgoing[left] if edge.target == right)
        for middle in middles:
            path = prefix + middle + suffix
            if any(a.target != b.source for a, b in zip(path, path[1:])):
                raise AssertionError("disconnected lexical path")
            words = complete_words(path)
            tape = "".join(edge.char for edge in path)
            if tape != "".join(words) or tape != tape[::-1]:
                raise AssertionError("invalid exact whole-sentence closure")
            records.append({"words": words, "tape": tape, "letters": len(tape), "matched_depth": depth,
                            "center_characters": len(middle), "center_kind": "even" if not middle else "odd",
                            "left_boundary_depths": left_cuts, "right_boundary_depths": right_cuts})
        advances = 0
        for first in outgoing[left]:
            for last in incoming[right]:
                if first.char == last.char and reachable(first.target, last.source):
                    next_depth = depth + 1
                    stack.append((first.target, last.source, prefix + (first,), (last,) + suffix, next_depth,
                                  left_cuts + ((next_depth,) if lattice.nodes[first.target].boundary else ()),
                                  right_cuts + ((next_depth,) if lattice.nodes[last.source].boundary else ())))
                    advances += 1
        if not advances and not middles:
            stats["character_dead_frontiers"] += 1
    # Channel equivalence ignores word cuts: retain them only as diagnostics.
    # A final token may straddle any provisional cut in another analysis.
    channel_rows = []
    for tape, variants in sorted(channels.items()):
        cuts_left = {tuple(row["left_boundary_depths"]) for row in variants}
        cuts_right = {tuple(row["right_boundary_depths"]) for row in variants}
        crossings = []
        for side, alternatives in (("left", cuts_left), ("right", cuts_right)):
            for provisional in sorted(alternatives):
                for final in sorted(alternatives):
                    for cut in set(provisional) - set(final):
                        if cut < probe_pairs:
                            crossings.append({"side": side, "provisional_cut": cut,
                                              "provisional_boundaries": provisional, "relexicalized_boundaries": final})
        channel_rows.append({"tape": tape, "actual_pairs": probe_pairs, "states": len(variants),
                             "left_boundary_analyses": sorted(cuts_left), "right_boundary_analyses": sorted(cuts_right),
                             "crossing_witnesses": crossings, "sample_states": variants[:3]})
    for record in records:
        key = record["tape"][:probe_pairs]
        row = next((row for row in channel_rows if row["tape"] == key), None)
        record["provisional_boundary_crossings"] = [] if row is None else [w for w in row["crossing_witnesses"]
            if w["provisional_cut"] not in record[f'{w["side"]}_boundary_depths']]
    return {"language": language, "stats": dict(stats), "state_depth_distribution": dict(depths),
            "boundary_depth_distribution": {k: v for k, v in boundaries.items() if v},
            "deepest": deepest, "channels": channel_rows, "records": records,
            "prune_witnesses": witnesses, "states_exhausted": True}


def run(min_letters=100, max_letters=240):
    lattice = compile_lattice(ParseState("start"), transitions, lambda s: s.phase in {"patient_done", "done"})
    result = search_lattice(lattice, probe_pairs=5)
    rejected, pending = Counter(), []
    # Each exact surface is sent once to the independent whole-sentence parser.
    # The lattice never imports the validator's vocabulary or parse states.
    unique = {tuple(row["words"]): row for row in result["records"]}
    for words, row in unique.items():
        parses = parse_sentence(words)
        valid = [p for p in parses if p["semantic_relation_valid"]]
        checks = mechanical_admission_checks(" ".join(words), min_letters=min_letters, max_letters=max_letters)
        failures = [key for key, passed in checks.items() if not passed]
        if not valid:
            failures.append("independent_whole_sentence_semantics")
        if row["matched_depth"] < 5:
            failures.append("fewer_than_five_actual_pairs")
        if failures:
            rejected.update(failures)
        else:
            pending.append({"tokens": words, "normalized": row["tape"], "letters": row["letters"],
                            "mechanical_checks": checks, "independent_semantic_parses": valid,
                            "provenance": "external review required; unchecked", "promoted": False})
    validator = ROOT / "experiments/art_conservation_sentence_validator_20260913.py"
    return {"status": "boundary_shifting_whole_sentence_lexical_lattice", "config": {
        "min_letters": min_letters, "max_letters": max_letters, "required_actual_pairs": 5,
        "variable_token_count": True, "lexical_prefix_state_online": True, "semantic_parser_state_online": True,
        "endpoint_channel_key_is_letters_only": True, "provisional_boundaries_are_not_constraints": True,
        "whole_sentence_independent_reparse_once": True, "free_even_odd_midpoint": True,
        "online_anti_island_guard": True, "source_tape_reflection": False, "mirrored_clause_construction": False,
        "external_provenance_required_before_promotion": True},
        "inventory": {"actors": ACTORS, "objects": OBJECTS, "observations": OBSERVED, "operations": OPERATIONS,
                      "spaced_compound_lexical_alternations": [["printmakers", ["print", "makers"]], ["woodcarvers", ["wood", "carvers"]],
                                                               ["artwork", ["art", "work"]], ["woodcuts", ["wood", "cuts"]]]},
        "lattice": {"nodes": len(lattice.nodes), "character_edges": len(lattice.edges),
                    "semantic_states": len({n.parser_state for n in lattice.nodes}), "accepting_states": len(lattice.accepting),
                    "lexical_prefix_nodes": sum(not n.boundary for n in lattice.nodes)},
        **{k: v for k, v in result.items() if k != "records"}, "exact_closures": len(result["records"]),
        "unique_exact_surfaces": len(unique), "independent_parse_calls": len(unique),
        "rejections": dict(rejected), "pending_external_review": pending, "promoted_candidates": [],
        "source_hashes": {"generator": sha256(Path(__file__).read_bytes()).hexdigest(),
                          "independent_validator": sha256(validator.read_bytes()).hexdigest()},
        "scope": "Finite art-conservation grammar only. Boundary-shift mechanism fixtures are excluded from search. A semantic reduction is a qualitative event check, never a readability or originality certificate."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-letters", type=int, default=100)
    parser.add_argument("--max-letters", type=int, default=240)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.min_letters, args.max_letters)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: report[key] for key in ("lattice", "stats", "deepest", "exact_closures", "unique_exact_surfaces")}))
        print(json.dumps({"output": str(args.output), "pending_external_review": len(report["pending_external_review"]), "promoted_candidates": 0}))
    else:
        print(json.dumps(report, indent=2))
