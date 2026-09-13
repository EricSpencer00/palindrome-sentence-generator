"""Matched lexicon × representation ablation over the existing character DAG.

This is deliberately one runner, rather than four search programs.  The four
cells differ only in a frozen lexical source (L0/L1) and a representation
(hard layouts/soft word lattices).  All use the same character-level product,
free one-character-or-boundary centre, online boundary-island rule and central
mechanical admission gate.  It is a closure-accounting experiment: it never
publishes, ranks semantically, or promotes a generated surface.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from functools import lru_cache
from hashlib import sha256
import heapq
import json
from pathlib import Path
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.boundary_disjoint_semantic_chart_20260913 import (
    FRAMES, LAYOUTS, boundary_island, domains, lexical_boundaries,
)
from experiments.whole_text_palindrome_product_20260913 import compile_slots, replay_path
from llm_palindrome.admission import REPEATABLE_FUNCTION_WORDS, mechanical_admission_checks
from llm_palindrome.safe_vocab import is_allowed


SEED = 20260913
DEFAULT_EXPANSIONS = 100_000
MIN_LETTERS, MAX_LETTERS = 100, 240
HARD_BUCKET_SIZE = 16
# A scheduler guard, not a vocabulary filter.  It bounds work per state while
# preserving the complete S-lattice in the frozen inventory and DAG.
SCHEDULER_EDGES_PER_CHARACTER = 32
LONG_LAYOUTS = ("relative_two_events", "relative_object_origin")
POS_FOR_ROLE = {
    "subject": "NOUN", "object": "NOUN", "location": "NOUN", "source": "NOUN",
    "relative_object": "NOUN", "relative_location": "NOUN",
    "verb": "VERB", "relative_verb": "VERB",
    "subject_modifier": "ADJ", "object_modifier": "ADJ", "location_modifier": "ADJ",
    "relative_object_modifier": "ADJ", "relative_location_modifier": "ADJ", "source_modifier": "ADJ",
    "manner": "ADV",
    "location_prep": "ADP", "relative_location_prep": "ADP", "origin_prep": "ADP",
    "relative_marker": "PRON",
}


def _sha(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _word_hash(words: Iterable[str]) -> str:
    return sha256("\n".join(words).encode()).hexdigest()


def _rank(seed: int, value: Any) -> str:
    return sha256(f"{seed}:".encode() + repr(value).encode()).hexdigest()


def _actual_templates() -> tuple[dict[str, Any], ...]:
    """Freeze only terminals actually present in boundary_disjoint frames."""
    templates = []
    for frame in FRAMES:
        inventory = domains(frame)
        for layout in LONG_LAYOUTS:
            roles = LAYOUTS[layout]
            slots = tuple(tuple(dict.fromkeys(inventory[role])) for role in roles)
            # compile_slots is the authoritative reachable-terminal source.
            grammar = compile_slots(slots)
            terminals = tuple(sorted({edge.completed_word for edge in grammar.edges if edge.completed_word}))
            templates.append({"frame": frame.name, "layout": layout, "roles": roles,
                              "slots": slots, "word_count": len(roles),
                              "reachable_terminal_sha256": _word_hash(terminals),
                              "reachable_terminal_count": len(terminals)})
    return tuple(templates)


ACTUAL_TEMPLATES = _actual_templates()
HARD_WORD_COUNTS = tuple(sorted({row["word_count"] for row in ACTUAL_TEMPLATES}))


@lru_cache(maxsize=1)
def independent_lexicon() -> dict[str, Any]:
    """Select L1 before any endpoint or DAG is inspected.

    Brown and wordfreq are evidence filters only.  If either optional resource
    is absent, the frozen evidence records that fact instead of silently
    widening the lexical source.
    """
    from llm_palindrome.lexicon import load_lexicon

    base = sorted(load_lexicon(str(ROOT / "data" / "lexicon.txt")))
    evidence: dict[str, Any] = {"lexicon_sha256": _word_hash(base), "safe_vocab": True,
                                "brown_universal_pos": "unavailable", "wordfreq_min_zipf": 3.0}
    tags: dict[str, set[str]] = defaultdict(set)
    try:
        from nltk.corpus import brown
        for word, tag in brown.tagged_words(tagset="universal"):
            word = word.casefold()
            if word.isascii() and word.isalpha():
                tags[word].add(tag)
        evidence["brown_universal_pos"] = "available"
        evidence["brown_tagged_token_count"] = sum(len(v) for v in tags.values())
    except Exception as error:  # no corpus is not permission to invent POS evidence
        evidence["brown_error"] = type(error).__name__
    try:
        from wordfreq import zipf_frequency
        frequency = lambda word: float(zipf_frequency(word, "en"))
        evidence["wordfreq"] = "available"
    except Exception as error:
        frequency = lambda word: 3.0
        evidence["wordfreq"] = "unavailable"
        evidence["wordfreq_error"] = type(error).__name__

    buckets: dict[str, list[str]] = defaultdict(list)
    for word in base:
        # Six letters keeps every 19-word L1 language within the fixed range.
        if len(word) < 6 or not is_allowed(word) or frequency(word) < 3.0:
            continue
        for tag in tags.get(word, ()):
            if tag in {"NOUN", "VERB", "ADJ", "ADV", "ADP", "PRON"}:
                buckets[tag].append(word)
    # No fallback that weakens the evidence: absent required POS means an empty L1.
    # ``buckets`` is the complete frozen L1 source used by S1.  H1 alone uses
    # the predeclared, hash-balanced fixed choice below to keep hard layouts
    # tractable; it is chosen before any endpoint or reverse character match.
    frozen = {tag: tuple(sorted(set(words))) for tag, words in buckets.items()}
    hard_buckets = {tag: tuple(sorted(words, key=lambda word: _rank(SEED, ("H1", tag, word)))[:HARD_BUCKET_SIZE])
                    for tag, words in frozen.items()}
    evidence["selection_rule"] = "lexicon ∩ safe_vocab ∩ Brown-universal-POS ∩ wordfreq>=3; len>=6"
    evidence["selection_timing"] = "completed before grammar construction, endpoint eligibility, or reverse-character matching"
    evidence["bucket_counts"] = {tag: len(words) for tag, words in frozen.items()}
    evidence["bucket_hashes"] = {tag: _word_hash(words) for tag, words in frozen.items()}
    evidence["hard_choice"] = {"size_per_pos": HARD_BUCKET_SIZE,
                                "method": "deterministic hash order independent of endpoints"}
    evidence["hard_bucket_hashes"] = {tag: _word_hash(words) for tag, words in hard_buckets.items()}
    evidence["selection_sha256"] = _sha(frozen)
    return {"buckets": frozen, "hard_buckets": hard_buckets, "evidence": evidence}


def _l1_slots(roles: tuple[str, ...], *, hard: bool) -> tuple[tuple[str, ...], ...]:
    source = independent_lexicon()
    buckets = source["hard_buckets"] if hard else source["buckets"]
    # Determiners are deliberately not smuggled in: long L1 slots use only
    # independently Brown-attested categories, and fail closed if unavailable.
    return tuple(buckets.get(POS_FOR_ROLE.get(role, "NOUN"), ()) for role in roles)


def build_cells() -> dict[str, dict[str, Any]]:
    """Build all cells before endpoint expansion; H and S share counts."""
    l1 = independent_lexicon()
    l0_hard = tuple({k: v for k, v in row.items() if k != "frame"} for row in ACTUAL_TEMPLATES)
    l1_hard = tuple({"roles": row["roles"], "slots": _l1_slots(row["roles"], hard=True),
                     "word_count": row["word_count"]} for row in ACTUAL_TEMPLATES)

    def soft_from(hard: tuple[dict[str, Any], ...], *, full_l1: dict[str, tuple[str, ...]] | None = None) -> tuple[dict[str, Any], ...]:
        pooled: dict[str, tuple[str, ...]] = {}
        if full_l1 is not None:
            pooled = {tag: tuple(words) for tag, words in full_l1.items()}
        else:
            for tag in set(POS_FOR_ROLE.values()):
                words = {word for row in hard for role, slot in zip(row["roles"], row["slots"])
                         if POS_FOR_ROLE.get(role, "NOUN") == tag for word in slot}
                # S0 keeps every actual POS-compatible terminal.
                pooled[tag] = tuple(sorted(words))
        specs = []
        for count in HARD_WORD_COUNTS:
            source = next(row for row in hard if row["word_count"] == count)
            # The lattice retains only slot-local POS/word-count structure. It
            # has no frame identity and cannot feed a semantic decision.
            specs.append({"word_count": count,
                          "slots": tuple(pooled[POS_FOR_ROLE.get(role, "NOUN")] for role in source["roles"]),
                          "representation": "soft_lattice"})
        return tuple(specs)

    return {
        "H0": {"lexicon": "L0_actual_boundary_disjoint_terminals", "representation": "hard_layout",
               "grammars": l0_hard, "lexicon_evidence": {"templates": len(ACTUAL_TEMPLATES),
               "terminal_hashes": [row["reachable_terminal_sha256"] for row in ACTUAL_TEMPLATES]}},
        "H1": {"lexicon": "L1_independent_lexicon_safe_brown_wordfreq", "representation": "hard_layout",
               "grammars": l1_hard, "lexicon_evidence": l1["evidence"]},
        "S0": {"lexicon": "L0_actual_boundary_disjoint_terminals", "representation": "soft_lattice",
               "grammars": soft_from(l0_hard), "lexicon_evidence": {"templates": len(ACTUAL_TEMPLATES),
               "terminal_hashes": [row["reachable_terminal_sha256"] for row in ACTUAL_TEMPLATES]}},
        "S1": {"lexicon": "L1_independent_lexicon_safe_brown_wordfreq", "representation": "soft_lattice",
               "grammars": soft_from(l1_hard, full_l1=l1["buckets"]), "lexicon_evidence": l1["evidence"]},
    }


def _boundary_content_center(slots: tuple[tuple[str, ...], ...], left: int, right: int,
                             boundary: dict[int, int], depth: int) -> bool:
    if depth == 0 or left not in boundary or right not in boundary:
        return False
    if boundary[right] - boundary[left] != 1:
        return False
    return not any(word == word[::-1] and word in REPEATABLE_FUNCTION_WORDS
                   for word in slots[boundary[left]])


def _state_key(state: tuple) -> tuple:
    left, right, prefix, suffix, *_ = state
    return left, right, tuple(edge.char for edge in prefix), tuple(edge.char for edge in suffix)


def semantic_review_payload(rendered: str, tape_sha256: str) -> dict[str, str]:
    """The sole semantic-review handoff: no arm, lexicon, layout or score."""
    return {"review_id": sha256((rendered + tape_sha256).encode()).hexdigest(),
            "surface": rendered, "instruction": "Assess the surface only; do not infer its source."}


def chart_kernel(slots: tuple[tuple[str, ...], ...], *, grammar_id: str, seed: int,
                 state_budget: int, collect_closures: bool = True) -> dict[str, Any]:
    """Deterministic, generic character-DAG chart with complete closure audit."""
    if not slots or any(not slot for slot in slots):
        return {"states": 0, "pending": 0, "exhausted": True, "unavailable": True,
                "reason": "frozen_lexicon_has_empty_required_pos_bucket", "closures": [],
                "depth_residual_survival": {}, "coaccessible_next_letters": {}, "terminations": {},
                "eligible_endpoint_vocabularies": {"start": [], "final": []},
                "effective_endpoint_vocabularies": {"left": [], "right": []}}
    grammar = compile_slots(slots)
    boundary = lexical_boundaries(grammar)
    outgoing, incoming = defaultdict(list), defaultdict(list)
    for edge in grammar.edges:
        outgoing[edge.source].append(edge)
        incoming[edge.target].append(edge)

    # Every character edge stays within a lexical slot; a completed-word edge
    # advances exactly one slot.  This compact layer index replaces a cached
    # transitive-closure set (which is prohibitively large for full S1).
    layer = {grammar.start: 0}
    todo = [grammar.start]
    while todo:
        node = todo.pop()
        for edge in outgoing[node]:
            next_layer = edge.slot + 1 if edge.completed_word is not None else layer[node]
            prior = layer.get(edge.target)
            if prior is None:
                layer[edge.target] = next_layer
                todo.append(edge.target)
            elif prior != next_layer:
                raise AssertionError("character DAG has incompatible lexical layers")

    @lru_cache(maxsize=None)
    def can_reach(source: int, target: int) -> bool:
        """Exact only inside one lexical layer; later layers reconverge."""
        if source == target:
            return True
        if layer[source] > layer[target]:
            return False
        if layer[source] < layer[target]:
            return True
        return any(can_reach(edge.target, target) for edge in outgoing[source])

    initial = (grammar.start, grammar.end, (), (), 0)
    heap = [(_rank(seed, (grammar_id, _state_key(initial))), 0, initial)]
    seen, serial = set(), 1
    depth, next_letters, terminations = defaultdict(Counter), defaultdict(Counter), Counter()
    endpoint_left, endpoint_right = set(), set()
    closures = []
    while heap and len(seen) < state_budget:
        _, _, state = heapq.heappop(heap)
        key = _state_key(state)
        if key in seen:
            continue
        seen.add(key)
        left, right, prefix, suffix, matched_depth = state
        depth[matched_depth]["entered"] += 1
        island = boundary_island(left, right, boundary, matched_depth)
        if island:
            depth[matched_depth]["island_pruned"] += 1
            terminations["online_proper_multiword_island"] += 1
            continue
        if _boundary_content_center(slots, left, right, boundary, matched_depth):
            depth[matched_depth]["content_center_pruned"] += 1
            terminations["forbidden_single_content_center"] += 1
            continue

        first_by_char, last_by_char = defaultdict(list), defaultdict(list)
        for first in outgoing[left]:
            first_by_char[first.char].append(first)
        for last in incoming[right]:
            last_by_char[last.char].append(last)
        viable_chars = set(first_by_char).intersection(last_by_char)
        # Sampling is strictly a state-budget scheduler.  All POS-compatible
        # words remain installed in the lattice and reported as eligible.
        pairs = []
        for char in viable_chars:
            firsts = sorted(first_by_char[char], key=lambda edge: _rank(seed, (grammar_id, left, "L", edge)))[:SCHEDULER_EDGES_PER_CHARACTER]
            lasts = sorted(last_by_char[char], key=lambda edge: _rank(seed, (grammar_id, right, "R", edge)))[:SCHEDULER_EDGES_PER_CHARACTER]
            pairs.extend((first, last) for first in firsts for last in lasts
                         if can_reach(first.target, last.source))
        for char in viable_chars:
            next_letters[matched_depth][char] += 1
        if matched_depth == 0:
            endpoint_left.update(word for word in slots[0] if word[0] in viable_chars)
            endpoint_right.update(word for word in slots[-1] if word[-1] in viable_chars)

        middles = [()] if left == right else []
        middles.extend((edge,) for edge in outgoing[left] if edge.target == right)
        if middles:
            for middle in middles:
                replay = replay_path(grammar, prefix + middle + suffix)
                if not replay["ok"] or not replay["exact"]:
                    raise AssertionError("character DAG emitted an invalid closure")
                terminations["closure"] += 1
                rendered = " ".join(replay["words"]).capitalize() + "."
                tape_hash = sha256(replay["tape"].encode()).hexdigest()
                derivation = tuple((edge.source, edge.target, edge.char, edge.slot) for edge in prefix + middle + suffix)
                checks = mechanical_admission_checks(rendered, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
                closure = {"derivation_id": _sha(derivation), "tape_id": tape_hash,
                           "render_id": sha256(rendered.encode()).hexdigest(), "letters": replay["letters"],
                           "word_count": len(replay["words"]), "exact": True,
                           "identity": {"derivation_replays_to_tape": True,
                                        "render_normalizes_to_tape": True,
                                        "tape_sha256": tape_hash},
                           "admission": checks, "admitted": all(checks.values()),
                           "semantic_review": semantic_review_payload(rendered, tape_hash),
                           "promotion": "forbidden; diagnostic closure only"}
                if collect_closures:
                    closures.append(closure)
        if not pairs and not middles:
            depth[matched_depth]["dead"] += 1
            terminations["character_dead_frontier"] += 1
        for first, last in pairs:
            depth[matched_depth]["survived"] += 1
            child = (first.target, last.source, prefix + (first,), (last,) + suffix, matched_depth + 1)
            child_key = _state_key(child)
            if child_key not in seen:
                heapq.heappush(heap, (_rank(seed, (grammar_id, child_key)), serial, child))
                serial += 1
    truncated = bool(heap)
    return {"states": len(seen), "pending": len(heap), "exhausted": not truncated, "unavailable": False,
            "scheduler": {"edges_per_character_side": SCHEDULER_EDGES_PER_CHARACTER,
                          "can_suppress_installed_lexemes": True,
                          "scope": "budgeted traversal order only; not a lattice-vocabulary filter"},
            "closures": closures, "depth_residual_survival": {str(k): dict(v) for k, v in depth.items()},
            "coaccessible_next_letters": {str(k): dict(sorted(v.items())) for k, v in next_letters.items()},
            "terminations": dict(terminations),
            "eligible_endpoint_vocabularies": {"start": sorted(slots[0]), "final": sorted(slots[-1])},
            "effective_endpoint_vocabularies": {"left": sorted(endpoint_left), "right": sorted(endpoint_right)}}


def _merge_counter_dicts(rows: Iterable[dict[str, Any]], key: str) -> dict[str, dict[str, int]]:
    merged: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        for depth, counts in row[key].items():
            merged[depth].update(counts)
    return {depth: dict(counts) for depth, counts in sorted(merged.items(), key=lambda item: int(item[0]))}


def run(*, expansions_per_cell: int = DEFAULT_EXPANSIONS, seed: int = SEED) -> dict[str, Any]:
    if expansions_per_cell < 1:
        raise ValueError("expansions_per_cell must be positive")
    cells = build_cells()
    results: dict[str, Any] = {}
    for cell_id, cell in cells.items():
        rows, remaining, index = [], expansions_per_cell, 0
        # Stable round-robin stops one hard layout from owning the cell budget.
        while remaining and index < len(cell["grammars"]):
            spec = cell["grammars"][index]
            share = max(1, remaining // (len(cell["grammars"]) - index))
            row = chart_kernel(tuple(spec["slots"]), grammar_id=f"{cell_id}:{index}", seed=seed,
                               state_budget=share)
            rows.append(row)
            remaining -= row["states"]
            index += 1
        closures = [closure for row in rows for closure in row["closures"]]
        eligible_start = sorted({word for row in rows for word in row["eligible_endpoint_vocabularies"]["start"]})
        eligible_final = sorted({word for row in rows for word in row["eligible_endpoint_vocabularies"]["final"]})
        effective_left = sorted({word for row in rows for word in row.get("effective_endpoint_vocabularies", {}).get("left", [])})
        effective_right = sorted({word for row in rows for word in row.get("effective_endpoint_vocabularies", {}).get("right", [])})
        terminations = Counter()
        for row in rows:
            terminations.update(row["terminations"])
        # Cell output intentionally keeps experiment arm/template metadata out
        # of every semantic payload; there is no semantic ranking function.
        results[cell_id] = {"lexicon": cell["lexicon"], "representation": cell["representation"],
                            "expansion_budget": expansions_per_cell, "unique_canonical_state_expansions": sum(row["states"] for row in rows),
                            "budget_exhausted": remaining == 0,
                            "exploration_scope": {
                                "lattice_inventory": "complete frozen POS-compatible source" if cell["representation"] == "soft_lattice" else "frozen hard-layout source",
                                "scheduler_edges_per_character_side": SCHEDULER_EDGES_PER_CHARACTER,
                                "scheduler_can_suppress_installed_lexemes": True,
                                "coverage_claim": "budgeted character-DAG exploration; not all-lexeme coverage"},
                            # This means the priority frontier was empty under
                            # the stated scheduler, never that all L1 words or
                            # all unscheduled character transitions were tried.
                            "state_space_exhausted": all(row["exhausted"] for row in rows),
                            "frozen_lexicon_evidence": cell["lexicon_evidence"],
                            "hard_layout_word_counts": HARD_WORD_COUNTS,
                            "eligible_endpoint_vocabularies": {"start": eligible_start, "final": eligible_final},
                            "effective_endpoint_vocabularies": {"left": effective_left, "right": effective_right},
                            "depth_residual_survival": _merge_counter_dicts(rows, "depth_residual_survival"),
                            "coaccessible_next_letters": _merge_counter_dicts(rows, "coaccessible_next_letters"),
                            "mutually_exclusive_terminations": dict(terminations),
                            "closure_audit": closures,
                            "closure_identity_accounting": {"closures": len(closures),
                                "unique_derivations": len({x["derivation_id"] for x in closures}),
                                "unique_tapes": len({x["tape_id"] for x in closures}),
                                "unique_renders": len({x["render_id"] for x in closures}),
                                "all_identity_mappings_verified": all(all(x["identity"].values()) for x in closures)},
                            "semantic_ranking": "none; review payload is surface-only", "promotion": "forbidden"}
    return {"status": "complete_matched_lexicon_representation_ablation", "config": {
                "seed": seed, "unique_canonical_state_expansions_per_cell": expansions_per_cell,
                "budget_policy": "expand unique canonical states until the fixed cap or scheduled-frontier exhaustion",
                "scheduler_policy": f"at most {SCHEDULER_EDGES_PER_CHARACTER} deterministic edges per character-side; this bounds state work only, never an S-lattice inventory, and can suppress installed lexemes from traversal",
                "letter_range": [MIN_LETTERS, MAX_LETTERS], "cells": ["H0", "H1", "S0", "S1"],
                "free_character_center": True, "online_boundary_disjoint_anti_island": True,
                "central_mechanical_admission": True, "semantic_rules_use_templates": False,
                "candidate_promotion": "forbidden"}, "cells": results,
            "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                           "actual_template_sha256": _sha(ACTUAL_TEMPLATES)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expansions-per-cell", type=int, default=DEFAULT_EXPANSIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    result = run(expansions_per_cell=args.expansions_per_cell, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "states": {key: value["unique_canonical_state_expansions"]
          for key, value in result["cells"].items()}, "closures": {key: len(value["closure_audit"])
          for key, value in result["cells"].items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
