"""Typed complete sentences intersected with a boundary-disjoint palindrome chart.

Only individual authored lexemes enter generation. The midpoint is a live
character-graph meeting, never a preassembled word or phrase. At any proper
matched frontier whose two nodes are word boundaries, two or more remaining
lexical slots would necessarily form a prohibited palindromic island. Such a
state is pruned before its interior is lexicalized. The scheduler then returns
to the remaining lexical and constituent-layout alternatives.

This is a finite search, not a natural-language completeness or readability
claim. Output text is rendered only after independent typed reparsing and the
unchanged central mechanical gate. Human and external provenance review remain
necessary for any survivor.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from hashlib import sha256
import json
from math import prod
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.whole_text_palindrome_product_20260913 import compile_slots, replay_path
from llm_palindrome.admission import REPEATABLE_FUNCTION_WORDS, mechanical_admission_checks


@dataclass(frozen=True)
class Frame:
    name: str
    subjects: tuple[str, ...]
    verbs: tuple[str, ...]
    objects: tuple[str, ...]
    object_type: str
    object_number: str
    subject_modifiers: tuple[str, ...]
    object_modifiers: tuple[str, ...]
    locations: tuple[str, ...]
    sources: tuple[str, ...]


# Every verb is a present-tense transitive base form; subjects are plural.
# Each frame licenses the full Cartesian product, not a high-scoring sample.
FRAMES = (
    Frame("writing", ("writers", "authors", "editors"), ("read", "draft", "edit", "revise"),
          ("reports", "letters", "essays", "notes"), "text", "plural", ("careful", "skilled", "tired"),
          ("short", "detailed", "brief"), ("offices", "studios", "libraries"), ("authors", "editors")),
    Frame("cooking", ("cooks", "chefs", "parents"), ("stir", "heat", "serve"),
          ("soups", "sauces", "stews"), "food", "plural", ("careful", "skilled", "tired"),
          ("hot", "spicy", "thick"), ("kitchens", "cafeterias"), ("cooks", "chefs")),
    Frame("growing", ("farmers", "gardeners", "growers"), ("plant", "grow", "sell"),
          ("plants", "herbs", "flowers", "seedlings"), "plant", "plural", ("careful", "skilled", "tired"),
          ("large", "small", "healthy"), ("gardens", "nurseries", "greenhouses"), ("farmers", "growers")),
    Frame("manufacturing", ("workers", "makers", "artisans"), ("stamp", "press", "shape"),
          ("mats", "sheets", "panels"), "material", "plural", ("careful", "skilled", "tired"),
          ("flat", "soft", "thin"), ("factories", "workshops"), ("makers", "artisans")),
    Frame("transport", ("drivers", "workers", "porters"), ("move", "push", "pull"),
          ("carts", "trucks", "wagons"), "vehicle", "plural", ("careful", "skilled", "tired"),
          ("empty", "heavy", "small"), ("depots", "stations", "yards"), ("drivers", "workers")),
    Frame("mechanics", ("mechanics", "technicians", "workers"), ("repair", "inspect", "test"),
          ("engines", "motors", "pumps"), "machine", "plural", ("careful", "skilled", "tired"),
          ("old", "damaged", "small"), ("garages", "factories", "workshops"), ("mechanics", "technicians")),
    Frame("fish_inspection", ("doctors", "scientists", "researchers"), ("inspect", "test", "study"),
          ("cod", "trout", "herring"), "fish", "invariant_plural", ("careful", "skilled", "tired"),
          ("fresh", "large", "small"), ("laboratories", "harbors", "ports"), ("fishers", "anglers")),
)

# PP movement and noun attachment are real constituent alternatives. No
# lexical choice is frozen as a prefix, suffix, center, or complete control.
LAYOUTS = {
    "bare": ("subject", "verb", "object"),
    "subject_modifier": ("subject_modifier", "subject", "verb", "object"),
    "object_modifier": ("subject", "verb", "object_modifier", "object"),
    "preverb_manner": ("subject", "manner", "verb", "object"),
    "event_location_final": ("subject", "verb", "object", "location_prep", "location"),
    "event_location_front": ("location_prep", "location", "subject", "verb", "object"),
    "object_origin": ("subject", "verb", "object", "origin_prep", "source"),
    "expanded_event": ("subject_modifier", "subject", "manner", "verb", "object_modifier", "object", "location_prep", "location_modifier", "location"),
    "relative_two_events": ("subject_modifier", "subject", "relative_marker", "relative_verb", "relative_object_modifier", "relative_object", "relative_location_prep", "relative_location_modifier", "relative_location", "manner", "verb", "object_modifier", "object", "location_prep", "location_modifier", "location"),
    "relative_object_origin": ("subject_modifier", "subject", "relative_marker", "relative_verb", "relative_object_modifier", "relative_object", "relative_location_prep", "relative_location_modifier", "relative_location", "manner", "verb", "object_modifier", "object", "origin_prep", "source_modifier", "source", "location_prep", "location_modifier", "location"),
    "front_location_relative": ("location_prep", "location_modifier", "location", "subject_modifier", "subject", "relative_marker", "relative_verb", "relative_object_modifier", "relative_object", "relative_location_prep", "relative_location_modifier", "relative_location", "manner", "verb", "object_modifier", "object", "origin_prep", "source_modifier", "source"),
}


def domains(frame: Frame) -> dict[str, tuple[str, ...]]:
    inventory = {
        "subject": frame.subjects, "verb": frame.verbs, "object": frame.objects,
        "subject_modifier": frame.subject_modifiers, "object_modifier": frame.object_modifiers,
        "location": frame.locations, "source": frame.sources,
        "manner": ("carefully", "slowly", "silently"),
        "location_prep": ("in", "inside"), "origin_prep": ("from",),
        "location_modifier": ("quiet", "large"),
        "relative_marker": ("who",), "source_modifier": frame.subject_modifiers,
    }
    for role in ("verb", "object_modifier", "object", "location_prep", "location_modifier", "location"):
        inventory["relative_" + role] = inventory[role]
    return inventory


def typed_parse(words: tuple[str, ...]) -> list[dict]:
    """Reparse the full token sequence without any construction path or frame ID.

    The parser independently recognizes the finite syntax alternatives, then
    unifies subject plurality, verb valency and object selection with a frame.
    It does not accept a successful character-path replay as a syntax witness.
    """
    parses = []
    for frame in FRAMES:
        inventory = domains(frame)
        for layout, roles in LAYOUTS.items():
            if len(words) != len(roles):
                continue
            if not all(word in inventory[role] for word, role in zip(words, roles)):
                continue
            assignment = dict(zip(roles, words))
            parses.append({
                "frame": frame.name, "layout": layout,
                "subject": {"word": assignment["subject"], "pos": "NNS", "number": "plural", "type": "person"},
                "predicate": {"word": assignment["verb"], "pos": "VB", "tense": "present", "valency": "transitive"},
                "object": {"word": assignment["object"], "type": frame.object_type, "number": frame.object_number},
                "attachments": (["event_location"] if "location" in assignment else []) + (["object_origin"] if "source" in assignment else []),
                "relative_event": ({"subject": assignment["subject"], "verb": assignment["relative_verb"], "object": assignment["relative_object"],
                                    "subject_shared_with_main_clause": True, "valency": "transitive", "object_type": frame.object_type,
                                    "location": assignment.get("relative_location")} if "relative_verb" in assignment else None),
                "agreement_ok": True, "valency_ok": True, "selection_ok": True, "complete": True,
            })
    return parses


def lexical_boundaries(grammar) -> dict[int, int]:
    """Map graph nodes to token offsets, including both full-text endpoints."""
    boundary = {grammar.start: 0}
    for edge in grammar.edges:
        if edge.completed_word is not None:
            offset = edge.slot + 1
            if edge.target in boundary and boundary[edge.target] != offset:
                raise AssertionError("word boundary maps to incompatible token offsets")
            boundary[edge.target] = offset
    return boundary


def boundary_island(left: int, right: int, boundary: dict[int, int], depth: int) -> dict | None:
    """A matched proper frontier encloses a palindromic token span.

    Equal *depth* matters, not equal number of words. Empty and one-word
    interiors are handled separately; the initial complete-text span is exempt.
    """
    if depth == 0 or left not in boundary or right not in boundary:
        return None
    i, j = boundary[left], boundary[right]
    if j - i >= 2:
        return {"matched_depth": depth, "left_token_offset": i, "right_token_offset": j,
                "interior_token_count": j - i, "reason": "forced_proper_multiword_palindromic_island"}
    return None


def boundary_chart(slots: tuple[tuple[str, ...], ...], *, max_states: int | None = None,
                   prune_islands: bool = True, prune_content_center: bool = True) -> dict:
    grammar = compile_slots(slots)
    boundary = lexical_boundaries(grammar)
    outgoing, incoming = defaultdict(list), defaultdict(list)
    for edge in grammar.edges:
        outgoing[edge.source].append(edge)
        incoming[edge.target].append(edge)

    @lru_cache(maxsize=None)
    def reachable(node):
        return frozenset({node}.union(*(reachable(edge.target) for edge in outgoing[node])))

    reachable(grammar.start)
    # State explicitly carries matched character depth and each side's
    # historical lexical-completion depths. No split word is preselected.
    stack = [(grammar.start, grammar.end, (), (), 0, (), ())]
    stats = Counter()
    records, prune_witnesses = [], []
    deepest = None
    seen_records = set()
    while stack and (max_states is None or stats["states"] < max_states):
        left, right, prefix, suffix, depth, left_depths, right_depths = stack.pop()
        stats["states"] += 1
        if deepest is None or depth > deepest["matched_depth"]:
            deepest = {"matched_depth": depth, "left_boundary_depths": left_depths,
                       "right_boundary_depths": right_depths,
                       "left_token_offset": boundary.get(left), "right_token_offset": boundary.get(right)}
        witness = boundary_island(left, right, boundary, depth) if prune_islands else None
        if witness:
            stats["proper_island_prunes"] += 1
            if len(prune_witnesses) < 8:
                prune_witnesses.append({**witness, "left_boundary_depths": left_depths,
                                        "right_boundary_depths": right_depths})
            continue
        if (prune_content_center and depth > 0 and left in boundary and right in boundary
                and boundary[right] - boundary[left] == 1):
            center_choices = slots[boundary[left]]
            if not any(word == word[::-1] and word in REPEATABLE_FUNCTION_WORDS for word in center_choices):
                stats["forbidden_single_content_center_prunes"] += 1
                continue

        middles = [()] if left == right else []
        middles.extend((edge,) for edge in outgoing[left] if edge.target == right)
        for middle in middles:
            replay = replay_path(grammar, prefix + middle + suffix)
            if not replay["ok"] or not replay["exact"]:
                raise AssertionError("invalid palindrome chart closure")
            key = tuple(replay["words"])
            if key not in seen_records:
                seen_records.add(key)
                records.append({**replay, "matched_depth": depth, "center_characters": len(middle),
                                "left_boundary_depths": left_depths, "right_boundary_depths": right_depths})

        advanced = False
        for first in outgoing[left]:
            for last in incoming[right]:
                if first.char != last.char or last.source not in reachable(first.target):
                    continue
                advanced = True
                next_depth = depth + 1
                ld = left_depths + ((next_depth,) if first.target in boundary else ())
                rd = right_depths + ((next_depth,) if last.source in boundary else ())
                stack.append((first.target, last.source, prefix + (first,), (last,) + suffix,
                              next_depth, ld, rd))
        if not advanced and not middles:
            stats["character_dead_frontiers"] += 1
    return {"stats": dict(stats), "records": records, "prune_witnesses": prune_witnesses,
            "deepest": deepest, "states_exhausted": not stack, "pending_states": len(stack)}


def run(*, min_letters: int = 100, max_letters: int = 240, max_states: int | None = None) -> dict:
    rows, survivors = [], []
    total = Counter()
    for frame in FRAMES:
        inventory = domains(frame)
        for layout, roles in LAYOUTS.items():
            slots = tuple(inventory[role] for role in roles)
            result = boundary_chart(slots, max_states=max_states)
            total.update(result["stats"])
            rejected = Counter()
            for rec in result["records"]:
                words = tuple(rec["words"])
                parses = typed_parse(words)
                # This internal tape is passed to admission; rejected strings
                # are never rendered or emitted in the result artifact.
                checks = mechanical_admission_checks(" ".join(words), min_letters=min_letters, max_letters=max_letters)
                codes = [key for key, value in checks.items() if not value]
                if not parses:
                    codes.append("independent_typed_reparse_failed")
                if codes:
                    rejected.update(codes)
                    continue
                rendered = " ".join(words).capitalize() + "."
                if layout in ("event_location_front", "front_location_relative"):
                    split = 2 if layout == "event_location_front" else 3
                    rendered = " ".join(words[:split]).capitalize() + ", " + " ".join(words[split:]) + "."
                final_checks = mechanical_admission_checks(rendered, min_letters=min_letters, max_letters=max_letters)
                if not all(final_checks.values()):
                    raise AssertionError("rendering changed admission")
                survivors.append({"rendered": rendered, "normalized": rec["tape"], "letters": rec["letters"],
                                  "normalized_sha256": sha256(rec["tape"].encode()).hexdigest(),
                                  "exact": True, "typed_parses": parses, "mechanical_checks": final_checks,
                                  "reader_status": "unreviewed", "external_provenance": "unchecked; no originality claim"})
            rows.append({"frame": frame.name, "layout": layout, "roles": roles,
                         "lexical_realizations": prod(map(len, slots)),
                         "length_range": [sum(min(map(len, slot)) for slot in slots), sum(max(map(len, slot)) for slot in slots)],
                         "exact_closure_count": len(result["records"]), "rejections": dict(rejected),
                         **{key: value for key, value in result.items() if key != "records"}})
    return {"status": "boundary_disjoint_semantic_chart", "config": {
        "min_letters": min_letters, "max_letters": max_letters, "max_states_per_layout": max_states,
        "complete_typed_nns_vb_object": True, "modifier_and_attachment_alternatives": True,
        "all_finite_lexical_realizations": True, "free_character_midpoint": True,
        "actual_matched_boundary_depths": True, "online_proper_multiword_island_prune": True,
        "independent_complete_reparse": True, "central_admission_before_rendering": True,
        "corpus_catalogue_or_palindrome_seed_generation": False},
        "grammar_inventory": [asdict(frame) for frame in FRAMES], "layouts": LAYOUTS,
        "lexical_realizations": sum(row["lexical_realizations"] for row in rows),
        "stats": dict(total), "rows": rows, "survivors": survivors,
        "states_exhausted": all(row["states_exhausted"] for row in rows),
        "repair_operator": "A forced island abandons its interior and returns to other lexical paths; the outer scheduler then explores modifier, PP movement, and object-attachment layouts with different word-boundary positions.",
        "invariant": "At a proper equal-character-depth frontier, two remaining lexical boundaries enclose a palindrome; an interior spanning at least two words is forbidden.",
        "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                       "material": "task-authored individual lexemes, typed event frames, and constituent layouts"},
        "scope": "Exhaustion concerns only the reported finite grammar. Exactness, parsing and local catalogue absence do not establish readable prose or global novelty."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-letters", type=int, default=100)
    parser.add_argument("--max-letters", type=int, default=240)
    parser.add_argument("--max-states", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(min_letters=args.min_letters, max_letters=args.max_letters, max_states=args.max_states)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
        print(json.dumps({"output": str(args.output), "states_exhausted": result["states_exhausted"],
                          "lexical_realizations": result["lexical_realizations"], "stats": result["stats"],
                          "survivors": len(result["survivors"])}))
    else:
        print(text)
