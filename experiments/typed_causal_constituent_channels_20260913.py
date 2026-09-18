"""Whole-sentence causal grammar with discovered five-pair constituent edges.

The reason clause states a property of the main action's target contents.
Generation and a separately declared parser both compute the direction of
the action's corrective effect. No known sentence, question/reply pair,
source-tape reversal or clause-boundary midpoint is installed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from itertools import product
import json
from math import prod
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.lexicon_indexed_semantic_edge_channels_20260913 import chart
from experiments.causal_temperature_validator_20260913 import parse_complete
from llm_palindrome.admission import mechanical_admission_checks

ACTORS = tuple("cooks chefs servers parents doctors nurses workers helpers caretakers attendants volunteers hosts guests".split())
ACTOR_MODIFIERS = ("careful", "skilled", "tired")
CONTAINERS = tuple("bowls cups pots pans plates trays jars".split())
CONTAINER_MODIFIERS = ("small", "large", "deep")
FOODS = tuple("soup stew sauce curry rice pasta cod trout salmon chicken beef pork beans peas carrots potatoes noodles grits porridge pudding".split())
ACTION_EFFECT = {"cool": -1, "chill": -1, "warm": 1, "heat": 1}
OBSERVED_STATE = {"hot": 1, "cold": -1}


def generator_causal_effect(action, state):
    initial = OBSERVED_STATE[state]
    predicted = initial + ACTION_EFFECT[action]
    return {"initial": initial, "effect": ACTION_EFFECT[action], "predicted": predicted,
            "corrective": abs(predicted) < abs(initial)}


def first_constituents():
    for actor in ACTORS:
        yield (actor,)
        for modifier in ACTOR_MODIFIERS:
            yield (modifier, actor)


def last_constituents():
    for temperature, food in product(OBSERVED_STATE, FOODS):
        yield (temperature, food)
        yield ("very", temperature, food)


def matching_depth(first_words, last_words):
    left, right = "".join(first_words), "".join(last_words)
    count = 0
    # This is a spelling compatibility test of independently generated
    # constituents. It never creates source or target tokens by reversal.
    for i in range(min(len(left), len(right))):
        if left[i] != right[len(right) - 1 - i]:
            break
        count += 1
    return count


def discover_channels(min_pairs=5):
    if min_pairs < 5:
        raise ValueError("causal edge eligibility requires at least five actual letter pairs")
    index = {}
    for last in last_constituents():
        tape = "".join(last)
        key = "".join(tape[-i - 1] for i in range(min_pairs)) if len(tape) >= min_pairs else None
        if key is not None:
            index.setdefault(key, []).append(last)
    channels = []
    for first in first_constituents():
        key = "".join(first)[:min_pairs]
        for last in index.get(key, ()):
            state = last[-2]
            for action in ACTION_EFFECT:
                effect = generator_causal_effect(action, state)
                if effect["corrective"]:
                    channels.append({"first": first, "action": action, "last": last,
                                     "indexed_pairs": matching_depth(first, last), "causal_effect": effect})
    return channels


def layouts(channel):
    first = tuple((word,) for word in channel["first"])
    verb = ((channel["action"],),)
    reason = (("because",), ("their",), ("contents",), ("include",)) + tuple((word,) for word in channel["last"])
    container = (CONTAINERS,)
    modified_container = (CONTAINER_MODIFIERS,) + container
    manner = (("carefully", "slowly", "gently"),)
    relative = (("who",), ("read", "reviewed", "checked"), ("brief", "detailed", "old"), ("recipes", "menus", "instructions"))
    source = (("from",), ("skilled", "experienced", "careful"), ("cooks", "chefs", "hosts"))
    location = (("in", "inside", "near"), ("quiet", "large", "old"), ("kitchens", "cafeterias"))
    return {
        "causal": first + verb + container + reason,
        "modified_target": first + verb + modified_container + reason,
        "manner_causal": first + manner + verb + modified_container + reason,
        "located_causal": first + verb + modified_container + location + reason,
        "relative_causal": first + relative + verb + container + reason,
        "relative_source_causal": first + relative + source + manner + verb + modified_container + reason,
        "relative_source_located_causal": first + relative + source + manner + verb + modified_container + location + reason,
    }


def run(min_letters=100, max_letters=240):
    rows, pending = [], []
    totals, depths, boundary_depths, maximum_depths, rejection_codes = Counter(), Counter(), Counter(), Counter(), Counter()
    exact_count = eligible_products = lexical_realizations = 0
    channels = discover_channels()
    for identifier, channel in enumerate(channels):
        for name, slots in layouts(channel).items():
            result = chart(slots)
            actual_pairs = result["deepest"]["depth"]
            eligible = actual_pairs >= 5
            eligible_products += eligible
            lexical_realizations += prod(map(len, slots))
            totals.update(result["stats"])
            depths.update(result["state_depth_distribution"])
            boundary_depths.update(result["boundary_depth_distribution"])
            maximum_depths[actual_pairs] += 1
            exact_count += len(result["records"])
            for record in result["records"]:
                words = tuple(record["words"])
                parses = parse_complete(words)
                valid = [parse for parse in parses if parse["semantic_relation_valid"]]
                checks = mechanical_admission_checks(" ".join(words), min_letters=min_letters, max_letters=max_letters)
                codes = [key for key, value in checks.items() if not value]
                if not eligible:
                    codes.append("actual_edge_depth_below_five")
                if not valid:
                    codes.append("independent_causal_reparse_failed")
                if codes:
                    rejection_codes.update(codes)
                else:
                    pending.append({"tokens": words, "normalized": record["tape"], "letters": record["letters"],
                                    "normalized_sha256": sha256(record["tape"].encode()).hexdigest(),
                                    "mechanical_checks": checks, "independent_causal_parses": valid,
                                    "external_provenance": "required; not checked", "promoted": False})
            rows.append({"channel": identifier, "layout": name, "actual_edge_eligible": eligible,
                         "actual_matched_pairs": actual_pairs,
                         "length_range": [sum(min(map(len, slot)) for slot in slots), sum(max(map(len, slot)) for slot in slots)],
                         "lexical_realizations": prod(map(len, slots)), "exact_closures": len(result["records"]),
                         **{k: v for k, v in result.items() if k != "records"}})
    validator = ROOT / "experiments/causal_temperature_validator_20260913.py"
    return {"status": "typed_causal_full_constituent_channel_search", "config": {
        "min_letters": min_letters, "max_letters": max_letters, "required_actual_outer_pairs": 5,
        "one_whole_sentence_causal_grammar": True, "free_character_midpoint": True,
        "full_first_and_last_constituent_index": True, "separately_declared_causal_reparse": True,
        "causal_validation_by_predicted_state_change": True, "online_anti_island_guard": True,
        "question_reply": False, "table_membership_as_semantics": False,
        "fixed_clause_boundary_center": False, "source_tape_reflection": False,
        "external_provenance_required_before_promotion": True},
        "inventory_counts": {"actors": len(ACTORS), "first_constituents": len(list(first_constituents())),
                             "last_constituents": len(list(last_constituents())), "actions": len(ACTION_EFFECT),
                             "containers": len(CONTAINERS), "food_forms": len(FOODS)},
        "channels": channels, "channel_count": len(channels), "products": len(rows), "actual_eligible_products": eligible_products,
        "lexical_realizations": lexical_realizations, "stats": dict(totals), "state_depth_distribution": dict(depths),
        "boundary_depth_distribution": dict(boundary_depths), "maximum_depth_distribution": dict(maximum_depths),
        "rows": rows, "exact_closures": exact_count, "rejections": dict(rejection_codes),
        "pending_external_review": pending, "promoted_candidates": [], "states_exhausted": True,
        "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                       "independent_validator_sha256": sha256(validator.read_bytes()).hexdigest(),
                       "material": "authored individual lexemes, temperature-state signs, action effects and compositional grammar; no source sentences or catalogue seeds"},
        "scope": "Finite grammar exhaustion only. The qualitative thermal model checks the direction of a corrective causal relation; it does not certify physical necessity, reader acceptance or novelty."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-letters", type=int, default=100)
    parser.add_argument("--max-letters", type=int, default=240)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.min_letters, args.max_letters)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: report[key] for key in ("inventory_counts", "channel_count", "products", "actual_eligible_products", "stats", "maximum_depth_distribution", "exact_closures")}))
        print(json.dumps({"output": str(args.output), "pending_external_review": len(report["pending_external_review"]), "promoted_candidates": 0}))
    else:
        print(json.dumps(report, indent=2))
