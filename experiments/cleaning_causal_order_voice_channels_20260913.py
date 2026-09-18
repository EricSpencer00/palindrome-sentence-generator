"""Causal order/voice alternation with structural coreference and free centers."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json
from math import prod
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.lexicon_indexed_semantic_edge_channels_20260913 import chart
from experiments.cleaning_causal_voice_validator_20260913 import parse_complete
from llm_palindrome.admission import mechanical_admission_checks

HUMANS = tuple("workers owners cooks chefs servers cleaners helpers attendants".split())
PLURAL_ARTIFACTS = tuple("ovens trays counters plates pans grills stoves tables boards bowls aprons towels".split())
SINGULAR_ARTIFACTS = tuple("oven tray counter plate pan grill stove table board bowl apron towel".split())
CLASSIFIERS = ("deli", "cafe", "kitchen")
ACTION = {"wash": ("washed", -1), "clean": ("cleaned", -1), "scrub": ("scrubbed", -1), "wipe": ("wiped", -1), "soil": ("soiled", 1)}
OBSERVATION = {"dirty": 1, "soiled": 2, "stained": 2, "dusty": 1, "clean": 0}
MODES = ("reason_last_active", "reason_last_passive", "condition_first_active", "condition_first_passive")


@dataclass(frozen=True)
class NP:
    words: tuple[str, ...]
    number: str
    kind: str


def human_nps():
    for head in HUMANS:
        yield NP((head,), "plural", "human")
        for word in CLASSIFIERS + ("tired", "careful", "skilled"):
            yield NP((word, head), "plural", "human")


def artifact_nps():
    for number, heads in (("plural", PLURAL_ARTIFACTS), ("singular", SINGULAR_ARTIFACTS)):
        for head in heads:
            for prefix in ((), *((word,) for word in CLASSIFIERS + ("small", "large", "old"))):
                if number == "plural":
                    yield NP(prefix + (head,), number, "artifact")
                else:
                    initial = prefix[0] if prefix else head
                    article = "an" if initial[0] in "aeiou" else "a"
                    for det in (article, "the"):
                        yield NP((det,) + prefix + (head,), number, "artifact")


def pro(np, case):
    return "it" if np.number == "singular" else "they" if case == "subject" else "them"


def be(np):
    return "is" if np.number == "singular" else "are"


def depth(first, last):
    a, b = "".join(first), "".join(last)
    matched = 0
    for i in range(min(len(a), len(b))):
        if a[i] != b[-i - 1]:
            break
        matched += 1
    return matched


def constructive_cause(verb, state):
    before = OBSERVATION[state]
    effect = ACTION[verb][1]
    after = max(0, before + effect)
    return {"before": before, "effect": effect, "after": after, "corrective": after < before}


def endpoints(mode, agent, patient, state):
    if mode == "reason_last_active":
        return agent.words, (pro(patient, "subject"), be(patient), state)
    if mode == "reason_last_passive":
        return patient.words, (pro(patient, "subject"), be(patient), state)
    first = ("because",) + patient.words + (be(patient), state)
    last = (pro(patient, "object"),) if mode == "condition_first_active" else ("by",) + agent.words
    return first, last


def discover_channels(min_pairs=5):
    if min_pairs < 5:
        raise ValueError("five actual character pairs are required")
    agents, patients = tuple(human_nps()), tuple(artifact_nps())
    selected, audit = [], []
    for mode in MODES:
        histogram = Counter()
        count = 0
        for agent, patient, state in product(agents, patients, OBSERVATION):
            first, last = endpoints(mode, agent, patient, state)
            matched = depth(first, last)
            histogram[matched] += 1
            if matched < min_pairs:
                continue
            for verb in ACTION:
                cause = constructive_cause(verb, state)
                if cause["corrective"]:
                    count += 1
                    selected.append({"mode": mode, "agent": agent, "patient": patient, "state": state,
                                     "verb": verb, "indexed_pairs": matched, "first_constituent": first,
                                     "last_constituent": last, "causal_effect": cause})
        audit.append({"mode": mode, "tested_constituent_bindings": sum(histogram.values()),
                      "indexed_depth_distribution": dict(histogram), "selected_channels": count,
                      "condition_first_endpoint_certificate": ({
                          "first_letter": "b", "plain_last_letters": ["m", "t"] if mode.endswith("active") else ["s"],
                          "relative_or_location_extension_last_letters": ["s"],
                          "reason": "Because and all installed action endings disagree at the outermost letter; no condition-first product is claimed eligible."
                      } if mode.startswith("condition_first") else None)})
    return selected, audit


def surfaces(channel):
    mode, agent, patient, verb, state = (channel[k] for k in ("mode", "agent", "patient", "verb", "state"))
    H = tuple((word,) for word in agent.words)
    P = tuple((word,) for word in patient.words)
    relative = (("who",), ("read", "review", "follow"), ("brief", "detailed", "clear"), ("instructions", "manuals", "guidelines"),
                ("from",), ("careful", "skilled"), ("supervisors", "managers", "inspectors"))
    location = (("in", "inside", "near"), ("small", "large", "quiet"), ("kitchens", "cafes", "workshops"))
    out = {}
    for add_relative, add_location in product((False, True), repeat=2):
        human = H + (relative if add_relative else ())
        pp = location if add_location else ()
        if mode == "reason_last_active":
            action = human + ((verb,),) + P + pp
            reason = (("because",), (pro(patient, "subject"),), (be(patient),), (state,))
            slots = action + reason
        elif mode == "reason_last_passive":
            action = P + ((be(patient),), (ACTION[verb][0],), ("by",)) + human + pp
            slots = action + (("because",), (pro(patient, "subject"),), (be(patient),), (state,))
        else:
            reason = (("because",),) + P + ((be(patient),), (state,))
            if mode == "condition_first_active":
                action = human + ((verb,), (pro(patient, "object"),)) + pp
            else:
                action = ((pro(patient, "subject"),), (be(patient),), (ACTION[verb][0],), ("by",)) + human + pp
            slots = reason + action
        out[f"relative_{int(add_relative)}_location_{int(add_location)}"] = slots
    return out


def run(min_letters=100, max_letters=240):
    channels, edge_audit = discover_channels()
    rows, pending = [], []
    total, depths, boundary_depths, maximum, rejection_codes, mode_stats = Counter(), Counter(), Counter(), Counter(), Counter(), Counter()
    exact_count = lexical_realizations = 0
    for identifier, channel in enumerate(channels):
        for name, slots in surfaces(channel).items():
            result = chart(slots)
            actual = result["deepest"]["depth"]
            eligible = actual >= 5
            mode_stats[channel["mode"]] += eligible
            total.update(result["stats"])
            depths.update(result["state_depth_distribution"])
            boundary_depths.update(result["boundary_depth_distribution"])
            maximum[actual] += 1
            lexical_realizations += prod(map(len, slots))
            exact_count += len(result["records"])
            for record in result["records"]:
                words = tuple(record["words"])
                parses = [row for row in parse_complete(words) if row["semantic_relation_valid"]]
                checks = mechanical_admission_checks(" ".join(words), min_letters=min_letters, max_letters=max_letters)
                failures = [name for name, value in checks.items() if not value]
                if not eligible:
                    failures.append("actual_edge_below_five")
                if not parses:
                    failures.append("independent_structural_causal_reparse_failed")
                if failures:
                    rejection_codes.update(failures)
                else:
                    pending.append({"tokens": words, "normalized": record["tape"], "letters": record["letters"],
                                    "normalized_sha256": sha256(record["tape"].encode()).hexdigest(),
                                    "mechanical_checks": checks, "structural_causal_parses": parses,
                                    "external_provenance": "required; unchecked", "promoted": False})
            rows.append({"channel": identifier, "mode": channel["mode"], "layout": name,
                         "actual_edge_eligible": eligible, "actual_matched_pairs": actual,
                         "length_range": [sum(min(map(len, slot)) for slot in slots), sum(max(map(len, slot)) for slot in slots)],
                         "lexical_realizations": prod(map(len, slots)), "exact_closures": len(result["records"]),
                         **{k: v for k, v in result.items() if k != "records"}})
    serial_channels = [{**row, "agent": {"words": row["agent"].words, "number": row["agent"].number},
                        "patient": {"words": row["patient"].words, "number": row["patient"].number}} for row in channels]
    validator = ROOT / "experiments/cleaning_causal_voice_validator_20260913.py"
    return {"status": "cleaning_causal_order_voice_constituent_channels", "config": {
        "min_letters": min_letters, "max_letters": max_letters, "required_actual_pairs": 5,
        "condition_first_and_reason_last_grammar": True, "active_and_passive_voice": True,
        "structural_coreference_case_number_type_role": True, "separate_semantic_reparser": True,
        "free_character_midpoint": True, "online_anti_island_guard": True,
        "same_doctors_hot_cod_inventory_expansion": False, "external_provenance_before_promotion": True},
        "grammar_inventory": {"human_heads": HUMANS, "artifact_plural_heads": PLURAL_ARTIFACTS,
                              "artifact_singular_heads": SINGULAR_ARTIFACTS, "classifiers": CLASSIFIERS,
                              "actions": ACTION, "contamination_states": OBSERVATION},
        "edge_audit": edge_audit, "channel_count": len(channels), "channels": serial_channels,
        "products": len(rows), "actual_eligible_products_by_mode": dict(mode_stats), "lexical_realizations": lexical_realizations,
        "stats": dict(total), "state_depth_distribution": dict(depths), "boundary_depth_distribution": dict(boundary_depths),
        "maximum_depth_distribution": dict(maximum), "rows": rows, "exact_closures": exact_count,
        "rejections": dict(rejection_codes), "pending_external_review": pending, "promoted_candidates": [], "states_exhausted": True,
        "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                       "independent_validator_sha256": sha256(validator.read_bytes()).hexdigest(),
                       "material": "new cleaning-domain lexical declarations, compositional active/passive causal grammar and structural antecedent binding"},
        "scope": "Finite grammar evidence only. Condition-first branches with incompatible outer letters remain explicitly ineligible. No reader or global novelty claim follows from parsing, state reduction or local catalogue absence."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-letters", type=int, default=100)
    parser.add_argument("--max-letters", type=int, default=240)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.min_letters, args.max_letters)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: result[key] for key in ("channel_count", "products", "actual_eligible_products_by_mode", "stats", "maximum_depth_distribution", "exact_closures")}))
        print(json.dumps({"output": str(args.output), "pending_external_review": len(result["pending_external_review"]), "promoted_candidates": 0}))
    else:
        print(json.dumps(result, indent=2))
