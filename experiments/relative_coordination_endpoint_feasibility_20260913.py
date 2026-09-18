"""Finite endpoint discovery only; no full palindrome product is constructed.

Source: [workers who repair [covers and printers]] inspect objects.
Target: [[workers who repair covers] and printers] inspect objects.
The source coordinates repairable machines inside the relative clause; the
target coordinates human tradespeople outside it.  These are distinct valid
attachment readings, not equivalent whole semantic graphs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import (has_forbidden_catalogue_endpoint_scaffold,
                                      is_catalogue_family_derivative)

# Independently declared source/target inventories and syntax below.  No
# catalogue phrase or anadrome pair supplies a production or a fixture.
SOURCE_WORKERS = ("bookbinders", "woodworkers", "metalworkers", "glassblowers", "shopkeepers", "storekeepers")
SOURCE_PATIENTS = ("toolboxes", "paintbrushes", "handcarts", "bookshelves", "worktables")
SOURCE_REPAIR = ("repair", "fix")
SOURCE_REPAIR_OBJECTS = ("covers", "lamps")
SOURCE_MACHINES = ("printers", "cutters")
SOURCE_ACTIONS = ("inspect", "move")

TARGET_WORKERS = (("store", "keepers"), ("shop", "keepers"), ("glass", "blowers"),
                  ("metal", "workers"), ("wood", "workers"), ("book", "binders"))
TARGET_PATIENTS = (("work", "tables"), ("book", "shelves"), ("hand", "carts"),
                   ("paint", "brushes"), ("tool", "boxes"))
TARGET_RELATIVE_VERBS = ("fix", "repair")
TARGET_ARTIFACTS = ("lamps", "covers")
TARGET_HUMAN_COORDINATES = ("cutters", "printers")
TARGET_MAIN_VERBS = ("move", "inspect")


def source_paths():
    for worker, verb, worn, artifact, machine, action, patient in product(
            SOURCE_WORKERS, SOURCE_REPAIR, ((), ("worn",)), SOURCE_REPAIR_OBJECTS,
            SOURCE_MACHINES, SOURCE_ACTIONS, SOURCE_PATIENTS):
        yield (worker, "who", verb) + worn + (artifact, "and", machine, action, patient)


def target_paths():
    for worker, repair, modifier, artifact, colleague, action, object_np in product(
            TARGET_WORKERS, TARGET_RELATIVE_VERBS, (("worn",), ()), TARGET_ARTIFACTS,
            TARGET_HUMAN_COORDINATES, TARGET_MAIN_VERBS, TARGET_PATIENTS):
        yield worker + ("who", repair) + modifier + (artifact, "and", colleague, action) + object_np


def parse_source(words):
    words = tuple(words)
    if len(words) not in {8, 9} or words[0] not in SOURCE_WORKERS or words[1] != "who" or words[2] not in SOURCE_REPAIR:
        return None
    i = 3
    if words[i] == "worn":
        i += 1
    if len(words) != i + 5 or words[i] not in SOURCE_REPAIR_OBJECTS or words[i + 1] != "and" or words[i + 2] not in SOURCE_MACHINES or words[i + 3] not in SOURCE_ACTIONS or words[i + 4] not in SOURCE_PATIENTS:
        return None
    return {"complete": True, "main_agents": [{"surface": words[0], "type": "human_trade"}],
            "relative_clause": {"subject": words[0], "predicate": words[2],
                                "coordinated_objects": [{"surface": words[i], "type": "repairable_artifact"},
                                                        {"surface": words[i + 2], "type": "repairable_machine"}]},
            "main_event": {"predicate": words[i + 3], "patient": words[i + 4], "patient_type": "workshop_artifact"},
            "coordination_attaches_to": "relative_clause_objects"}


def parse_target(words):
    words = tuple(words)
    if len(words) not in {10, 11} or words[:2] not in TARGET_WORKERS or words[2] != "who" or words[3] not in TARGET_RELATIVE_VERBS:
        return None
    position = 4
    if words[position] == "worn":
        position += 1
    if len(words) != position + 6 or words[position] not in TARGET_ARTIFACTS or words[position + 1] != "and" or words[position + 2] not in TARGET_HUMAN_COORDINATES or words[position + 3] not in TARGET_MAIN_VERBS or words[position + 4:] not in TARGET_PATIENTS:
        return None
    worker = " ".join(words[:2])
    return {"complete": True, "main_agents": [{"surface": worker, "type": "human_trade"},
                                               {"surface": words[position + 2], "type": "human_trade"}],
            "relative_clause": {"subject": worker, "predicate": words[3],
                                "object": {"surface": words[position], "type": "repairable_artifact"}},
            "main_event": {"predicate": words[position + 3], "patient": words[position + 4:], "patient_type": "workshop_artifact"},
            "coordination_attaches_to": "main_clause_subjects"}


def tape(words):
    return "".join(words)


def cuts(words):
    position, boundaries = 0, set()
    for word in words[:-1]:
        position += len(word)
        boundaries.add(position)
    return boundaries


def outer_anadrome_scaffold(words):
    """Conservatively exclude whole endpoint-unit reversals and plural variants."""
    def variants(value):
        return {value, value[:-1]} if value.endswith("s") and len(value) > 3 else {value}
    for width_left in (1, 2):
        for width_right in (1, 2):
            if width_left + width_right > len(words):
                continue
            for first in variants("".join(words[:width_left])):
                for last in variants("".join(words[-width_right:])):
                    if len(first) >= 3 and first == last[::-1]:
                        return True
    return False


def excluded(words, known):
    return (tape(words) in known or has_forbidden_catalogue_endpoint_scaffold(tuple(words))
            or is_catalogue_family_derivative(tuple(words)) or outer_anadrome_scaffold(words))


def endpoint_witness(source_words, target_words, pairs=6):
    a, b = tape(source_words), tape(target_words)
    if a != b:
        return None
    matched = 0
    for offset in range(min(pairs + 1, len(b) // 2)):
        if b[offset] != b[-offset - 1]:
            break
        matched += 1
    source_cuts, target_cuts = cuts(source_words), cuts(target_words)
    disagreements = source_cuts ^ target_cuts
    left = sorted(position for position in disagreements if 0 < position < pairs)
    right = sorted(len(b) - position for position in disagreements if 0 < len(b) - position < pairs)
    return {"matched_pairs_capped_at_seven": matched, "prefix": b[:pairs],
            "right_letters_read_inward": "".join(b[-i - 1] for i in range(min(pairs, len(b)))),
            "left_boundary_disagreements": left, "right_boundary_disagreements": right,
            "left_next_letter": b[pairs] if len(b) > 2 * pairs else None,
            "right_next_letter": b[-pairs - 1] if len(b) > 2 * pairs else None,
            "six_actual_pairs_and_both_shifts": matched >= pairs and bool(left) and bool(right)}


def discover(source_language, target_language, source_parser, target_parser, known=frozenset(), pairs=6, next_letters=6):
    if pairs < 6 or next_letters < 6:
        raise ValueError("the production gate cannot be relaxed below six pairs and six paired continuations")
    source_index, source_count, source_rejected = defaultdict(list), 0, 0
    for words in source_language:
        source_count += 1
        parse = source_parser(words)
        if parse is None or excluded(words, known):
            source_rejected += 1
            continue
        source_index[tape(words)].append((tuple(words), parse))
    target_count, target_rejected, derivation_pairs = 0, 0, 0
    surfaces, frontier_surfaces, all_rows = set(), set(), []
    matched_histogram, channels = Counter(), defaultdict(list)
    for words in target_language:
        target_count += 1
        parse = target_parser(words)
        if parse is None or excluded(words, known):
            target_rejected += 1
            continue
        for source_words, source_parse in source_index.get(tape(words), ()):
            derivation_pairs += 1
            surfaces.add(tuple(words))
            witness = endpoint_witness(source_words, words, pairs)
            matched_histogram[witness["matched_pairs_capped_at_seven"]] += 1
            row = {"source_tokens": source_words, "target_tokens": tuple(words), "source_parse": source_parse,
                   "target_parse": parse, "normalized": tape(words), "letters": len(tape(words)), **witness}
            all_rows.append(row)
            if witness["six_actual_pairs_and_both_shifts"]:
                frontier_surfaces.add(tuple(words))
                channels[witness["prefix"]].append(row)
    channel_rows, qualified_surfaces, qualified_pairs = [], set(), 0
    for prefix, witnesses in sorted(channels.items()):
        left = {row["left_next_letter"] for row in witnesses if row["left_next_letter"]}
        right = {row["right_next_letter"] for row in witnesses if row["right_next_letter"]}
        paired = {row["left_next_letter"] for row in witnesses if row["left_next_letter"] == row["right_next_letter"] and row["left_next_letter"]}
        qualified = min(len(left), len(right), len(paired)) >= next_letters
        channel_rows.append({"prefix": prefix, "actual_pairs": pairs, "derivation_pairs": len(witnesses),
                             "distinct_target_surfaces": len({row["target_tokens"] for row in witnesses}),
                             "left_next_letters": sorted(left), "right_next_letters": sorted(right),
                             "paired_next_letters": sorted(paired), "qualified": qualified})
        if qualified:
            qualified_pairs += len(witnesses)
            qualified_surfaces.update(row["target_tokens"] for row in witnesses)
    return {"source_paths_enumerated": source_count, "target_paths_enumerated": target_count,
            "source_paths_rejected": source_rejected, "target_paths_rejected": target_rejected,
            "same_tape_derivation_pairs": derivation_pairs, "distinct_same_tape_target_surfaces": len(surfaces),
            "actual_matched_pair_distribution": dict(matched_histogram),
            "six_pair_two_sided_target_surfaces": len(frontier_surfaces), "channels": channel_rows,
            "qualified_channels": sum(row["qualified"] for row in channel_rows),
            "qualified_derivation_pairs": qualified_pairs, "qualified_target_surfaces": len(qualified_surfaces),
            "enumerated_derivation_pairs": all_rows, "finite_enumeration_complete": True}


def run():
    known = frozenset(json.loads((ROOT / "data/known_palindromes.json").read_text()))
    result = discover(source_paths(), target_paths(), parse_source, parse_target, known)
    return {"status": "endpoint_feasibility_qualified" if result["qualified_channels"] else "endpoint_feasibility_unqualified",
            "family": "relative-object coordination versus main-subject coordination, with source closed compounds and target compositional NPs",
            "semantic_scope": "Both whole-sentence attachment readings are independently parsed; their repair objects and complete agent groups differ, so whole-graph equivalence is not claimed.",
            "config": {"required_actual_pairs": 6, "required_paired_continuation_letters": 6,
                       "both_endpoint_word_boundary_disagreements_required": True,
                       "catalogue_used_only_for_exclusion": True, "outer_anadrome_scaffolds_excluded": True},
            **result, "full_palindrome_product_compiled": False, "candidate_count": 0,
            "next_step": "Compile a whole-sentence product only after a production endpoint channel passes all gates; this unqualified inventory does not authorize that expansion." if not result["qualified_channels"] else "Qualified endpoint evidence exists; a separate full-product implementation may now be built.",
            "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run()
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: value for key, value in result.items() if key != "enumerated_derivation_pairs"}))
    else:
        print(json.dumps(result, indent=2))
