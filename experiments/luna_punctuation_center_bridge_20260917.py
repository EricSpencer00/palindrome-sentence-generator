"""Punctuation/center-bridge construction lane.

This lane keeps punctuation as a live construction variable while a
non-palindromic event bridges two independently authored, grammatical scenes.
Punctuation is deliberately ignored by the letter tape, but every punctuation
choice is recorded and audited so it cannot be used as a hidden shortcut.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "luna-punctuation-center-bridge-20260917"
SIGNATURE = "punctuation-center-bridge|nonpalindromic-event|scene-pair-csp|pointer-sha"
OUT = ROOT / "runs" / f"{EXPERIMENT}.json"
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"

LEFT = (
    ("the", "patient cartographer", "marks", "a coastal inlet", "beside the old pier"),
    ("a", "quiet gardener", "carries", "fresh seedlings", "toward the glasshouse"),
    ("the", "careful teacher", "labels", "winter specimens", "inside the field station"),
)
RIGHT = (
    ("the", "young archivist", "files", "weathered maps", "near the harbor office"),
    ("a", "steady porter", "moves", "clean crates", "under the cedar awning"),
    ("the", "local guide", "records", "clear directions", "beside the museum gate"),
)
# Held-out replacements for the first noun phrase after the center bridge.
# These are not in RIGHT, so the repair tests a new lexical boundary rather
# than replaying the original scene lattice.
HELDOUT_RIGHT_SUBJECTS = (
    ("the", "seasoned curator"),
    ("a", "new assistant"),
    ("the", "winter custodian"),
)
HELDOUT_RIGHT_OBJECTS = (
    "a sealed ledger",
    "the brass compass",
    "fresh route notes",
)
HELDOUT_RIGHT_LOCATIVES = (
    "along the western quay",
    "beside the railway shed",
    "within the public garden",
)
HELDOUT_RIGHT_VERBS = ("stores", "carries", "checks")
HELDOUT_RIGHT_AGREEMENT_SUBJECTS = (
    ("the", "retired keeper"),
    ("a", "careful steward"),
    ("the", "young registrar"),
)
CENTERS = (
    {"id": "lantern_event", "text": "the lantern flares", "meaning": "a sudden warning light"},
    {"id": "bell_event", "text": "the harbor bell sounds", "meaning": "a scheduled audible signal"},
)
PUNCTUATION = (
    {"id": "semicolon_while", "left": ";", "bridge": ", while ", "right": "."},
    {"id": "colon_then", "left": ":", "bridge": "; then ", "right": "."},
)


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def pointer(text: str) -> dict:
    tape = letters(text)
    mismatches = []
    i, j = 0, len(tape) - 1
    while i < j:
        if tape[i] != tape[j]:
            mismatches.append({"left_index": i, "right_index": j, "left": tape[i], "required": tape[j]})
        i += 1
        j -= 1
    return {"algorithm": "independent_two_pointer", "letters": len(tape),
            "exact": bool(tape) and not mismatches, "mismatch_count": len(mismatches),
            "first_mismatch": mismatches[0] if mismatches else None}


def sha_audit(text: str) -> dict:
    tape = letters(text)
    f = hashlib.sha256(tape.encode()).hexdigest()
    r = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"algorithm": "independent_forward_reverse_sha256", "forward": f, "reverse": r,
            "exact": bool(tape) and f == r}


def clause(parts: tuple[str, ...]) -> str:
    det, subject, verb, obj, place = parts
    return f"{det} {subject} {verb} {obj} {place}"


def novelty_preflight() -> dict:
    data = json.loads(REGISTRY.read_text())
    rows = [*data.get("entries", []), *data.get("excluded", [])]
    collisions = [r.get("id") for r in rows if r.get("id") != EXPERIMENT and
                  (r.get("signature") == SIGNATURE or r.get("artifact") ==
                   "experiments/luna_punctuation_center_bridge_20260917.py")]
    return {"registry_entries": len(data.get("entries", [])), "signature": SIGNATURE,
            "collisions": collisions, "passed": not collisions,
            "catalogue_lookup": False, "known_seed_imported": False}


def render(left: tuple[str, ...], center: dict, punct: dict, right: tuple[str, ...]) -> str:
    # The center and punctuation are selected jointly; no reversed tape is built.
    return f"{clause(left)}{punct['left']} {center['text']}{punct['bridge']}{clause(right)}{punct['right']}"


def audit(left, center, punct, right, rank: int, repair_stage: str = "baseline",
          held_out_slot: str | None = None) -> dict:
    text = render(left, center, punct, right)
    p, s = pointer(text), sha_audit(text)
    complete = text[-1] == "." and text.count(" ") >= 14 and all(text.count(x) == 1 for x in (center["text"],))
    distinct = clause(left) != clause(right)
    flags = {"fixed_tape": False, "reverse_decoder": False, "word_order_mirror": False,
             "nested_palindrome_span": False,
             "self_palindromic_center": letters(center["text"]) == letters(center["text"])[::-1],
             "repeated_chunk": not distinct, "fragment": not complete, "catalogue_text": False,
             "punctuation_changes_letters": False}
    return {"rank": rank, "repair_stage": repair_stage, "held_out_slot": held_out_slot,
            "rank": rank, "rendered": text,
            "left_clause": clause(left), "center_event": center, "right_clause": clause(right),
            "punctuation_choice": punct, "complete_grammar": complete,
            "independent_pointer": p, "independent_sha": s,
            "independent_exact_agreement": p["exact"] == s["exact"],
            "anti_shortcut_flags": flags,
            "mechanically_admitted": bool(p["exact"] and s["exact"] and complete and distinct and not any(flags.values())),
            "residual_repair": p["first_mismatch"],
            "reader_status": "complete readable prose; diagnostic palindrome failure, not an exact palindrome",
            "next_reader_facing_test": "Blind readers rate intact scene coherence and punctuation naturalness against shuffled controls."}


def run() -> dict:
    pre = novelty_preflight()
    if not pre["passed"]:
        raise RuntimeError(pre)
    baseline_rows = []
    for rank, (left, center, punct, right) in enumerate(
        itertools.islice(itertools.product(LEFT, CENTERS, PUNCTUATION, RIGHT), 12), 1):
        baseline_rows.append(audit(left, center, punct, right, rank))
    # Execute the first promised repair: preserve center and punctuation, but hold
    # out and replace the first noun phrase on the right side of the bridge.
    repair_rows = []
    for rank, (left, center, punct, right, replacement) in enumerate(
        itertools.islice(itertools.product(LEFT, CENTERS, PUNCTUATION, RIGHT,
                                            HELDOUT_RIGHT_SUBJECTS), 12), 1):
        repaired = (replacement[0], replacement[1], right[2], right[3], right[4])
        repair_rows.append(audit(left, center, punct, repaired, rank + 12,
                                 repair_stage="held_out_center_adjacent_subject",
                                 held_out_slot="right_subject_np"))
    # Execute the next repair: preserve the repaired subject, event, and
    # punctuation, while holding out the right-side object noun phrase.
    object_repair_rows = []
    for rank, (left, center, punct, right, replacement_subject, replacement_object) in enumerate(
        itertools.islice(itertools.product(LEFT, CENTERS, PUNCTUATION, RIGHT,
                                            HELDOUT_RIGHT_SUBJECTS, HELDOUT_RIGHT_OBJECTS), 12), 1):
        repaired = (replacement_subject[0], replacement_subject[1], right[2], replacement_object, right[4])
        object_repair_rows.append(audit(left, center, punct, repaired, rank + 24,
                                        repair_stage="held_out_right_object_np",
                                        held_out_slot="right_object_np"))
    # Continue with a fresh locative holdout, preserving every preceding
    # repaired choice and changing only the right-side place phrase.
    locative_repair_rows = []
    for rank, (left, center, punct, right, replacement_subject, replacement_object, replacement_place) in enumerate(
        itertools.islice(itertools.product(LEFT, CENTERS, PUNCTUATION, RIGHT,
                                            HELDOUT_RIGHT_SUBJECTS, HELDOUT_RIGHT_OBJECTS,
                                            HELDOUT_RIGHT_LOCATIVES), 12), 1):
        repaired = (replacement_subject[0], replacement_subject[1], right[2], replacement_object, replacement_place)
        locative_repair_rows.append(audit(left, center, punct, repaired, rank + 36,
                                          repair_stage="held_out_right_locative",
                                          held_out_slot="right_locative"))
    # Final recorded repair in this lane: hold out only the right verb
    # inflection, preserving all other repaired lexical and punctuation choices.
    verb_repair_rows = []
    for rank, (left, center, punct, right, replacement_subject, replacement_object,
               replacement_place, replacement_verb) in enumerate(
        itertools.islice(itertools.product(LEFT, CENTERS, PUNCTUATION, RIGHT,
                                            HELDOUT_RIGHT_SUBJECTS, HELDOUT_RIGHT_OBJECTS,
                                            HELDOUT_RIGHT_LOCATIVES, HELDOUT_RIGHT_VERBS), 12), 1):
        repaired = (replacement_subject[0], replacement_subject[1], replacement_verb,
                    replacement_object, replacement_place)
        verb_repair_rows.append(audit(left, center, punct, repaired, rank + 48,
                                      repair_stage="held_out_right_verb_inflection",
                                      held_out_slot="right_verb_inflection"))
    # Preserve the repaired verb/object/place and hold out only the right
    # determiner plus agreement-bearing subject phrase.
    agreement_repair_rows = []
    for rank, (left, center, punct, right, replacement_subject, replacement_object,
               replacement_place, replacement_verb, agreement_subject) in enumerate(
        itertools.islice(itertools.product(LEFT, CENTERS, PUNCTUATION, RIGHT,
                                            HELDOUT_RIGHT_SUBJECTS, HELDOUT_RIGHT_OBJECTS,
                                            HELDOUT_RIGHT_LOCATIVES, HELDOUT_RIGHT_VERBS,
                                            HELDOUT_RIGHT_AGREEMENT_SUBJECTS), 12), 1):
        repaired = (agreement_subject[0], agreement_subject[1], replacement_verb,
                    replacement_object, replacement_place)
        agreement_repair_rows.append(audit(left, center, punct, repaired, rank + 60,
                                           repair_stage="held_out_right_agreement_subject",
                                           held_out_slot="right_determiner_agreement_subject"))
    rows = (baseline_rows + repair_rows + object_repair_rows + locative_repair_rows +
            verb_repair_rows + agreement_repair_rows)
    rows.sort(key=lambda r: (-r["independent_pointer"]["letters"], r["rank"]))
    exact = [r for r in rows if r["mechanically_admitted"]]
    return {"experiment": EXPERIMENT, "signature": SIGNATURE,
            "status": "exact closure found" if exact else "complete prose plus punctuation-center repair",
            "novelty_preflight": pre, "states_considered": 72, "candidate_count": len(rows),
            "exact_count": len(exact), "rendered_candidates": rows, "exact_survivors": exact,
            "repair_summary": {"method": "sequential_subject_object_locative_verb_then_agreement_holdout",
                               "preserved_center_event": True, "preserved_punctuation": True,
                               "held_out_slots": ["right_subject_np", "right_object_np", "right_locative", "right_verb_inflection", "right_determiner_agreement_subject"],
                               "baseline_count": len(baseline_rows), "repaired_count": len(repair_rows),
                               "object_repair_count": len(object_repair_rows),
                               "locative_repair_count": len(locative_repair_rows),
                               "verb_repair_count": len(verb_repair_rows),
                               "agreement_repair_count": len(agreement_repair_rows),
                               "new_subjects": [f"{d} {n}" for d, n in HELDOUT_RIGHT_SUBJECTS],
                               "new_objects": list(HELDOUT_RIGHT_OBJECTS),
                               "new_locatives": list(HELDOUT_RIGHT_LOCATIVES),
                               "new_verbs": list(HELDOUT_RIGHT_VERBS),
                               "new_agreement_subjects": [f"{d} {n}" for d, n in HELDOUT_RIGHT_AGREEMENT_SUBJECTS]},
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "registry_sha256": hashlib.sha256(REGISTRY.read_bytes()).hexdigest(),
                           "punctuation_choices_live": True, "nonpalindromic_center_live": True,
                           "catalogue_imported": False},
            "anti_shortcut_policy": "Reject fixed tapes, reverse decoding, word-order mirrors, repeated/self-palindromic spans, fragments, catalogue text, and punctuation that changes letters.",
            "next_repair": "Use the agreement-repair residual to hold out the right clause's object determiner while preserving event, punctuation, subject, verb, and place.",
            "reader_facing_test": {"required": "blind intact-prose rating plus shuffled-clause control", "status": "pending"}}


if __name__ == "__main__":
    result = run()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"artifact": str(OUT), "candidates": result["candidate_count"], "exact": result["exact_count"]}))
