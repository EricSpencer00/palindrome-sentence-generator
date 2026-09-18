"""Semantic endpoint algebra before whole-text palindrome compilation.

This successor indexes typed initial-role realizations and typed final-phrase
realizations by their actual letter signatures.  A join is eligible only when
the opening prefix and reversed final phrase agree for at least two characters.
Only then is that complete, semantically fixed clause compiled through the
audited free-midpoint kernel.  The endpoint algebra never joins a subject to an
unrelated object frame: both realizations retain the same derivation ID and
typed clause kind.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.whole_text_palindrome_product_20260913 import Grammar, compile_slots, construct
from experiments.dependent_modifier_whole_text_product_20260913 import (
    ARTIFACTS, ARTIFACT_ADJECTIVES, BASE_VERBS, PERSONS, PERSON_ADJECTIVES,
    PLACE_ADJECTIVES, PLACES, Derivation, derive_imperative, derive_svo,
    exact_audit, independent_parse, lex, render, replay_words,
)
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

MIN_LETTERS, MAX_LETTERS = 30, 160
WORD_RE = re.compile(r"[a-z]+")


# These are additional authored role alternatives chosen for endpoint algebra,
# not copied phrases.  The first/last signatures are discovered by the index.
EXTRA_PERSON_ADJECTIVES = PERSON_ADJECTIVES + (lex("happy", "adjective", semantic="person"),)
EXTRA_ARTIFACTS = ARTIFACTS + (lex("light", "noun", number="sing", semantic="artifact"),)
EXTRA_ARTIFACT_ADJECTIVES = ARTIFACT_ADJECTIVES + (lex("bright", "adjective", semantic="artifact"),)
EXTRA_PLACES = PLACES + (lex("shelter", "noun", number="sing", semantic="place"),)


def all_derivations() -> tuple[Derivation, ...]:
    # The endpoint join is performed after these legal local realizations are
    # generated, but before any grammar is compiled or searched.
    rows = [derive_svo(subject, subject_adj, obj, object_adj)
            for subject in PERSONS for subject_adj in EXTRA_PERSON_ADJECTIVES
            for obj in EXTRA_ARTIFACTS for object_adj in EXTRA_ARTIFACT_ADJECTIVES]
    rows += [derive_imperative(obj, object_adj, place, place_adj)
             for obj in EXTRA_ARTIFACTS for object_adj in EXTRA_ARTIFACT_ADJECTIVES
             for place in EXTRA_PLACES for place_adj in PLACE_ADJECTIVES]
    return tuple(rows)


DERIVATIONS = all_derivations()


def _letters(words: tuple[str, ...]) -> str:
    return "".join(words).lower()


def endpoint_realizations(derivation: Derivation) -> tuple[dict, ...]:
    """Return typed initial and final phrase realizations for one derivation."""
    if derivation.kind == "declarative_svo":
        opening = tuple(slot[0].form for slot in derivation.choices[:3])
        final = tuple(slot[0].form for slot in derivation.choices[4:])
    else:
        # Imperative verb alternatives are all valency-compatible for the
        # fixed artifact; the locative phrase is the typed final role.
        final = tuple(slot[0].form for slot in derivation.choices[4:])
        return tuple({"derivation": derivation.identifier, "kind": derivation.kind,
                      "side": "opening", "words": (verb.form,),
                      "letters": verb.form, "semantic_role": "agent_action"}
                     for verb in derivation.choices[0]) + ({"derivation": derivation.identifier,
                      "kind": derivation.kind, "side": "final", "words": final,
                      "letters": _letters(final), "semantic_role": "locative_complement"},)
    return ({"derivation": derivation.identifier, "kind": derivation.kind, "side": "opening",
             "words": opening, "letters": _letters(opening), "semantic_role": "subject_np"},
            {"derivation": derivation.identifier, "kind": derivation.kind, "side": "final",
             "words": final, "letters": _letters(final), "semantic_role": "object_np"})


def endpoint_algebra(derivations: tuple[Derivation, ...] = DERIVATIONS) -> dict:
    opening_index: dict[tuple[str, str], list[dict]] = defaultdict(list)
    reverse_suffix_index: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for derivation in derivations:
        for realization in endpoint_realizations(derivation):
            letters = realization["letters"]
            if len(letters) < 2:
                continue
            key = (realization["kind"], letters[:2]) if realization["side"] == "opening" else (realization["kind"], letters[::-1][:2])
            if realization["side"] == "opening":
                opening_index[key].append(realization)
            else:
                reverse_suffix_index[key].append(realization)
    joins = []
    eligible = set()
    for derivation in derivations:
        realizations = endpoint_realizations(derivation)
        openings = [r for r in realizations if r["side"] == "opening"]
        finals = [r for r in realizations if r["side"] == "final"]
        for opening in openings:
            key = (opening["kind"], opening["letters"][:2])
            for ending in reverse_suffix_index.get(key, ()):
                if ending["derivation"] != derivation.identifier:
                    continue
                matched = 0
                for left, right in zip(opening["letters"], ending["letters"][::-1]):
                    if left != right:
                        break
                    matched += 1
                if matched < 2:
                    continue
                eligible.add(derivation.identifier)
                joins.append({"derivation": derivation.identifier, "kind": derivation.kind,
                              "opening_words": list(opening["words"]), "final_words": list(ending["words"]),
                              "opening_letters": opening["letters"], "reversed_final_letters": ending["letters"][::-1],
                              "matched_pairs": matched, "semantic_roles": [opening["semantic_role"], ending["semantic_role"]]})
    return {"opening_index": {f"{kind}:{key}": len(values) for (kind, key), values in opening_index.items()},
            "reverse_suffix_index": {f"{kind}:{key}": len(values) for (kind, key), values in reverse_suffix_index.items()},
            "joins": joins, "eligible_derivation_ids": sorted(eligible),
            "minimum_matched_pairs": 2}


def grammar_for(derivation: Derivation) -> Grammar:
    return compile_slots(derivation.slots)


def audit(derivation: Derivation, grammar: Grammar, words: tuple[str, ...], rec: dict, join: dict) -> dict:
    text = render(words)
    exact = exact_audit(text)
    replay = replay_words(grammar, words)
    parsed = independent_parse(derivation, text)
    central = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    codes = [key for key, value in central.items() if not value]
    if not exact["exact"]: codes.append("independent_exact_audit_failed")
    if not replay["ok"]: codes.append("independent_terminal_path_replay_failed")
    if not parsed["ok"]: codes.append("independent_complete_reparse_failed")
    return {"record_kind": "semantic_endpoint_algebra_whole_text_closure",
            "derivation": derivation.identifier, "rendered": text,
            "endpoint_join": join, "independent_exact_audit": exact,
            "independent_terminal_path_replay": replay, "independent_parse": parsed,
            "kernel_center_characters": rec["center_characters"], "kernel_midpoint_letter_offset": rec["midpoint_letter_offset"],
            "central_admission": central, "mechanically_admitted": not codes,
            "rejection_codes": codes,
            "reader_status": "human-unreviewed; programmatic checks do not certify readability"}


def run(max_states: int = 100_000) -> dict:
    if max_states < 1: raise ValueError("max_states must be positive")
    algebra = endpoint_algebra()
    joins_by_derivation = defaultdict(list)
    for row in algebra["joins"]: joins_by_derivation[row["derivation"]].append(row)
    derivation_runs, exact = [], []
    for derivation in DERIVATIONS:
        joins = joins_by_derivation.get(derivation.identifier, ())
        if not joins: continue
        grammar = grammar_for(derivation)
        kernel = construct(grammar, max_states=max_states)
        outputs = []
        for rec in kernel["records"]:
            row = audit(derivation, grammar, tuple(rec["words"]), rec, joins[0])
            outputs.append(row); exact.append(row)
        derivation_runs.append({"derivation": derivation.identifier, "kind": derivation.kind,
                               "endpoint_joins": joins, "slot_roles": list(derivation.roles),
                               "slot_inventory_sizes": [len(x) for x in derivation.choices],
                               "kernel": {key: value for key, value in kernel.items() if key != "records"},
                               "complete_exact_records": outputs})
    admitted = [row for row in exact if row["mechanically_admitted"]]
    return {"status": "semantic_endpoint_algebra_whole_text_product",
            "config": {"min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS,
                       "max_states_per_eligible_derivation": max_states,
                       "endpoint_algebra_before_kernel": True, "minimum_endpoint_matched_pairs": 2,
                       "plans": ["SVO", "imperative-transitive"], "single_intact_clause": True,
                       "even_and_odd_center_meets": True, "center_may_be_inside_word": True,
                       "independent_terminal_path_replay": True, "independent_complete_reparse": True,
                       "corpus_generation": False, "human_readability_required_after_admission": True},
            "authored_derivation_count": len(DERIVATIONS), "eligible_derivation_count": len(derivation_runs),
            "endpoint_algebra": algebra, "derivation_runs": derivation_runs,
            "exact_closures": exact, "admitted_closures": admitted,
            "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                           "kernel": "experiments/whole_text_palindrome_product_20260913.py",
                           "material": "authored typed semantic role inventories; endpoint signatures discovered by index; no catalogue or fixed output",
                           "grammar_sha256": sha256(json.dumps([d.slots for d in DERIVATIONS], sort_keys=True).encode()).hexdigest(),
                           "known_tape_check": "central catalogue and endpoint-scaffold gates applied"},
            "reader_facing_next_test": "Only an admitted closure may enter randomized blinded intact-prose and shuffled-control reading; programmatic checks do not certify readability.",
            "scope": "Finite endpoint-indexed typed clause products; zero reports endpoint depth and per-plan exhaustion versus truncation."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-states", type=int, default=100_000)
    args = parser.parse_args()
    if args.out.exists(): parser.error(f"output already exists: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = run(args.max_states)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "authored": result["authored_derivation_count"],
                      "eligible": result["eligible_derivation_count"], "exact": len(result["exact_closures"]),
                      "admitted": len(result["admitted_closures"])}, indent=2))


if __name__ == "__main__": main()
