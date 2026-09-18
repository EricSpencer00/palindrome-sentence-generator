"""Whole-text exact-palindrome product over separately compiled clause plans.

This experiment uses the audited character product as the construction kernel.
Each plan is a single connected clause with typed lexical roles; plans are
compiled independently so an SVO subject cannot silently acquire a copular or
imperative object role.  The kernel matches the actual rendered tape from both
ends and permits an even or odd centre at any lexical edge, including inside a
word.  A complete path is then reparsed by an independent role/valency parser.

The finite inventories are authored ordinary words, not catalogue text.  A
programmatic gate can reject a candidate, but never certifies readability.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.whole_text_palindrome_product_20260913 import (
    Grammar, compile_slots, construct,
)
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

MIN_LETTERS, MAX_LETTERS = 100, 180
WORD_RE = re.compile(r"[a-z]+")


@dataclass(frozen=True)
class Word:
    form: str
    category: str
    number: str = "none"
    semantic: str = "any"
    subject_types: tuple[str, ...] = ()
    object_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class Plan:
    name: str
    kind: str
    slot_roles: tuple[str, ...]
    words: tuple[tuple[Word, ...], ...]

    @property
    def slots(self) -> tuple[tuple[str, ...], ...]:
        return tuple(tuple(word.form for word in choices) for choices in self.words)


def _w(form: str, category: str, *, number: str = "none", semantic: str = "any",
       subject_types: tuple[str, ...] = (), object_types: tuple[str, ...] = ()) -> Word:
    return Word(form, category, number, semantic, subject_types, object_types)


DET_SING = (_w("a", "det", number="sing"), _w("an", "det", number="sing"), _w("the", "det", number="sing"))
DET_PLUR = (_w("the", "det", number="plur"),)
ADJ_PERSON = tuple(_w(word, "adj", semantic="person") for word in ("calm", "careful", "patient", "skilled"))
ADJ_ARTIFACT = tuple(_w(word, "adj", semantic="artifact") for word in ("brief", "clean", "detailed", "solid"))
ADJ_PLACE = tuple(_w(word, "adj", semantic="place") for word in ("quiet", "public", "remote", "small"))
PERSON_SING = tuple(_w(word, "noun", number="sing", semantic="person") for word in ("artist", "captain", "editor", "teacher"))
PERSON_PLUR = tuple(_w(word, "noun", number="plur", semantic="person") for word in ("artists", "captains", "editors", "teachers"))
ARTIFACT = tuple(_w(word, "noun", number="sing", semantic="artifact") for word in ("canvas", "letter", "model", "report"))
PLACE = tuple(_w(word, "noun", number="sing", semantic="place") for word in ("garden", "office", "studio", "workshop"))
PAST_OBJECT = (
    _w("built", "verb", semantic="transitive", subject_types=("person",), object_types=("artifact",)),
    _w("cleaned", "verb", semantic="transitive", subject_types=("person",), object_types=("artifact", "place")),
    _w("drafted", "verb", semantic="transitive", subject_types=("person",), object_types=("artifact",)),
    _w("painted", "verb", semantic="transitive", subject_types=("person",), object_types=("artifact",)),
)
BASE_OBJECT = (
    _w("build", "verb", semantic="transitive", subject_types=("person",), object_types=("artifact",)),
    _w("clean", "verb", semantic="transitive", subject_types=("person",), object_types=("artifact", "place")),
    _w("draft", "verb", semantic="transitive", subject_types=("person",), object_types=("artifact",)),
    _w("paint", "verb", semantic="transitive", subject_types=("person",), object_types=("artifact",)),
)
COPULA = tuple(_w(word, "copula", semantic="copular", subject_types=("person",)) for word in ("is", "looks"))


# Each entry is one ordinary clause shape.  The choices are broad at the
# opening and ending lexical roles while all roles remain in their own plan.
PLANS = (
    Plan("svo", "declarative_svo",
         ("subject_det", "subject_adj", "subject", "verb", "object_det", "object_adj", "object"),
         (DET_SING + DET_PLUR, ADJ_PERSON, PERSON_SING + PERSON_PLUR, PAST_OBJECT,
          DET_SING, ADJ_ARTIFACT, ARTIFACT)),
    Plan("copular", "declarative_copular",
         ("subject_det", "subject_adj", "subject", "copula", "complement_det", "complement_adj", "complement"),
         (DET_SING + DET_PLUR, ADJ_PERSON, PERSON_SING + PERSON_PLUR, COPULA,
          DET_SING, ADJ_PLACE, PLACE)),
    Plan("imperative", "imperative_transitive",
         ("verb", "object_det", "object_adj", "object", "place_det", "place_adj", "place"),
         (BASE_OBJECT, DET_SING, ADJ_ARTIFACT, ARTIFACT, DET_SING, ADJ_PLACE, PLACE)),
)


def grammar_for(plan: Plan) -> Grammar:
    return compile_slots(plan.slots)


def exact_audit(text: str) -> dict:
    tape = normalize_letters(text)
    mismatches = [(i, len(tape) - 1 - i) for i in range(len(tape) // 2) if tape[i] != tape[-1-i]]
    return {"exact": bool(tape) and not mismatches, "letters": len(tape),
            "mismatches": mismatches,
            "normalized_sha256": sha256(tape.encode()).hexdigest()}


def independent_parse(plan: Plan, text: str) -> dict:
    tokens = tuple(WORD_RE.findall(text.lower()))
    if len(tokens) != len(plan.words):
        return {"ok": False, "reason": "wrong_word_count", "tokens": list(tokens)}
    chosen = []
    for role, choices, token in zip(plan.slot_roles, plan.words, tokens):
        match = next((word for word in choices if word.form == token), None)
        if match is None:
            return {"ok": False, "reason": f"unknown_{role}", "tokens": list(tokens)}
        chosen.append(match)
    # Determiner agreement and typed transitive valency are checked from the
    # complete surface, not copied from a construction trace.
    if plan.kind == "imperative_transitive":
        agreement = chosen[1].number == chosen[3].number
    else:
        agreement = (chosen[0].number == chosen[2].number and
                     chosen[4].number == chosen[6].number)
    if not agreement:
        return {"ok": False, "reason": "agreement_failure", "tokens": list(tokens)}
    if plan.kind == "declarative_copular":
        agreement = chosen[0].number == chosen[2].number
        valency = chosen[3].semantic == "copular" and chosen[6].semantic == "place"
    elif plan.kind == "imperative_transitive":
        # The imperative subject is an understood second person; its semantic
        # type is fixed by the plan, not inferred from an absent surface word.
        valency = chosen[0].semantic == "transitive" and chosen[3].semantic in chosen[0].object_types
    else:
        verb = chosen[0] if plan.kind == "imperative_transitive" else chosen[3]
        subject = chosen[3] if plan.kind == "imperative_transitive" else chosen[2]
        obj = chosen[3] if plan.kind == "imperative_transitive" else chosen[6]
        valency = verb.semantic == "transitive" and subject.semantic == "person" and obj.semantic in verb.object_types
    return {"ok": bool(agreement and valency), "tokens": list(tokens),
            "roles": list(plan.slot_roles), "agreement_ok": agreement,
            "valency_ok": valency, "plan": plan.name}


def render(words: tuple[str, ...]) -> str:
    sentence = " ".join(words)
    return sentence[:1].upper() + sentence[1:] + "."


def audit(plan: Plan, words: tuple[str, ...], *, record_kind: str, center_characters: int,
          midpoint_letter_offset: int) -> dict:
    text = render(words)
    parse = independent_parse(plan, text)
    exact = exact_audit(text)
    central = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    codes = [key for key, value in central.items() if not value]
    if not parse["ok"]:
        codes.append("independent_complete_reparse_failed")
    return {"record_kind": record_kind, "plan": plan.name, "rendered": text,
            "independent_exact_audit": exact, "independent_parse": parse,
            "central_admission": central, "mechanically_admitted": not codes,
            "rejection_codes": codes, "center_characters": center_characters,
            "midpoint_letter_offset": midpoint_letter_offset,
            "reader_status": "human-unreviewed; programmatic checks do not certify readability"}


def run(max_states: int = 100_000) -> dict:
    if max_states < 1:
        raise ValueError("max_states must be positive")
    plan_runs = []
    exact = []
    admitted = []
    for plan in PLANS:
        grammar = grammar_for(plan)
        result = construct(grammar, max_states=max_states)
        records = []
        for rec in result["records"]:
            row = audit(plan, tuple(rec["words"]), record_kind="whole_text_product_closure",
                        center_characters=rec["center_characters"],
                        midpoint_letter_offset=rec["midpoint_letter_offset"])
            row["kernel_replay"] = {
                "connected_path": True, "all_leaves_consumed": len(rec["words"]) == len(plan.words),
                "kernel_exact": rec["exact"], "letters": rec["letters"]}
            records.append(row)
            exact.append(row)
            if row["mechanically_admitted"]:
                admitted.append(row)
        plan_runs.append({"plan": plan.name, "kind": plan.kind,
                          "slot_roles": list(plan.slot_roles),
                          "slot_inventory_sizes": [len(x) for x in plan.words],
                          "kernel": {key: value for key, value in result.items() if key != "records"},
                          "complete_exact_records": records})
    return {
        "status": "whole_text_typed_clause_product",
        "config": {"min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS,
                   "max_states_per_plan": max_states,
                   "separate_plan_compilation": True, "single_intact_clause_per_plan": True,
                   "even_and_odd_center_meets": True, "center_may_be_inside_word": True,
                   "independent_complete_reparse": True, "corpus_generation": False,
                   "human_readability_required_after_admission": True},
        "grammar_inventory": [{"plan": p.name, "kind": p.kind, "slots": list(p.slot_roles),
                               "choices": [list(s) for s in p.slots]} for p in PLANS],
        "plan_runs": plan_runs, "exact_closures": exact, "admitted_closures": admitted,
        "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                       "kernel": "experiments/whole_text_palindrome_product_20260913.py",
                       "material": "authored ordinary typed lexical inventories; no catalogue or fixed output",
                       "known_tape_check": "central admission local-catalogue and endpoint-scaffold gates applied",
                       "grammar_sha256": sha256(json.dumps([p.slots for p in PLANS], sort_keys=True).encode()).hexdigest()},
        "reader_facing_next_test": "Only a mechanically admitted closure can enter a randomized blinded intact-prose versus shuffled-control study; none is readable-certified by this run.",
        "scope": "Finite separately compiled clause products; exactness is mechanically audited and readability remains a human question.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-states", type=int, default=100_000)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = run(args.max_states)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "exact": len(result["exact_closures"]),
                      "admitted": len(result["admitted_closures"])}, indent=2))


if __name__ == "__main__":
    main()
