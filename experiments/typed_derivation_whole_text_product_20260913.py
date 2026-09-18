"""Strictly typed whole-text clause products.

This is a new wrapper around the audited whole-text product.  A grammar is
compiled separately for each authored subject/object derivation, so
determiner phonology, number agreement, valency, and semantic role selection
are true properties of every lexical alternative in that grammar.  The
product is still character-level and accepts centres inside words or at
clause boundaries.  No generated row is called readable without people.
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

from experiments.whole_text_palindrome_product_20260913 import Grammar, compile_slots, construct
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

MIN_LETTERS, MAX_LETTERS = 30, 160
WORD_RE = re.compile(r"[a-z]+")


@dataclass(frozen=True)
class Lexeme:
    form: str
    category: str
    number: str = "none"
    semantic: str = "any"
    subject_types: tuple[str, ...] = ()
    object_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class Derivation:
    identifier: str
    kind: str
    roles: tuple[str, ...]
    choices: tuple[tuple[Lexeme, ...], ...]
    fixed_roles: tuple[tuple[str, str], ...]

    @property
    def slots(self) -> tuple[tuple[str, ...], ...]:
        return tuple(tuple(item.form for item in choices) for choices in self.choices)


def lex(form: str, category: str, *, number: str = "none", semantic: str = "any",
        subject_types: tuple[str, ...] = (), object_types: tuple[str, ...] = ()) -> Lexeme:
    return Lexeme(form, category, number, semantic, subject_types, object_types)


PERSON_NOUNS = (
    lex("artist", "noun", number="sing", semantic="person"),
    lex("captain", "noun", number="sing", semantic="person"),
    lex("editor", "noun", number="sing", semantic="person"),
    lex("teacher", "noun", number="sing", semantic="person"),
)
ARTIFACT_NOUNS = (
    lex("canvas", "noun", number="sing", semantic="artifact"),
    lex("letter", "noun", number="sing", semantic="artifact"),
    lex("model", "noun", number="sing", semantic="artifact"),
    lex("report", "noun", number="sing", semantic="artifact"),
)
PLACE_NOUNS = (
    lex("garden", "noun", number="sing", semantic="place"),
    lex("office", "noun", number="sing", semantic="place"),
    lex("studio", "noun", number="sing", semantic="place"),
    lex("workshop", "noun", number="sing", semantic="place"),
)
PERSON_ADJECTIVES = tuple(lex(word, "adjective", semantic="person") for word in ("agile", "calm", "careful", "patient", "skilled"))
ARTIFACT_ADJECTIVES = tuple(lex(word, "adjective", semantic="artifact") for word in ("brief", "clean", "detailed", "solid", "useful"))
PLACE_ADJECTIVES = tuple(lex(word, "adjective", semantic="place") for word in ("quiet", "public", "remote", "small"))
PAST_VERBS = tuple(lex(word, "past_verb", semantic="transitive", subject_types=("person",), object_types=("artifact",))
                  for word in ("built", "drafted", "painted", "repaired"))
BASE_VERBS = tuple(lex(word, "base_verb", semantic="transitive", subject_types=("person",), object_types=("artifact",))
                   for word in ("build", "draft", "paint", "repair"))


def determiner_for(noun: Lexeme) -> tuple[Lexeme, ...]:
    """Return only forms licensed by number and English article phonology."""
    if noun.number == "plur":
        return (lex("the", "determiner", number="plur"),)
    article = "an" if noun.form[0] in "aeiou" else "a"
    return (lex(article, "determiner", number="sing"), lex("the", "determiner", number="sing"))


def _choices(*items: tuple[Lexeme, ...] | Lexeme) -> tuple[tuple[Lexeme, ...], ...]:
    return tuple(item if isinstance(item, tuple) else (item,) for item in items)


def derive_svo(subject: Lexeme, obj: Lexeme) -> Derivation:
    assert subject.semantic == "person" and obj.semantic == "artifact"
    choices = _choices(determiner_for(subject), PERSON_ADJECTIVES, subject, PAST_VERBS,
                      determiner_for(obj), ARTIFACT_ADJECTIVES, obj)
    return Derivation(f"svo-{subject.form}-{obj.form}", "declarative_svo",
                      ("subject_det", "subject_adj", "subject", "verb", "object_det", "object_adj", "object"),
                      choices, (("subject", subject.form), ("object", obj.form)))


def derive_imperative(obj: Lexeme, place: Lexeme) -> Derivation:
    assert obj.semantic == "artifact" and place.semantic == "place"
    choices = _choices(BASE_VERBS, determiner_for(obj), ARTIFACT_ADJECTIVES, obj,
                      (lex("in", "preposition"),), determiner_for(place), PLACE_ADJECTIVES, place)
    return Derivation(f"imperative-{obj.form}-{place.form}", "imperative_transitive",
                      ("verb", "object_det", "object_adj", "object", "prep", "place_det", "place_adj", "place"),
                      choices, (("object", obj.form), ("place", place.form)))


DERIVATIONS = tuple(
    [derive_svo(subject, obj) for subject in PERSON_NOUNS for obj in ARTIFACT_NOUNS]
    + [derive_imperative(obj, place) for obj in ARTIFACT_NOUNS for place in PLACE_NOUNS]
)


def grammar_for(derivation: Derivation) -> Grammar:
    # Every slot in this compilation is role-valid for the fixed derivation;
    # there is no later cross-plan phrase stitching.
    return compile_slots(derivation.slots)


def replay_words(grammar: Grammar, words: tuple[str, ...]) -> dict:
    """Independently replay every terminal edge, including its word boundary."""
    if len(words) != len(grammar.slots):
        return {"ok": False, "reason": "wrong_slot_count"}
    cursor = grammar.start
    edge_trace = []
    for slot_index, word in enumerate(words):
        if word not in grammar.slots[slot_index]:
            return {"ok": False, "reason": "word_not_in_slot", "slot": slot_index}
        for offset, char in enumerate(word):
            options = [edge for edge in grammar.edges if edge.source == cursor and edge.char == char]
            edge = next((edge for edge in options if (offset < len(word)-1 or edge.completed_word == word)), None)
            if edge is None:
                return {"ok": False, "reason": "edge_mismatch", "slot": slot_index, "offset": offset}
            cursor = edge.target
            edge_trace.append({"slot": slot_index, "char": char, "source": edge.source, "target": edge.target,
                               "completed_word": edge.completed_word})
        if edge_trace[-1]["completed_word"] != word:
            return {"ok": False, "reason": "missing_terminal", "slot": slot_index}
    return {"ok": cursor == grammar.end and len(edge_trace) == sum(map(len, words)),
            "edge_count": len(edge_trace), "words": list(words)}


def independent_parse(derivation: Derivation, text: str) -> dict:
    tokens = tuple(WORD_RE.findall(text.lower()))
    if len(tokens) != len(derivation.choices):
        return {"ok": False, "reason": "wrong_word_count", "tokens": list(tokens)}
    chosen = []
    for role, options, token in zip(derivation.roles, derivation.choices, tokens):
        item = next((x for x in options if x.form == token), None)
        if item is None:
            return {"ok": False, "reason": f"unknown_{role}", "tokens": list(tokens)}
        chosen.append(item)
    fixed = dict(derivation.fixed_roles)
    if derivation.kind == "declarative_svo":
        subject, verb, obj = chosen[2], chosen[3], chosen[6]
        agreement = chosen[0].number == subject.number and chosen[4].number == obj.number
        phonology = chosen[0].form in {"a", "an", "the"} and (chosen[0].form == "the" or chosen[0].form == ("an" if subject.form[0] in "aeiou" else "a"))
        phonology = phonology and (chosen[4].form == "the" or chosen[4].form == ("an" if obj.form[0] in "aeiou" else "a"))
        valency = (verb.semantic == "transitive" and subject.semantic in verb.subject_types and
                   obj.semantic in verb.object_types and subject.form == fixed["subject"] and obj.form == fixed["object"])
    else:
        verb, obj, place = chosen[0], chosen[3], chosen[7]
        agreement = chosen[1].number == obj.number and chosen[5].number == place.number
        phonology = all(item.form == "the" or item.form == ("an" if noun.form[0] in "aeiou" else "a")
                        for item, noun in ((chosen[1], obj), (chosen[5], place)))
        valency = (verb.semantic == "transitive" and "person" in verb.subject_types and
                   obj.semantic in verb.object_types and obj.form == fixed["object"] and place.form == fixed["place"])
    return {"ok": bool(agreement and phonology and valency), "tokens": list(tokens),
            "roles": list(derivation.roles), "agreement_ok": agreement,
            "determiner_phonology_ok": phonology, "valency_ok": valency,
            "semantic_frame": derivation.kind}


def render(words: tuple[str, ...]) -> str:
    text = " ".join(words)
    return text[:1].upper() + text[1:] + "."


def exact_audit(text: str) -> dict:
    tape = normalize_letters(text)
    mismatches = [(i, len(tape)-1-i) for i in range(len(tape)//2) if tape[i] != tape[-1-i]]
    return {"exact": bool(tape) and not mismatches, "letters": len(tape),
            "mismatches": mismatches, "normalized_sha256": sha256(tape.encode()).hexdigest()}


def audit(derivation: Derivation, grammar: Grammar, words: tuple[str, ...], rec: dict) -> dict:
    text = render(words)
    replay = replay_words(grammar, words)
    parsed = independent_parse(derivation, text)
    exact = exact_audit(text)
    central = mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    codes = [key for key, value in central.items() if not value]
    if not replay["ok"]:
        codes.append("independent_kernel_path_replay_failed")
    if not parsed["ok"]:
        codes.append("independent_complete_reparse_failed")
    if not exact["exact"]:
        codes.append("independent_exact_audit_failed")
    return {"record_kind": "whole_text_typed_derivation_closure", "derivation": derivation.identifier,
            "rendered": text, "independent_exact_audit": exact,
            "independent_path_replay": replay, "independent_parse": parsed,
            "kernel_center_characters": rec["center_characters"],
            "kernel_midpoint_letter_offset": rec["midpoint_letter_offset"],
            "central_admission": central, "mechanically_admitted": not codes,
            "rejection_codes": codes,
            "reader_status": "human-unreviewed; programmatic checks do not certify readability"}


def run(max_states: int = 100_000) -> dict:
    if max_states < 1:
        raise ValueError("max_states must be positive")
    rows = []
    derivation_runs = []
    for derivation in DERIVATIONS:
        grammar = grammar_for(derivation)
        kernel = construct(grammar, max_states=max_states)
        outputs = []
        for rec in kernel["records"]:
            row = audit(derivation, grammar, tuple(rec["words"]), rec)
            outputs.append(row)
            rows.append(row)
        derivation_runs.append({"derivation": derivation.identifier, "kind": derivation.kind,
                               "fixed_roles": dict(derivation.fixed_roles),
                               "slot_roles": list(derivation.roles),
                               "slot_inventory_sizes": [len(x) for x in derivation.choices],
                               "kernel": {key: value for key, value in kernel.items() if key != "records"},
                               "complete_exact_records": outputs})
    admitted = [row for row in rows if row["mechanically_admitted"]]
    return {
        "status": "strict_typed_derivation_whole_text_product",
        "config": {"min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS,
                   "max_states_per_derivation": max_states, "discovery_band": "30-160 letters",
                   "strict_per_derivation_compilation": True, "plans": ["SVO", "imperative-transitive"],
                   "even_and_odd_center_meets": True, "center_may_be_inside_word": True,
                   "independent_terminal_path_replay": True, "independent_complete_reparse": True,
                   "corpus_generation": False, "human_readability_required_after_admission": True},
        "eligible_derivation_count": len(DERIVATIONS), "derivation_runs": derivation_runs,
        "exact_closures": rows, "admitted_closures": admitted,
        "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                       "kernel": "experiments/whole_text_palindrome_product_20260913.py",
                       "material": "authored typed role inventories; no catalogue or fixed output",
                       "grammar_sha256": sha256(json.dumps([d.slots for d in DERIVATIONS], sort_keys=True).encode()).hexdigest(),
                       "known_tape_check": "central admission catalogue and endpoint-scaffold gates applied"},
        "reader_facing_next_test": "Only an admitted closure may enter a randomized blinded intact-prose versus shuffled-control study; programmatic checks do not certify readability.",
        "scope": "Finite typed derivation inventory; zero closures report true per-derivation exhaustion versus truncation.",
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
    print(json.dumps({"out": str(args.out), "eligible_derivations": result["eligible_derivation_count"],
                      "exact": len(result["exact_closures"]), "admitted": len(result["admitted_closures"])}, indent=2))


if __name__ == "__main__":
    main()
