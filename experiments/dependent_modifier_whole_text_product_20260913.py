"""Dependent determiner--modifier whole-text palindrome product.

This successor makes the article dependency explicit.  A grammar is compiled
for each complete typed lexical realization of every determiner plus its
immediate adjective: ``an agile artist`` and ``a calm artist`` are legal local
variants, while ``an calm artist`` can never enter a compiled grammar.  The
remaining verb inventory is constrained by the fixed semantic roles in that
same derivation.  Each product still represents one intact clause and uses the
audited character-level whole-text kernel with free even/odd centre meets.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
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


PERSONS = tuple(lex(word, "noun", number="sing", semantic="person")
                for word in ("artist", "captain", "editor", "teacher"))
ARTIFACTS = tuple(lex(word, "noun", number="sing", semantic="artifact")
                  for word in ("canvas", "letter", "model", "report"))
PLACES = tuple(lex(word, "noun", number="sing", semantic="place")
               for word in ("garden", "office", "studio", "workshop"))
PERSON_ADJECTIVES = tuple(lex(word, "adjective", semantic="person")
                          for word in ("agile", "calm", "careful", "patient", "skilled"))
ARTIFACT_ADJECTIVES = tuple(lex(word, "adjective", semantic="artifact")
                            for word in ("brief", "clean", "detailed", "solid", "useful"))
PLACE_ADJECTIVES = tuple(lex(word, "adjective", semantic="place")
                         for word in ("quiet", "public", "remote", "small"))
PAST_VERBS = tuple(lex(word, "past_verb", semantic="transitive", subject_types=("person",), object_types=("artifact",))
                  for word in ("built", "drafted", "painted", "repaired"))
BASE_VERBS = tuple(lex(word, "base_verb", semantic="transitive", subject_types=("person",), object_types=("artifact",))
                   for word in ("build", "draft", "paint", "repair"))


def article_for(adjective: Lexeme, number: str = "sing") -> Lexeme:
    """Bind article choice to the immediately following adjective."""
    if number == "plur":
        return lex("the", "determiner", number="plur")
    form = "an" if adjective.form[0] in "aeiou" else "a"
    return lex(form, "determiner", number="sing")


def _fixed(item: Lexeme) -> tuple[Lexeme, ...]:
    return (item,)


def derive_svo(subject: Lexeme, subject_adj: Lexeme, obj: Lexeme, object_adj: Lexeme) -> Derivation:
    assert subject.semantic == "person" and obj.semantic == "artifact"
    choices = (
        _fixed(article_for(subject_adj, subject.number)), _fixed(subject_adj), _fixed(subject), PAST_VERBS,
        _fixed(article_for(object_adj, obj.number)), _fixed(object_adj), _fixed(obj),
    )
    return Derivation(
        f"svo-{subject.form}-{subject_adj.form}-{obj.form}-{object_adj.form}", "declarative_svo",
        ("subject_det", "subject_adj", "subject", "verb", "object_det", "object_adj", "object"),
        choices, (("subject", subject.form), ("object", obj.form)),
    )


def derive_imperative(obj: Lexeme, object_adj: Lexeme, place: Lexeme, place_adj: Lexeme) -> Derivation:
    assert obj.semantic == "artifact" and place.semantic == "place"
    choices = (
        BASE_VERBS, _fixed(article_for(object_adj, obj.number)), _fixed(object_adj), _fixed(obj),
        _fixed(lex("in", "preposition")), _fixed(article_for(place_adj, place.number)), _fixed(place_adj), _fixed(place),
    )
    return Derivation(
        f"imperative-{obj.form}-{object_adj.form}-{place.form}-{place_adj.form}", "imperative_transitive",
        ("verb", "object_det", "object_adj", "object", "prep", "place_det", "place_adj", "place"),
        choices, (("object", obj.form), ("place", place.form)),
    )


# The cross-product is over typed local realizations, not arbitrary strings.
# This gives broad opening/end alternatives without leaving dependent slots to
# a later repair or mixing semantic frames between compiled grammars.
DERIVATIONS = tuple(
    [derive_svo(subject, subject_adj, obj, object_adj)
     for subject, subject_adj, obj, object_adj in product(PERSONS, PERSON_ADJECTIVES, ARTIFACTS, ARTIFACT_ADJECTIVES)]
    + [derive_imperative(obj, object_adj, place, place_adj)
       for obj, object_adj, place, place_adj in product(ARTIFACTS, ARTIFACT_ADJECTIVES, PLACES, PLACE_ADJECTIVES)]
)


def grammar_for(derivation: Derivation) -> Grammar:
    return compile_slots(derivation.slots)


def replay_words(grammar: Grammar, words: tuple[str, ...]) -> dict:
    """Replay each chosen terminal through the compiled connected path."""
    if len(words) != len(grammar.slots):
        return {"ok": False, "reason": "wrong_slot_count"}
    cursor = grammar.start
    events = []
    for slot, word in enumerate(words):
        if word not in grammar.slots[slot]:
            return {"ok": False, "reason": "word_not_in_slot", "slot": slot}
        for offset, char in enumerate(word):
            options = [edge for edge in grammar.edges if edge.source == cursor and edge.char == char]
            edge = next((candidate for candidate in options
                         if offset < len(word)-1 or candidate.completed_word == word), None)
            if edge is None:
                return {"ok": False, "reason": "edge_mismatch", "slot": slot, "offset": offset}
            cursor = edge.target
            events.append({"slot": slot, "char": char, "source": edge.source, "target": edge.target,
                           "completed_word": edge.completed_word})
        if events[-1]["completed_word"] != word:
            return {"ok": False, "reason": "missing_terminal", "slot": slot}
    return {"ok": cursor == grammar.end, "edge_count": len(events), "words": list(words)}


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
        subj_det, subj_adj, subject, verb = chosen[:4]
        obj_det, obj_adj, obj = chosen[4:]
        agreement = subj_det.number == subject.number and obj_det.number == obj.number
        phonology = (subj_det.form == "the" or subj_det.form == ("an" if subj_adj.form[0] in "aeiou" else "a"))
        phonology = phonology and (obj_det.form == "the" or obj_det.form == ("an" if obj_adj.form[0] in "aeiou" else "a"))
        valency = (verb.semantic == "transitive" and subject.semantic in verb.subject_types and
                   obj.semantic in verb.object_types and subject.form == fixed["subject"] and obj.form == fixed["object"])
    else:
        verb, obj_det, obj_adj, obj, prep, place_det, place_adj, place = chosen
        agreement = obj_det.number == obj.number and place_det.number == place.number
        phonology = (obj_det.form == "the" or obj_det.form == ("an" if obj_adj.form[0] in "aeiou" else "a"))
        phonology = phonology and (place_det.form == "the" or place_det.form == ("an" if place_adj.form[0] in "aeiou" else "a"))
        valency = (prep.form == "in" and verb.semantic == "transitive" and "person" in verb.subject_types and
                   obj.semantic in verb.object_types and obj.form == fixed["object"] and place.form == fixed["place"])
    return {"ok": bool(agreement and phonology and valency), "tokens": list(tokens),
            "roles": list(derivation.roles), "agreement_ok": agreement,
            "determiner_modifier_phonology_ok": phonology, "valency_ok": valency,
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
    if not replay["ok"]: codes.append("independent_terminal_path_replay_failed")
    if not parsed["ok"]: codes.append("independent_complete_reparse_failed")
    if not exact["exact"]: codes.append("independent_exact_audit_failed")
    return {"record_kind": "dependent_modifier_whole_text_closure", "derivation": derivation.identifier,
            "rendered": text, "independent_exact_audit": exact,
            "independent_terminal_path_replay": replay, "independent_parse": parsed,
            "kernel_center_characters": rec["center_characters"], "kernel_midpoint_letter_offset": rec["midpoint_letter_offset"],
            "central_admission": central, "mechanically_admitted": not codes, "rejection_codes": codes,
            "reader_status": "human-unreviewed; programmatic checks do not certify readability"}


def run(max_states: int = 100_000) -> dict:
    if max_states < 1: raise ValueError("max_states must be positive")
    derivation_runs, exact = [], []
    for derivation in DERIVATIONS:
        grammar = grammar_for(derivation)
        kernel = construct(grammar, max_states=max_states)
        outputs = [audit(derivation, grammar, tuple(rec["words"]), rec) for rec in kernel["records"]]
        exact.extend(outputs)
        derivation_runs.append({"derivation": derivation.identifier, "kind": derivation.kind,
                               "fixed_roles": dict(derivation.fixed_roles), "slot_roles": list(derivation.roles),
                               "slot_inventory_sizes": [len(x) for x in derivation.choices],
                               "kernel": {key: value for key, value in kernel.items() if key != "records"},
                               "complete_exact_records": outputs})
    admitted = [row for row in exact if row["mechanically_admitted"]]
    return {"status": "dependent_modifier_whole_text_product",
            "config": {"min_letters": MIN_LETTERS, "max_letters": MAX_LETTERS,
                       "max_states_per_derivation": max_states, "strict_dependent_slot_compilation": True,
                       "dependent_pair": "determiner immediately precedes adjective", "plans": ["SVO", "imperative-transitive"],
                       "even_and_odd_center_meets": True, "center_may_be_inside_word": True,
                       "independent_terminal_path_replay": True, "independent_complete_reparse": True,
                       "corpus_generation": False, "human_readability_required_after_admission": True},
            "eligible_derivation_count": len(DERIVATIONS), "derivation_runs": derivation_runs,
            "exact_closures": exact, "admitted_closures": admitted,
            "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                           "kernel": "experiments/whole_text_palindrome_product_20260913.py",
                           "material": "authored dependent typed lexical realizations; no catalogue or fixed output",
                           "grammar_sha256": sha256(json.dumps([d.slots for d in DERIVATIONS], sort_keys=True).encode()).hexdigest(),
                           "known_tape_check": "central catalogue and endpoint-scaffold gates applied"},
            "reader_facing_next_test": "Only an admitted closure may enter randomized blinded intact-prose and shuffled-control reading; programmatic checks do not certify readability.",
            "scope": "Finite per-derivation dependent-slot products; zero reports true exhaustion versus truncation."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-states", type=int, default=100_000)
    args = parser.parse_args()
    if args.out.exists(): parser.error(f"output already exists: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = run(args.max_states)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "eligible_derivations": result["eligible_derivation_count"],
                      "exact": len(result["exact_closures"]), "admitted": len(result["admitted_closures"])}, indent=2))


if __name__ == "__main__": main()
