"""Join a Brown-attested nominal to generated complete sentence frames.

This is the constructive follow-up to the intact-sentence join.  It keeps the
same measured paragraph equation but generates the right inner sentence from
agreement-valid typed frames, allowing combinations absent from Brown while
choosing every lexical item inside the exact hash join.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.packed_staggered_paragraph_automaton_20260922 import (
    AGENT, AGENTS, DET, DOCUMENT, INTRANSITIVE, NAMES, PAST, PLACE, PREP,
    PRESENT, PRONOUN, QUANT, SCENE_OBJECT,
)
from experiments.paragraph_note_sentence_join_20260922 import (
    _nominal_index, _pointer_audit,
)
from llm_palindrome.admission import mechanical_admission_checks
from llm_palindrome.paragraph_product import audit_staggered_abba
from llm_palindrome.recursive_product import tape


ID = "paragraph-note-grammar-join-20260922"


def _sentences() -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = defaultdict(list)

    def add(words: tuple[str, ...], frame: str, roles: tuple[str, ...]) -> None:
        chars = tape(" ".join(words))
        if len({word.casefold() for word in words}) != len(words):
            return
        rows[chars].append({"words": words, "frame": frame, "roles": roles})

    for verb in ("listen", "look", "note", "stop", "trace", "wait"):
        add((verb,), "imperative-intransitive", ("event",))
    for verb, determiner, obj in itertools.product(
            ("check", "file", "mark", "note", "read", "set", "trace"),
            DET, DOCUMENT + SCENE_OBJECT):
        add((verb, determiner, obj), "imperative-transitive",
            ("event", "determiner", "patient"))
    for name, verb in itertools.product(NAMES, INTRANSITIVE):
        add((name, verb), "named-intransitive", ("agent", "event"))
    for name, verb, determiner, obj in itertools.product(
            NAMES, PAST, DET, DOCUMENT + SCENE_OBJECT):
        add((name, verb, determiner, obj), "named-transitive",
            ("agent", "event", "determiner", "patient"))
    for determiner, agent, verb, object_determiner, obj in itertools.product(
            DET, AGENT, PAST, DET, DOCUMENT):
        add((determiner, agent, verb, object_determiner, obj),
            "typed-transitive",
            ("determiner", "agent", "event", "determiner", "patient"))
    for quantifier, agents, verb, patient in itertools.product(
            QUANT, AGENTS, PRESENT, NAMES):
        add((quantifier, agents, verb, patient), "plural-present",
            ("quantifier", "agents", "event", "patient"))
    for pronoun, verb, determiner, obj in itertools.product(
            PRONOUN, PAST, DET, DOCUMENT):
        add((pronoun, verb, determiner, obj), "pronoun-transitive",
            ("agent", "event", "determiner", "patient"))
    for name, verb, prep, determiner, place in itertools.product(
            NAMES, INTRANSITIVE, PREP, DET, PLACE):
        add((name, verb, prep, determiner, place), "motion-location",
            ("agent", "event", "preposition", "determiner", "place"))
    return rows


def search(tagged_sentences: list[list[tuple[str, str]]], *,
           maximum_np_words: int = 6, minimum_letters: int = 39,
           maximum_rows: int = 200) -> dict:
    nominals, nominal_occurrences = _nominal_index(tagged_sentences,
                                                   maximum_np_words)
    sentences = _sentences()
    rows = []
    for y_tape, sentence_rows in sentences.items():
        x_tape = "s" + y_tape[::-1]
        if x_tape not in nominals:
            continue
        for sentence_row in sentence_rows[:10]:
            y_words = sentence_row["words"]
            for x_words, x_count in nominals[x_tape].most_common(5):
                left_sentences = ("No trace.", f"Note {' '.join(x_words)}.")
                right_sentences = (
                    f"{' '.join(y_words).capitalize()}.",
                    "Set one carton.",
                )
                rendered = " ".join(left_sentences + right_sentences)
                exact = _pointer_audit(rendered)
                if not exact["exact"]:
                    raise AssertionError((x_words, y_words, exact))
                structural = audit_staggered_abba(left_sentences,
                                                    right_sentences)
                admission = mechanical_admission_checks(
                    rendered, min_letters=minimum_letters, max_letters=300
                )
                rows.append({
                    "rendered": rendered,
                    "x_object_phrase": " ".join(x_words),
                    "y_generated_sentence": " ".join(y_words),
                    "y_frame": sentence_row["frame"],
                    "y_roles": list(sentence_row["roles"]),
                    "x_brown_occurrences": x_count,
                    "equation": {"left": x_tape,
                                 "right": "s" + y_tape[::-1],
                                 "holds": x_tape == "s" + y_tape[::-1]},
                    "independent_exact_audit": exact,
                    "structural_audit": structural,
                    "mechanical_admission": admission,
                    "mechanically_admitted": (
                        structural["cross_sentence_coupled"]
                        and all(admission.values())
                    ),
                    "provenance": {
                        "outer_carrier": "authored open-residual sentence pair",
                        "x_object_phrase": "Brown-attested typed nominal",
                        "y_sentence": "generated from the recorded typed frame and lexical domains",
                    },
                })
    rows.sort(key=lambda row: (
        not row["mechanically_admitted"],
        -row["independent_exact_audit"]["letters"],
        -row["x_brown_occurrences"],
        row["rendered"],
    ))
    kept = rows[:maximum_rows]
    return {
        "experiment_id": ID,
        "method": "joint exact join of a typed nominal and a generated agreement-valid sentence frame",
        "fixed_equation": "tape(X) == 's' + reverse(tape(Y))",
        "stats": {
            "nominal_tapes": len(nominals),
            "nominal_occurrences": nominal_occurrences,
            "generated_sentence_tapes": len(sentences),
            "generated_sentence_realizations": sum(map(len, sentences.values())),
            "exact_joins": len(rows),
            "retained_rows": len(kept),
            "mechanically_admitted_rows": sum(
                row["mechanically_admitted"] for row in kept
            ),
        },
        "rows": kept,
        "reader_packet": [],
        "status": (
            "exact rows require direct prose review before blinded readers"
            if kept else "no exact typed grammar join"
        ),
        "provenance": {
            "corpus": "NLTK Brown corpus supplies typed nominal occurrences only",
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "reader_gate": "programmatic checks cannot certify readability",
        },
    }


def run(**kwargs) -> dict:
    from nltk.corpus import brown
    return search([list(sentence) for sentence in brown.tagged_sents()],
                  **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--maximum-np-words", type=int, default=6)
    parser.add_argument("--minimum-letters", type=int, default=39)
    parser.add_argument("--maximum-rows", type=int, default=200)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    result = run(maximum_np_words=args.maximum_np_words,
                 minimum_letters=args.minimum_letters,
                 maximum_rows=args.maximum_rows)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stats": result["stats"],
                      "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
