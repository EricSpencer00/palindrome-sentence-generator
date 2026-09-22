"""Compose several complete inner utterances under one exact paragraph join.

For each Brown-attested nominal X after the command ``Note X.``, decode
``reverse(tape(X)[1:])`` as one to three complete generated utterances.  This
tests whether sentence-level composition can lengthen the readable 32-letter
ABBA proof without repeating or nesting closed palindrome units.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.paragraph_note_grammar_join_20260922 import _sentences
from experiments.paragraph_note_sentence_join_20260922 import (
    _nominal_index, _pointer_audit,
)
from llm_palindrome.admission import mechanical_admission_checks
from llm_palindrome.paragraph_product import audit_staggered_abba


ID = "paragraph-note-dialogue-composition-20260922"


def _sentence_trie(sentences: dict[str, list[dict]]) -> dict:
    root: dict = {}
    for chars in sentences:
        node = root
        for character in chars:
            node = node.setdefault(character, {})
        node.setdefault("$", []).append(chars)
    return root


def _segment(target: str, trie: dict, *, maximum_sentences: int,
             maximum_results: int = 20) -> list[tuple[str, ...]]:
    results: list[tuple[str, ...]] = []

    def visit(offset: int, parts: tuple[str, ...]) -> None:
        if len(results) >= maximum_results:
            return
        if offset == len(target):
            if parts:
                results.append(parts)
            return
        if len(parts) >= maximum_sentences:
            return
        node = trie
        for end in range(offset, len(target)):
            character = target[end]
            if character not in node:
                break
            node = node[character]
            if "$" in node:
                for chars in node["$"]:
                    visit(end + 1, parts + (chars,))

    visit(0, ())
    return results


def search(tagged_sentences: list[list[tuple[str, str]]], *,
           maximum_np_words: int = 7, maximum_inner_sentences: int = 3,
           minimum_letters: int = 39, maximum_rows: int = 200) -> dict:
    nominals, nominal_occurrences = _nominal_index(tagged_sentences,
                                                   maximum_np_words)
    sentences = _sentences()
    sentences.setdefault("no", []).append({
        "words": ("no",), "frame": "dialogue-answer", "roles": ("answer",),
    })
    trie = _sentence_trie(sentences)
    rows = []
    decoded_nominals = 0
    for x_tape, phrase_counts in nominals.items():
        if not x_tape.startswith("s"):
            continue
        target = x_tape[1:][::-1]
        segmentations = _segment(target, trie,
                                 maximum_sentences=maximum_inner_sentences)
        if not segmentations:
            continue
        decoded_nominals += 1
        for segments in segmentations:
            choices = [sentences[chars][0] for chars in segments]
            y_words = tuple(word for choice in choices for word in choice["words"])
            y_sentences = tuple(
                " ".join(choice["words"]).capitalize() + "."
                for choice in choices
            )
            for x_words, x_count in phrase_counts.most_common(3):
                left_sentences = ("No trace.", f"Note {' '.join(x_words)}.")
                right_sentences = y_sentences + ("Set one carton.",)
                rendered = " ".join(left_sentences + right_sentences)
                exact = _pointer_audit(rendered)
                if not exact["exact"]:
                    raise AssertionError((x_words, y_sentences, exact))
                structural = audit_staggered_abba(left_sentences,
                                                    right_sentences)
                admission = mechanical_admission_checks(
                    rendered, min_letters=minimum_letters, max_letters=300
                )
                rows.append({
                    "rendered": rendered,
                    "x_object_phrase": " ".join(x_words),
                    "y_sentences": list(y_sentences),
                    "y_frames": [choice["frame"] for choice in choices],
                    "x_brown_occurrences": x_count,
                    "equation": {"left": x_tape,
                                 "right": "s" + target[::-1],
                                 "holds": x_tape == "s" + target[::-1]},
                    "independent_exact_audit": exact,
                    "structural_audit": structural,
                    "mechanical_admission": admission,
                    "mechanically_admitted": (
                        structural["cross_sentence_coupled"]
                        and all(admission.values())
                    ),
                    "provenance": {
                        "x_object_phrase": "Brown-attested typed nominal",
                        "y_sentences": "generated complete typed utterance sequence",
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
        "method": "trie intersection of typed nominals with one-to-three complete inner utterances",
        "stats": {
            "nominal_tapes": len(nominals),
            "nominal_occurrences": nominal_occurrences,
            "sentence_tapes": len(sentences),
            "decoded_nominals": decoded_nominals,
            "exact_compositions": len(rows),
            "retained_rows": len(kept),
            "mechanically_admitted_rows": sum(
                row["mechanically_admitted"] for row in kept
            ),
        },
        "rows": kept,
        "reader_packet": [],
        "status": (
            "exact compositions require direct prose review before blinded readers"
            if kept else "no exact dialogue composition"
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
    parser.add_argument("--maximum-np-words", type=int, default=7)
    parser.add_argument("--maximum-inner-sentences", type=int, default=3)
    parser.add_argument("--minimum-letters", type=int, default=39)
    parser.add_argument("--maximum-rows", type=int, default=200)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    result = run(maximum_np_words=args.maximum_np_words,
                 maximum_inner_sentences=args.maximum_inner_sentences,
                 minimum_letters=args.minimum_letters,
                 maximum_rows=args.maximum_rows)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stats": result["stats"],
                      "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
