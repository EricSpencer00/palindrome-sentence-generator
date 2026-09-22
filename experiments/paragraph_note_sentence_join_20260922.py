"""Join a typed `Note NP` command to an intact sentence across ABBA seams.

The outer sentences satisfy a useful open equation::

    No trace. ... Set one carton.
    notrace       reverse(setonecarton) = notracenotes

The residual ``notes`` lets the left inner sentence begin with ``Note`` and
carry one live ``s`` into its object.  For an object phrase X and an intact
right sentence Y, exactness reduces to ``tape(X) == 's' + reverse(tape(Y))``.
Both X and Y are selected in the same indexed join.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks
from llm_palindrome.paragraph_product import audit_staggered_abba
from llm_palindrome.recursive_product import tape


ID = "paragraph-note-sentence-join-20260922"
TOKEN = re.compile(r"^[A-Za-z]+$")


def _pointer_audit(text: str) -> dict:
    letters = tape(text)
    left, right = 0, len(letters) - 1
    while left < right and letters[left] == letters[right]:
        left += 1
        right -= 1
    digest = hashlib.sha256(letters.encode()).hexdigest()
    reverse_digest = hashlib.sha256(letters[::-1].encode()).hexdigest()
    return {
        "letters": len(letters),
        "exact": left >= right,
        "first_mismatch": None if left >= right else [left, right],
        "sha256": digest,
        "reverse_sha256": reverse_digest,
        "hashes_agree": digest == reverse_digest,
    }


def _nominal_index(tagged_sentences: list[list[tuple[str, str]]],
                   maximum_words: int) -> tuple[
                       dict[str, Counter[tuple[str, ...]]], int]:
    index: dict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)
    occurrences = 0

    def nominal_tag(tag: str) -> bool:
        return tag.startswith(("NN", "NP", "NR", "JJ", "AT", "AP", "CD",
                               "DT", "PP$", "QL"))

    def headed(tags: tuple[str, ...]) -> bool:
        return bool(tags) and tags[-1].startswith(("NN", "NP", "NR"))

    for sentence in tagged_sentences:
        words = [(word.casefold(), tag) for word, tag in sentence
                 if TOKEN.fullmatch(word)]
        for width in range(1, maximum_words + 1):
            for start in range(0, len(words) - width + 1):
                chunk = words[start:start + width]
                phrase = tuple(word for word, _tag in chunk)
                tags = tuple(tag for _word, tag in chunk)
                if (headed(tags) and all(nominal_tag(tag) for tag in tags)):
                    index[tape(" ".join(phrase))][phrase] += 1
                    occurrences += 1
    return index, occurrences


def _intact_sentence(sentence: list[tuple[str, str]],
                     maximum_words: int) -> tuple[str, ...] | None:
    words = tuple(word.casefold() for word, _tag in sentence
                  if TOKEN.fullmatch(word))
    tags = tuple(tag for word, tag in sentence if TOKEN.fullmatch(word))
    if not 1 <= len(words) <= maximum_words:
        return None
    # Retain a finite, explicit approximation to complete clauses.  One-word
    # imperatives are allowed; multiword rows require an overt verb.
    if not any(tag.startswith("VB") for tag in tags):
        return None
    return words


def search(tagged_sentences: list[list[tuple[str, str]]], *,
           maximum_np_words: int = 5, maximum_sentence_words: int = 9,
           minimum_letters: int = 39, maximum_rows: int = 200) -> dict:
    nominals, nominal_occurrences = _nominal_index(tagged_sentences,
                                                   maximum_np_words)
    sentence_counts: Counter[tuple[str, ...]] = Counter()
    for sentence in tagged_sentences:
        words = _intact_sentence(sentence, maximum_sentence_words)
        if words is not None:
            sentence_counts[words] += 1

    rows = []
    for y_words, y_count in sentence_counts.items():
        y_tape = tape(" ".join(y_words))
        x_tape = "s" + y_tape[::-1]
        if x_tape not in nominals:
            continue
        for x_words, x_count in nominals[x_tape].most_common(5):
            left_sentences = (
                "No trace.",
                f"Note {' '.join(x_words)}.",
            )
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
                "y_intact_sentence": " ".join(y_words),
                "x_brown_occurrences": x_count,
                "y_brown_occurrences": y_count,
                "equation": {
                    "left": x_tape,
                    "right": "s" + y_tape[::-1],
                    "holds": x_tape == "s" + y_tape[::-1],
                },
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
                    "y_sentence": "intact Brown sentence",
                },
            })
    rows.sort(key=lambda row: (
        not row["mechanically_admitted"],
        -row["independent_exact_audit"]["letters"],
        -(row["x_brown_occurrences"] + row["y_brown_occurrences"]),
        row["rendered"],
    ))
    kept = rows[:maximum_rows]
    return {
        "experiment_id": ID,
        "method": "exact typed-NP/intact-sentence hash join inside staggered paragraph seams",
        "fixed_equation": "tape(X) == 's' + reverse(tape(Y))",
        "stats": {
            "nominal_tapes": len(nominals),
            "nominal_occurrences": nominal_occurrences,
            "intact_sentence_types": len(sentence_counts),
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
            if kept else "no exact typed sentence join"
        ),
        "provenance": {
            "corpus": "NLTK Brown corpus; intact sentences are cited as corpus material, not claimed as newly authored",
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
    parser.add_argument("--maximum-np-words", type=int, default=5)
    parser.add_argument("--maximum-sentence-words", type=int, default=9)
    parser.add_argument("--minimum-letters", type=int, default=39)
    parser.add_argument("--maximum-rows", type=int, default=200)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    result = run(maximum_np_words=args.maximum_np_words,
                 maximum_sentence_words=args.maximum_sentence_words,
                 minimum_letters=args.minimum_letters,
                 maximum_rows=args.maximum_rows)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stats": result["stats"],
                      "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
