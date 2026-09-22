"""Join attested phrases inside a strict four-sentence ABBA paragraph seam.

The authored carrier has one open equation::

    No. Trace note stress as X. Y assert. Set one carton.

After normalization, every fixed character cancels and the remaining exact
condition is ``reverse(tape(Y)) == "s" + tape(X)``.  Brown n-grams are indexed
by normalized tape so the linguistic phrase choice and the palindrome
condition are solved together rather than repaired after generation.
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


ID = "paragraph-middle-seam-phrase-join-20260922"
TOKEN = re.compile(r"^[A-Za-z]+$")


def _two_pointer(value: str) -> dict:
    letters = tape(value)
    left, right = 0, len(letters) - 1
    while left < right and letters[left] == letters[right]:
        left += 1
        right -= 1
    exact = left >= right
    digest = hashlib.sha256(letters.encode()).hexdigest()
    reverse_digest = hashlib.sha256(letters[::-1].encode()).hexdigest()
    return {
        "letters": len(letters),
        "exact": exact,
        "first_mismatch": None if exact else [left, right],
        "sha256": digest,
        "reverse_sha256": reverse_digest,
        "hashes_agree": digest == reverse_digest,
    }


def _ngrams(sentences: list[list[str]], maximum_words: int) -> tuple[
        dict[str, Counter[tuple[str, ...]]], int]:
    by_tape: dict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)
    occurrences = 0
    for sentence in sentences:
        words = [word.casefold() for word in sentence if TOKEN.fullmatch(word)]
        for width in range(1, maximum_words + 1):
            for start in range(0, len(words) - width + 1):
                phrase = tuple(words[start:start + width])
                chars = tape(" ".join(phrase))
                if len(chars) < 3:
                    continue
                by_tape[chars][phrase] += 1
                occurrences += 1
    return by_tape, occurrences


def _typed_domains(tagged_sentences: list[list[tuple[str, str]]],
                   maximum_words: int) -> tuple[set[tuple[str, ...]],
                                                set[tuple[str, ...]]]:
    """Return complete `as` complements and plural subject-NP phrases."""
    x_phrases: set[tuple[str, ...]] = set()
    y_phrases: set[tuple[str, ...]] = set()

    def noun(tag: str) -> bool:
        return tag.startswith(("NN", "NP", "NR"))

    def adjective(tag: str) -> bool:
        return tag.startswith("JJ")

    def nominal(tags: tuple[str, ...]) -> bool:
        if not tags or not (noun(tags[-1]) or adjective(tags[-1])):
            return False
        return all(noun(tag) or adjective(tag)
                   or tag.startswith(("AT", "AP", "CD", "DT", "PP$", "QL"))
                   for tag in tags)

    for sentence in tagged_sentences:
        clean = [(word.casefold(), tag) for word, tag in sentence
                 if TOKEN.fullmatch(word)]
        for index, (word, _tag) in enumerate(clean):
            if word == "as":
                for width in range(1, maximum_words + 1):
                    chunk = clean[index + 1:index + 1 + width]
                    if len(chunk) != width:
                        break
                    words = tuple(item[0] for item in chunk)
                    tags = tuple(item[1] for item in chunk)
                    if nominal(tags):
                        x_phrases.add(words)
        for end, (_word, tag) in enumerate(clean):
            if not tag.startswith(("NNS", "NPS")):
                continue
            for width in range(1, min(maximum_words, end + 1) + 1):
                chunk = clean[end - width + 1:end + 1]
                words = tuple(item[0] for item in chunk)
                tags = tuple(item[1] for item in chunk)
                if nominal(tags):
                    y_phrases.add(words)
    return x_phrases, y_phrases


def _phrase_rows(sentences: list[list[str]], *, maximum_words: int,
                 maximum_rows: int,
                 x_allowed: set[tuple[str, ...]] | None = None,
                 y_allowed: set[tuple[str, ...]] | None = None) -> tuple[list[dict], dict]:
    by_tape, occurrences = _ngrams(sentences, maximum_words)
    rows = []
    seen_pairs = set()
    for y_tape, y_phrases in by_tape.items():
        if not y_tape.endswith("s") or len(y_tape) < 4:
            continue
        x_tape = y_tape[::-1][1:]
        if x_tape not in by_tape:
            continue
        for y_phrase, y_count in y_phrases.most_common(3):
            if y_allowed is not None and y_phrase not in y_allowed:
                continue
            for x_phrase, x_count in by_tape[x_tape].most_common(3):
                if x_allowed is not None and x_phrase not in x_allowed:
                    continue
                # A single reversed token pair is the old semordnilap shortcut;
                # require at least one side to carry an internal word boundary.
                if len(x_phrase) == len(y_phrase) == 1:
                    continue
                key = (x_phrase, y_phrase)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                left_sentences = (
                    "No.",
                    f"Trace note stress as {' '.join(x_phrase)}.",
                )
                right_sentences = (
                    f"{' '.join(y_phrase).capitalize()} assert.",
                    "Set one carton.",
                )
                rendered = " ".join(left_sentences + right_sentences)
                exact = _two_pointer(rendered)
                if not exact["exact"]:
                    raise AssertionError((x_phrase, y_phrase, exact))
                structural = audit_staggered_abba(left_sentences,
                                                    right_sentences)
                admission = mechanical_admission_checks(
                    rendered, min_letters=39, max_letters=300
                )
                rows.append({
                    "rendered": rendered,
                    "x_phrase": " ".join(x_phrase),
                    "y_phrase": " ".join(y_phrase),
                    "x_brown_occurrences": x_count,
                    "y_brown_occurrences": y_count,
                    "equation": {
                        "left": "s" + x_tape,
                        "right": y_tape[::-1],
                        "holds": "s" + x_tape == y_tape[::-1],
                    },
                    "independent_exact_audit": exact,
                    "structural_audit": structural,
                    "mechanical_admission": admission,
                    "mechanically_admitted": (
                        structural["cross_sentence_coupled"]
                        and all(admission.values())
                    ),
                    "provenance": {
                        "carrier": "authored typed four-sentence frame",
                        "x_phrase": "Brown-corpus attested n-gram",
                        "y_phrase": "Brown-corpus attested n-gram",
                    },
                })
    rows.sort(key=lambda row: (
        not row["mechanically_admitted"],
        -(row["x_brown_occurrences"] + row["y_brown_occurrences"]),
        -row["independent_exact_audit"]["letters"],
        row["rendered"],
    ))
    return rows[:maximum_rows], {
        "indexed_tapes": len(by_tape),
        "ngram_occurrences": occurrences,
        "exact_phrase_joins": len(rows),
    }


def run(*, maximum_words: int = 5, maximum_rows: int = 200,
        typed_only: bool = True) -> dict:
    from nltk.corpus import brown

    sentences = [list(sentence) for sentence in brown.sents()]
    x_allowed = y_allowed = None
    if typed_only:
        x_allowed, y_allowed = _typed_domains(
            [list(sentence) for sentence in brown.tagged_sents()],
            maximum_words,
        )
    rows, stats = _phrase_rows(sentences, maximum_words=maximum_words,
                               maximum_rows=maximum_rows,
                               x_allowed=x_allowed, y_allowed=y_allowed)
    admitted = [row for row in rows if row["mechanically_admitted"]]
    return {
        "experiment_id": ID,
        "method": "exact hash join of attested phrases inside an authored staggered ABBA discourse frame",
        "fixed_equation": "reverse(tape(Y)) == 's' + tape(X)",
        "stats": {
            **stats,
            "typed_only": typed_only,
            "typed_x_phrases": len(x_allowed or ()),
            "typed_y_phrases": len(y_allowed or ()),
            "retained_rows": len(rows),
            "mechanically_admitted_rows": len(admitted),
        },
        "rows": rows,
        "reader_packet": [],
        "status": (
            "mechanical candidates require direct prose review before blinded readers"
            if admitted else "no mechanically admitted phrase join"
        ),
        "provenance": {
            "corpus": "NLTK Brown corpus; phrases are attested, not claimed as newly authored",
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "admission_policy": "programmatic checks filter only; readability requires blinded human ratings",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--maximum-words", type=int, default=5)
    parser.add_argument("--maximum-rows", type=int, default=200)
    parser.add_argument("--allow-untyped", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    result = run(maximum_words=args.maximum_words,
                 maximum_rows=args.maximum_rows,
                 typed_only=not args.allow_untyped)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stats": result["stats"],
                      "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
