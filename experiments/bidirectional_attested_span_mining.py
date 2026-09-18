"""Mine long exact mirror pairs whose two sides are independently attested.

This is a construction experiment, not a readability score.  It indexes every
ordinary-language span in a fixed corpus, then intersects the index with its
letter reversal.  A reported pair therefore has two independently attested
English readings *before* any judge sees it, while exact palindromy is checked
directly from the rendered text.

The useful unit is a 15--30-letter side (30--60 letters total): short enough
to enumerate faithfully, long enough that a surviving pair could carry more
than a fragment.  The output is a reproducible candidate inventory for a
blinded reader study, never evidence that a candidate is readable.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Callable, Iterable, Iterator, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.paragraphs import is_novel_palindrome
from llm_palindrome.validator import is_palindrome, normalize


def _letters(words: Sequence[str]) -> str:
    return "".join(words)


def brown_sentences() -> list[list[str]]:
    """Frozen local Brown material, lowercased and stripped to word tokens."""
    from nltk.corpus import brown

    return [[word.lower() for word in sentence if word.isalpha()]
            for sentence in brown.sents()]


def wikitext_ngrams() -> list[list[str]]:
    """The project's frozen 3--6-gram inventory, treated as attested spans."""
    data = json.loads((ROOT / "data" / "ngrams_wikitext2.json").read_text())
    return [phrase.lower().split()
            for length in ("3", "4", "5", "6")
            for phrase in data[length]]


def observed_surface_forms(min_zipf: float, lexicon: set[str]) -> set[str]:
    """Return attested ordinary forms, including regular inflections.

    ``lexicon.txt`` is a headword list.  ``is_real_word`` deliberately knows
    how to accept forms such as ``rats`` and ``writes``, but that knowledge was
    previously unreachable here because we iterated only over headwords.  Use
    two fixed local observations as the source of possible *surface forms* and
    then apply the same conservative dictionary rule.  This does not invent
    inflections; a form must occur in the Brown corpus or the frozen wordfreq
    top-100k list before it is available to a constructor.
    """
    from nltk.corpus import brown
    from wordfreq import top_n_list, zipf_frequency
    from llm_palindrome.lexicon import is_real_word

    observed = {
        word.casefold() for word in brown.words()
        if word.isascii() and word.isalpha()
    }
    observed.update(
        word.casefold() for word in top_n_list("en", 100_000)
        if word.isascii() and word.isalpha()
    )
    return {
        word for word in observed
        if zipf_frequency(word, "en") >= min_zipf and is_real_word(word, lexicon)
    }


def common_lexicon(min_zipf: float) -> set[str]:
    """A conservative word-form gate, applied before indexing spans.

    Headwords and observed regular surface forms are both included.  Keeping
    the latter separate makes it auditable that a candidate form was observed,
    rather than mechanically manufactured from a lemma.
    """
    from llm_palindrome.lexicon import is_real_word, load_lexicon
    from wordfreq import zipf_frequency

    lexicon = load_lexicon(str(ROOT / "data" / "lexicon.txt"))
    headwords = {word for word in lexicon
                 if word.isalpha() and zipf_frequency(word, "en") >= min_zipf
                 and is_real_word(word, lexicon)}
    return headwords | observed_surface_forms(min_zipf, lexicon) | {"a", "i"}


def attested_spans(sentences: Iterable[Sequence[str]], *, vocab: set[str],
                   min_words: int, max_words: int, min_letters: int,
                   max_letters: int) -> Iterator[dict]:
    """Yield every admissible in-sentence span with stable provenance."""
    for sentence_id, sentence in enumerate(sentences):
        for start in range(len(sentence)):
            words: list[str] = []
            for end in range(start, min(len(sentence), start + max_words)):
                word = sentence[end]
                if word not in vocab:
                    break
                words.append(word)
                tape = _letters(words)
                if len(tape) > max_letters:
                    break
                if len(words) >= min_words and len(tape) >= min_letters:
                    yield {
                        "sentence_id": sentence_id,
                        "start": start,
                        "end": end + 1,
                        "words": tuple(words),
                        "text": " ".join(words),
                        "letters": tape,
                    }


def intersect_reverse_spans(rows: Iterable[dict], *,
                            novel_checker: Callable[[str], bool] = is_novel_palindrome
                            ) -> tuple[dict, list[dict]]:
    """Intersect independently attested spellings with their reversed tape."""
    index: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    total = 0
    for row in rows:
        total += 1
        index[row["letters"]][row["text"]].append(row)

    pairs: list[dict] = []
    checked = 0
    for tape in sorted(index):
        mirror = tape[::-1]
        if tape >= mirror or mirror not in index:
            continue
        for left, left_rows in index[tape].items():
            for right, right_rows in index[mirror].items():
                checked += 1
                text = f"{left} {right}"
                assert normalize(left) == normalize(right)[::-1]
                assert is_palindrome(text)
                pairs.append({
                    "left": left,
                    "right": right,
                    "text": text,
                    "letters_per_side": len(tape),
                    "left_attestations": len(left_rows),
                    "right_attestations": len(right_rows),
                    "novel_catalogue": novel_checker(text),
                    "exact_palindrome": True,
                })
    stats = {
        "indexed_spans": total,
        "distinct_letter_tapes": len(index),
        "reverse_compatible_tapes": sum(
            1 for tape in index if tape < tape[::-1] and tape[::-1] in index),
        "candidate_pairs": checked,
        "novel_candidate_pairs": sum(row["novel_catalogue"] for row in pairs),
    }
    return stats, pairs


def run(*, source: str, min_words: int, max_words: int, min_letters: int,
        max_letters: int, min_zipf: float) -> dict:
    sources = {
        "brown": ("NLTK Brown corpus installed locally", brown_sentences),
        "wikitext2": ("project's frozen WikiText-2 n-gram inventory", wikitext_ngrams),
    }
    if source not in sources:
        raise ValueError(f"unknown source {source!r}; choose one of {sorted(sources)}")
    source_label, load = sources[source]
    sentences = load()
    vocab = common_lexicon(min_zipf)
    stats, pairs = intersect_reverse_spans(attested_spans(
        sentences, vocab=vocab, min_words=min_words, max_words=max_words,
        min_letters=min_letters, max_letters=max_letters))
    return {
        "status": "complete_attested_material_inventory_not_readability_evidence",
        "source": source_label,
        "config": {
            "source": source,
            "min_words": min_words,
            "max_words": max_words,
            "min_letters_per_side": min_letters,
            "max_letters_per_side": max_letters,
            "min_zipf": min_zipf,
        },
        "source_spans": len(sentences),
        "vocabulary_words": len(vocab),
        "mechanical": stats,
        "pairs": pairs,
        "reader_gate": (
            "Every returned side is an independently attested corpus span and every combined "
            "string is exact by construction. Readability and discourse coherence still require "
            "a blinded reader study with intact prose and shuffled controls."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", choices=("brown", "wikitext2"), default="brown")
    parser.add_argument("--min-words", type=int, default=3)
    parser.add_argument("--max-words", type=int, default=8)
    parser.add_argument("--min-letters", type=int, default=15)
    parser.add_argument("--max-letters", type=int, default=30)
    parser.add_argument("--min-zipf", type=float, default=3.6)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = run(source=args.source, min_words=args.min_words, max_words=args.max_words,
                 min_letters=args.min_letters, max_letters=args.max_letters,
                 min_zipf=args.min_zipf)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.out),
        "indexed_spans": result["mechanical"]["indexed_spans"],
        "candidate_pairs": result["mechanical"]["candidate_pairs"],
        "novel_candidate_pairs": result["mechanical"]["novel_candidate_pairs"],
    }, indent=2))


if __name__ == "__main__":
    main()
