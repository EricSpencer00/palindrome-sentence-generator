"""Audit old exact tapes with an independent, stronger sentence-shape screen.

This is deliberately an audit, not a readability certificate.  It searches
the existing JSON artifacts for exact letter palindromes, independently
re-tokenizes them, applies the shared fail-closed construction checks, and
then asks whether the visible word sequence contains at least one ordinary
finite-clause shape.  The latter is only a diagnostic: passing it would still
require blinded readers.

The useful failure mode is explicit: a tape can be perfectly lexical and
exact while having no plausible clause spine.  Those tapes are retained as
near-misses together with their provenance and the next construction operator.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from wordfreq import zipf_frequency

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "existing-exact-tape-audit-20260915.json"
WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def _walk(value: Any, path: str = ""):
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _walk(child, f"{path}/{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, f"{path}/{index}")


def _pos_tags(words: tuple[str, ...]) -> tuple[str, ...]:
    # NLTK's tagger is a fixed local resource in the experiment environment.
    # Keep a conservative fallback so the audit remains reproducible on a
    # checkout without that optional model.
    try:
        import nltk

        return tuple(tag for _, tag in nltk.pos_tag(list(words)))
    except Exception:
        verbs = {
            "am", "is", "are", "was", "were", "be", "been", "being",
            "do", "does", "did", "have", "has", "had", "make", "makes",
            "made", "see", "saw", "say", "said", "go", "went", "get",
            "got", "set", "put", "let", "live", "lived", "draw", "drew",
        }
        return tuple("VB" if word in verbs else "NN" for word in words)


@lru_cache(maxsize=1)
def _brown_lexicon() -> dict[str, str]:
    try:
        from nltk.corpus import brown

        counts: dict[str, Counter[str]] = {}
        for word, tag in brown.tagged_words(tagset="universal"):
            counts.setdefault(word.casefold(), Counter())[tag] += 1
        return {word: max(counter, key=counter.get) for word, counter in counts.items()}
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _brown_bigrams() -> frozenset[tuple[str, str]]:
    try:
        from nltk.corpus import brown

        words = [word.casefold() for word in brown.words() if WORD.fullmatch(word)]
        return frozenset(zip(words, words[1:]))
    except Exception:
        return frozenset()


def _lexical_universal_tags(words: tuple[str, ...]) -> tuple[str, ...]:
    """Assign Brown-corpus majority tags, independent of candidate context."""
    try:
        counts = _brown_lexicon()
        return tuple(
            counts.get(word, "X")
            for word in words
        )
    except Exception:
        return tuple("VERB" if tag.startswith("VB") else "NOUN" for tag in _pos_tags(words))


def _clause_shape(tags: tuple[str, ...]) -> tuple[bool, str]:
    """Conservative finite-clause spine diagnostic.

    We permit punctuation-free runs because punctuation is outside the tape,
    but require a subject-like opening and a finite verb after it.  This does
    not parse a sentence and is intentionally weaker than a human judgment.
    """
    if not tags:
        return False, "empty"
    subject = {"NOUN", "PRON", "DET", "X"}
    finite = {"VERB", "AUX"}
    # Split at coordinating conjunctions and punctuation-free clause joins are
    # still accepted as one run.  Require every run to carry a subject and a
    # verb; a long stack of adjectives/nouns should not pass by frequency.
    verb_positions = [i for i, tag in enumerate(tags) if tag in finite]
    if not verb_positions:
        return False, "no_finite_verb"
    first_verb = verb_positions[0]
    if first_verb == 0 or not any(tag in subject for tag in tags[:first_verb]):
        return False, "no_subject_before_verb"
    # Reject the characteristic reverse-tape piles: a second verb cannot be
    # followed only by prepositions/articles and a mirror tail.
    if len(verb_positions) == 1 and len(tags) >= 8:
        tail = tags[first_verb + 1 :]
        if sum(tag in {"ADP", "DET", "PRON", "ADV", "ADJ"} for tag in tail) >= len(tail) - 1:
            return False, "verb_without_argument_spine"
    return True, "finite_clause_spine"


def _rendered_clause_shape(text: str) -> tuple[bool, str]:
    """Apply the spine test separately to visible punctuation-delimited runs."""
    chunks = [chunk for chunk in re.split(r"[.!?;:]", text) if WORD.search(chunk)]
    if not chunks:
        return False, "empty"
    reasons: list[str] = []
    for chunk in chunks:
        words = tokenize(chunk)
        ok, reason = _clause_shape(_lexical_universal_tags(words))
        if not ok:
            reasons.append(reason)
    if reasons:
        return False, ";".join(dict.fromkeys(reasons))
    return True, "finite_clause_spine_each_visible_clause"


def _bigram_diagnostic(words: tuple[str, ...]) -> tuple[float, bool]:
    """Measure local attestation; this never certifies sentence readability."""
    bigrams = _brown_bigrams()
    if len(words) < 2 or not bigrams:
        return 0.0, False
    rate = sum(pair in bigrams for pair in zip(words, words[1:])) / (len(words) - 1)
    return round(rate, 4), rate >= 0.35


def _candidate_score(text: str, words: tuple[str, ...], tags: tuple[str, ...]) -> float:
    freqs = [zipf_frequency(word, "en") for word in words]
    known = sum(freq >= 2.5 for freq in freqs) / max(1, len(freqs))
    common = sum(freq >= 3.5 for freq in freqs) / max(1, len(freqs))
    short = sum(len(word) <= 2 for word in words) / max(1, len(words))
    repeats = (len(words) - len(set(words))) / max(1, len(words))
    verbs = sum(tag.startswith("VB") or tag == "MD" for tag in tags)
    return 4 * known + 2 * common + sum(freqs) / max(1, len(freqs)) + min(verbs, 3) - 5 * short - 2 * repeats


def collect(limit: int = 120) -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for path in sorted(RUNS.rglob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for json_path, text in _walk(payload):
            try:
                tape = normalize_letters(text)
            except ValueError:
                continue
            words = tokenize(text)
            if len(tape) < 39 or len(tape) > 180 or len(words) < 5:
                continue
            if tape != tape[::-1] or tape in seen:
                continue
            # Avoid treating a prose paragraph from a control or a borrowed
            # corpus as a generated proposal.  Only compact renderings with
            # ordinary tokenization are admitted to this audit shortlist.
            if len(text) > 2_000 or any(len(word) > 24 for word in words):
                continue
            seen.add(tape)
            context_tags = _pos_tags(words)
            tags = _lexical_universal_tags(words)
            checks = mechanical_admission_checks(text, min_letters=39, max_letters=180)
            shape_ok, shape_reason = _rendered_clause_shape(text)
            bigram_rate, bigram_ok = _bigram_diagnostic(words)
            score = _candidate_score(text, words, tags)
            rows.append(
                {
                    "rendered": text,
                    "letters": len(tape),
                    "normalized_letters": tape,
                    "sha256": hashlib.sha256(tape.encode()).hexdigest(),
                    "provenance": {"file": str(path.relative_to(ROOT)), "json_path": json_path},
                    "words": list(words),
                    "pos_tags": list(context_tags),
                    "lexical_pos_tags": list(tags),
                    "mechanical_checks": checks,
                    "independent_exact": tape == tape[::-1],
                    "clause_spine": shape_ok,
                    "clause_spine_reason": shape_reason,
                    "brown_bigram_attestation": bigram_rate,
                    "brown_bigram_diagnostic_pass": bigram_ok,
                    "reader_status": "not_run",
                    "diagnostic_score": round(score, 4),
                    "next_operator": (
                        "retain exact tape as a near-miss and search a grammar-constrained boundary resegmentation"
                        if not (shape_ok and bigram_ok)
                        else "manual review, then blinded readers; no programmatic readability claim"
                    ),
                }
            )
    rows.sort(key=lambda row: (row["clause_spine"], row["diagnostic_score"], row["letters"]), reverse=True)
    return rows[:limit]


def main() -> None:
    rows = collect()
    mechanically_admitted = [row for row in rows if all(row["mechanical_checks"].values())]
    clause_candidates = [row for row in mechanically_admitted if row["clause_spine"]]
    report = {
        "status": "audit_complete_no_reader_promotion",
        "method": {
            "source": "all runs/**/*.json strings",
            "independent_normalization": "ASCII letters only, local re-tokenization",
            "grammar_diagnostic": "NLTK POS tags plus conservative subject-before-finite-verb spine",
            "readability_claim": "none; only blinded readers can certify readability",
            "shortlist_limit": 120,
        },
        "counts": {
            "shortlist": len(rows),
            "mechanically_admitted": len(mechanically_admitted),
            "with_clause_spine": len(clause_candidates),
            "reader_worthy_outputs": 0,
        },
        "near_misses": rows[:50],
        "strongest_mechanical_clause_rows": clause_candidates[:20],
        "conclusion": (
            "No historical exact tape was promoted. The strongest lexical rows are seam or POS-lattice artifacts; "
            "the next constructive operator is grammar-constrained boundary resegmentation with human review."
        ),
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["counts"], indent=2))
    for row in rows[:12]:
        print(f"{row['letters']:3d} spine={row['clause_spine']} mech={all(row['mechanical_checks'].values())} :: {row['rendered']}")


if __name__ == "__main__":
    main()
