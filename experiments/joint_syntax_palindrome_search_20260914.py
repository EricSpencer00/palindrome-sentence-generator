"""Joint syntax/character search for original English palindromes.

This experiment makes syntax part of the palindrome state rather than a
post-hoc score.  A state is legal only when its left suffix and right prefix
can still be completed by one of the authored, typed sentence plans.  The
center-out kernel simultaneously consumes the character residual, so a word
that is English in isolation but cannot occupy a live grammatical slot is
never expanded.  Brown transitions are a traversal prior only; exactness and
the final mechanical gates are independent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import (
    REPEATABLE_FUNCTION_WORDS,
    mechanical_admission_checks,
    normalize_letters,
)
from llm_palindrome.centerout import centerout_search
from llm_palindrome.search import WordTries


# These are sentence plans, not independent mirrored clauses.  The midpoint
# may fall inside any word or constituent.  PERSON/THING and transitive-verb
# slots are the smallest semantic guard that prevents a POS-shaped word salad.
PLANS: tuple[tuple[str, ...], ...] = (
    # A seed-shaped control: it is deliberately retained as a typed plan so
    # the joint search can replay the same constituent geometry while looking
    # for a longer lexicalization.  It is not itself a promoted result.
    ("DET", "PERSON", "VT", "NUM", "THING", "DET", "PERSON", "VT", "NAME"),
    ("DET", "ADJ", "PERSON", "VT", "NUM", "THING", "DET", "PERSON", "VT", "NAME"),
    ("DET", "PERSON", "VT", "DET", "THING", "CONJ", "DET", "PERSON", "VT", "NAME"),
    ("DET", "ADJ", "PERSON", "VT", "DET", "ADJ", "THING", "CONJ", "DET", "PERSON", "VT", "NAME"),
    # Generic noun frames widen the construction space without dropping the
    # live adjective/verb/noun typing.  They remain separate from
    # PERSON/THING so the audit can report which semantic guard admitted a
    # closure.
    ("DET", "ADJ", "NOUN", "VT", "DET", "NOUN"),
    ("DET", "NOUN", "VT", "DET", "ADJ", "NOUN"),
    ("NAME", "VT", "DET", "ADJ", "NOUN"),
    ("DET", "NOUN", "VT", "ADP", "DET", "NOUN"),
    ("DET", "ADJ", "NOUN", "VT", "DET", "NOUN", "CONJ", "DET", "NOUN"),
    ("DET", "PERSON", "VT", "DET", "THING"),
    ("DET", "ADJ", "PERSON", "VT", "DET", "THING"),
    ("DET", "PERSON", "VT", "DET", "ADJ", "THING"),
    ("NAME", "VT", "DET", "THING"),
    ("PRON", "VT", "DET", "THING"),
    ("DET", "PERSON", "VT", "ADP", "DET", "THING"),
    ("DET", "ADJ", "PERSON", "VT", "ADP", "DET", "THING"),
    ("DET", "PERSON", "VT", "DET", "THING", "ADP", "DET", "THING"),
    ("ADV", "DET", "PERSON", "VT", "DET", "THING"),
    ("PRON", "VT", "DET", "ADJ", "THING", "ADP", "DET", "THING"),
    ("DET", "PERSON", "VT", "DET", "THING", "CONJ", "PRON", "VT", "THING"),
    ("NAME", "VT", "DET", "THING", "CONJ", "DET", "PERSON", "VT"),
    ("DET", "ADJ", "PERSON", "VT", "DET", "ADJ", "THING", "CONJ", "PRON", "VT"),
    ("PRON", "VT", "DET", "THING", "CONJ", "DET", "PERSON", "VT", "DET", "THING"),
)

DETS = "a an the my our his her this that some many each one their".split()
NUMS = "one two three four five six seven eight nine ten".split()
PRONS = "i we you he she they it me us them who".split()
NAMES = "alice amelia anna ben bob carla clara damon diana emma eva grace helen iris jane laura liam leon lucy maria maya nina noel olivia paul peter sara sophia".split()
PERSONS = "aide artist baker child cook doctor farmer friend gardener guard helper man men parent poet pupil teacher worker writer reader singer dancer pilot editor nurse neighbor sailor woman girl boy mother father sister brother people team family".split()
THINGS = "note notes book books map maps letter letters memo memos plan plans key keys door doors song songs story stories task tasks test tests cup cake room garden bread tool horse dog cat bird gate road river flower apple drawer star time diary school town night rain wind fire water light mail message table chair house home car card money truth music place life work hand part age idea issue reason cause result event film group name answer code data line word reward devil dessert desserts tram trams".split()
ADJS = "able alive angry ancient blue brave bright calm careful clear clever cold dark eager fair false fast gentle good great green happy hard kind large late little lively long new nice old open patient plain quick quiet red rich safe sharp short smart soft strong sure true warm wise young".split()
ADPS = "at in on by for with from near after before under over into through around beside beyond".split()
CONJS = "and but or yet".split()
ADVS = "again always away early here never now often outside slowly then today well".split()
VERBS = "aided asked baked built called carried changed cleaned closed cooked drew drove ate found fixed helped held kept learned liked listened loved made marked met moved noticed opened painted planned read repaired rescued saved saw sent shared showed studied taught thanked told used visited watched wrote rip rips ripped inspire inspired served recorded tested worked mailed rewarded repaid stressed started ended stated gave took put brought left meant needed knew".split()


POOLS: dict[str, tuple[str, ...]] = {
    "DET": tuple(DETS), "PRON": tuple(PRONS), "NAME": tuple(NAMES),
    "PERSON": tuple(PERSONS), "THING": tuple(THINGS), "ADJ": tuple(ADJS),
    "ADP": tuple(ADPS), "CONJ": tuple(CONJS), "ADV": tuple(ADVS),
    "VT": tuple(VERBS), "NUM": tuple(NUMS),
}
POOLS["NOUN"] = tuple(dict.fromkeys(POOLS["PERSON"] + POOLS["THING"]))


def _tags(word: str) -> frozenset[str]:
    """Return all authored lexical categories for a word."""
    return frozenset(tag for tag, words in POOLS.items() if word in words)


TAG_BY_WORD = {word: _tags(word) for words in POOLS.values() for word in words}


def _matches_shape(words: Sequence[str], shape: Sequence[str], *, prefix: bool) -> bool:
    if len(words) > len(shape):
        return False
    expected = shape[:len(words)] if prefix else shape[-len(words):]
    return all(slot in TAG_BY_WORD.get(word, ()) for word, slot in zip(words, expected))


def syntax_state_possible(left: Sequence[str], right: Sequence[str]) -> bool:
    """Check whether both live grammar edges can occupy one plan.

    Center-out growth starts at the eventual clause boundary.  The current
    ``left`` tuple is therefore a *suffix of the yet-to-be-completed left
    prefix* (new outer words are prepended), while ``right`` is a prefix of the
    right suffix (new outer words are appended).  We retain the possible split
    point explicitly instead of pretending either edge is already at an
    absolute sentence position.
    """
    if len(left) + len(right) > max(map(len, PLANS)):
        return False
    for shape in PLANS:
        if len(left) + len(right) > len(shape):
            continue
        # k is the split between the left and right portions of the final
        # sentence.  The current left edge occupies [k-len(left):k], and the
        # current right edge occupies [k:k+len(right)].
        for k in range(len(left), len(shape) - len(right) + 1):
            left_slots = shape[k - len(left):k]
            right_slots = shape[k:k + len(right)]
            if all(slot in TAG_BY_WORD.get(word, ())
                   for word, slot in zip(left, left_slots)) and \
               all(slot in TAG_BY_WORD.get(word, ())
                   for word, slot in zip(right, right_slots)):
                return True
    return not left and not right


def syntax_complete(words: Sequence[str]) -> bool:
    return any(len(words) == len(shape)
               and _matches_shape(words, shape, prefix=True)
               for shape in PLANS)


def vocab() -> list[str]:
    words = set(word for values in POOLS.values() for word in values)
    # Exclude self-palindromic *content* units, while retaining ordinary
    # one-letter function words such as ``a`` and ``i`` that a grammatical
    # sentence may legitimately need.  This is a construction guard, not a
    # readability certificate.
    return sorted(word for word in words
                  if word == word.casefold() and word.isalpha()
                  and (word != word[::-1] or word in REPEATABLE_FUNCTION_WORDS))


def brown_bigrams() -> Counter[tuple[str, str]]:
    try:
        from nltk.corpus import brown
        out: Counter[tuple[str, str]] = Counter()
        for sentence in brown.sents():
            words = [word.casefold() for word in sentence
                     if word.isascii() and word.isalpha()]
            out.update(zip(words, words[1:]))
        return out
    except LookupError:
        return Counter()


class BrownJoinScorer:
    """Soft adjacent-word prior used *inside* the typed search.

    It ranks legal children but never admits one: syntax and exact character
    residuals remain hard state constraints, and the resulting score is not a
    readability certificate.  A missing Brown corpus simply gives every join
    a neutral score, preserving a reproducible grammar/character run.
    """

    def __init__(self, counts: Counter[tuple[str, str]]) -> None:
        self.counts = counts
        self.total = max(1, sum(counts.values()))

    def word_delta(self, left, right, placement, word, growth):
        if placement == "L":
            if len(left) < 2:
                return 0.0
            pair = (left[0], left[1])
        else:
            if len(right) < 2:
                return 0.0
            pair = (right[-2], right[-1])
        # log smoothing prevents a single frequent join from overwhelming
        # length-normalized ranking while leaving unseen joins neutral.
        return 0.25 * math.log1p(self.counts.get(pair, 0))


def run(*, seeds: int = 64, beam: int = 600, candidate_limit: int = 500,
        max_steps: int = 120, min_letters: int = 39,
        max_letters: int = 140) -> dict[str, object]:
    words = vocab()
    tries = WordTries(words)
    bigrams = brown_bigrams()
    scorer = BrownJoinScorer(bigrams)
    records: list[dict[str, object]] = []
    closure_count = 0

    def allow_state(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
        all_words = list(left + right)
        content = [word for word in all_words
                   if word not in REPEATABLE_FUNCTION_WORDS]
        if len(content) != len(set(content)):
            return False
        if not syntax_state_possible(left, right):
            return False
        return True

    def allow_closed(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
        words = left + right
        if not syntax_complete(words):
            return False
        text = " ".join(words)
        tape = normalize_letters(text)
        # Collect every exact typed closure, including short controls.  The
        # configured length band is reported on each row and decides promotion
        # independently; otherwise a short exact replay disappears before we
        # can diagnose whether the grammar or the character search failed.
        return len(tape) <= max_letters and bool(tape) and tape == tape[::-1]

    for seed in range(seeds):
        closed: list[list[str]] = []
        centerout_search(
            tries, scorer, min_letters=min_letters, max_steps=max_steps,
            beam_width=beam, candidate_limit=candidate_limit,
            # Keep a broad parent frontier: a narrow per-parent menu lets a
            # high-scoring local join erase the only exact long closure before
            # the grammar/character product has a chance to finish it.
            per_parent=min(beam, 256), seed=seed, diversity=0.45,
            max_overhang=40,
            allow_state=allow_state, allow_closed=allow_closed,
            on_closed=lambda candidate: closed.append(candidate),
        )
        closure_count += len(closed)
        for candidate in closed:
            text = " ".join(candidate)
            tape = normalize_letters(text)
            checks = mechanical_admission_checks(text, min_letters=min_letters,
                                                  max_letters=max_letters)
            independent = "".join(ch.casefold() for ch in text
                                   if ch.isascii() and ch.isalpha())
            records.append({
                "rendered": text,
                "words": candidate,
                "letters": len(tape),
                "normalized_letters": tape,
                "independent_normalized_letters": independent,
                "independent_exact_audit": bool(tape) and tape == tape[::-1] and tape == independent,
                "syntax_complete": syntax_complete(candidate),
                "mechanical_checks": checks,
                "mechanically_eligible": bool(all(checks.values())),
                "seed": seed,
                "reader_status": "human-unreviewed; programmatic filters do not certify readability",
            })

    unique = {}
    for row in records:
        unique.setdefault(row["normalized_letters"], row)
    return {
        "status": "joint_syntax_character_search",
        "config": {
            "seeds": seeds, "beam": beam, "candidate_limit": candidate_limit,
            "max_steps": max_steps, "minimum_letters": min_letters,
            "maximum_letters": max_letters, "plans": [list(p) for p in PLANS],
            "syntax_is_live_state_constraint": True,
            "character_residual_is_live_state_constraint": True,
            "max_overhang": 40,
            "typed_state_roles": ["VT", "NOUN", "PERSON", "THING", "ADJ", "DET", "NUM", "ADP", "CONJ", "PRON", "NAME"],
            "brown_bigrams_are_traversal_filter_only": True,
            "machine_readability_certification": False,
            "no_catalogue_text": True,
        },
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "lexical_source": "authored typed role inventory",
            "brown_source": "join filter only; no Brown sentence copied",
        },
        "closure_count": closure_count,
        "records": list(unique.values()),
        "mechanically_eligible": [row for row in unique.values()
                                   if row["mechanically_eligible"]],
        "next_operator_if_empty": "Expand the typed plan inventory at the deepest replayed residual while preserving joint syntax and character constraints; do not relax to free tape scoring.",
        "reader_facing_next_test": "Do not send the 38-letter seed control or any machine-only closure to readers; first promote a novel >=39-letter mechanically eligible closure, then run randomized blinded intact-prose and shuffled controls with the reproducible rater package.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seeds", type=int, default=64)
    parser.add_argument("--beam", type=int, default=600)
    parser.add_argument("--candidate-limit", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=120)
    args = parser.parse_args()
    result = run(seeds=args.seeds, beam=args.beam,
                 candidate_limit=args.candidate_limit,
                 max_steps=args.max_steps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "closures": result["closure_count"],
                      "records": len(result["records"]),
                      "mechanically_eligible": len(result["mechanically_eligible"])}))


if __name__ == "__main__":
    main()
