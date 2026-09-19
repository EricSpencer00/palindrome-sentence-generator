#!/usr/bin/env python3
"""Typed grammar intersection with character obligations.

Each candidate is assembled from complete typed word slots on two *different*
clause templates.  The search grows from the outside in: a word is legal only
when its letters consume the live reverse-tape obligation, and the role at the
next grammar position is checked before the state is enqueued.  No finished
candidate is reversed and no catalogue palindrome is used as a seed.

The experiment is intentionally small enough to audit in one run.  Its JSON
output keeps the rendered candidates, exact two-pointer audit, provenance, and
the next repair operator even when no new admission survives.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
import sys
from collections import Counter
from pathlib import Path

from nltk.corpus import wordnet as wn
from wordfreq import top_n_list, zipf_frequency

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/typed-grammar-equation-search-20260919.json"
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks  # noqa: E402
from llm_palindrome.lexicon import is_real_word, load_lexicon  # noqa: E402
from llm_palindrome.search import consume  # noqa: E402

LEXICON = load_lexicon(str(ROOT / "data/lexicon.txt"))


def letters(text: str) -> str:
    return "".join(re.findall(r"[a-z]", text.casefold()))


def audit(text: str) -> dict:
    tape = letters(text)
    mismatches = [(i, len(tape) - 1 - i, tape[i], tape[-1 - i])
                  for i in range(len(tape) // 2)
                  if tape[i] != tape[-1 - i]]
    return {
        "letters": len(tape),
        "exact": bool(tape) and not mismatches,
        "independent_two_pointer": all(
            tape[i] == tape[-1 - i] for i in range(len(tape) // 2)
        ),
        "first_mismatch": mismatches[0] if mismatches else None,
        "mismatch_count": len(mismatches),
        "sha256": hashlib.sha256(tape.encode()).hexdigest(),
    }


def wordnet_bank(pos: str, limit: int) -> list[str]:
    """Return frequent, alphabetic WordNet lemmas in deterministic order."""
    words: set[str] = set()
    for lemma in wn.all_lemma_names(pos=pos):
        word = lemma.casefold().replace("_", "")
        if (
            word.isascii()
            and word.isalpha()
            and len(word) >= 2
            and zipf_frequency(word, "en") >= 3.55
        ):
            words.add(word)
    return sorted(words, key=lambda w: (-zipf_frequency(w, "en"), len(w), w))[:limit]


def banks() -> dict[str, tuple[str, ...]]:
    """Build role banks without importing a finished palindrome."""
    # WordNet is lemma-oriented, so ordinary inflections needed by intact
    # present-tense clauses (``memos``, ``rips``, ``writes``) are added from a
    # small transparent role list rather than silently generated from a
    # completed palindrome.  These are vocabulary options, not output text.
    extra_nouns = "aide memo memos men man letter map note notes shore boat bell garden river keeper sailor poet artist captain courier arena area idea era media camera drama agenda flora formula quota villa swan book bridge child clerk dawn doctor door gate harbor home lantern messenger road scribe singer teacher witness roses".split()
    extra_verbs = "rips inspire inspires writes reads keeps carries marks guides opens guards brings waits measures sees meets draws repairs stores hears charts records copies asks bakes calls cleans counts finds helps leads lifts makes paints rides sends sets sings takes tells tests uses watches".split()
    safe_adjectives = "bright careful calm clear gentle kind old patient quiet red small steady swift young wise green silent warm".split()
    # A suffix-indexed noun supplement supplies ordinary words ending in ``a``
    # or ``na``; those endings are the natural entry points for an outer
    # ``a``/``an`` article on the opposite side of the equation.
    suffix_nouns = [
        "arena", "area", "camera", "drama", "era", "flora", "formula",
        "idea", "media", "quota", "villa",
    ]
    raw = {
        "DET": tuple("a an the some this that my your each".split()),
        "PRON": tuple("i we he she they it you".split()),
        "NUM": tuple("one two three four five nine ten".split()),
        "PREP": tuple("at by in on near with from for to after before over under".split()),
        "PROPN": tuple(
            "diana noel leon pam eros nomad mara nora anna lana sara ada maria "
            "leona elena aria olivia sophia nina rosa luna maya clara susan alan adam".split()
        ),
        "CONJ": tuple("and but or".split()),
        "NOUN": tuple(extra_nouns + suffix_nouns),
        "VERB": tuple(extra_verbs),
        "ADJ": tuple(safe_adjectives),
        "ADV": tuple(wordnet_bank("r", 50)),
    }
    # Keep every surface form independently lexical under the same dictionary
    # used by admission.  This prevents a character equation from promoting
    # an accidental abbreviation (for example ``dna``) as a noun.
    return {
        role: tuple(word for word in words if is_real_word(word, LEXICON))
        for role, words in raw.items()
    }


# These are ordinary clause shapes, not mirrored slots.  The right-hand
# variants end in a name so the outer character equation can cross a natural
# sentence boundary without forcing a self-palindromic content word.
TEMPLATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("det_noun_verb_num_noun", ("DET", "NOUN", "VERB", "NUM", "NOUN")),
    ("det_adj_noun_verb_det_noun", ("DET", "ADJ", "NOUN", "VERB", "DET", "NOUN")),
    ("det_noun_verb_det_noun", ("DET", "NOUN", "VERB", "DET", "NOUN")),
    ("det_noun_verb_det_noun_propn", ("DET", "NOUN", "VERB", "DET", "NOUN", "PROPN")),
    ("det_noun_verb_prep_det_noun", ("DET", "NOUN", "VERB", "PREP", "DET", "NOUN")),
    ("det_noun_verb_num_adj_noun", ("DET", "NOUN", "VERB", "NUM", "ADJ", "NOUN")),
    ("det_noun_verb_det_adj_noun", ("DET", "NOUN", "VERB", "DET", "ADJ", "NOUN")),
    ("det_adj_noun_verb_propn", ("DET", "ADJ", "NOUN", "VERB", "PROPN")),
    ("det_noun_verb_prep_propn", ("DET", "NOUN", "VERB", "PREP", "PROPN")),
    ("det_noun_verb_propn", ("DET", "NOUN", "VERB", "PROPN")),
    ("det_adj_noun_verb_det_noun_propn", ("DET", "ADJ", "NOUN", "VERB", "DET", "NOUN", "PROPN")),
    ("det_noun_verb_det_noun_prep_det_noun", ("DET", "NOUN", "VERB", "DET", "NOUN", "PREP", "DET", "NOUN")),
    ("det_noun_verb_prep_det_noun_propn", ("DET", "NOUN", "VERB", "PREP", "DET", "NOUN", "PROPN")),
    ("det_adj_noun_verb_det_noun_prep_propn", ("DET", "ADJ", "NOUN", "VERB", "DET", "NOUN", "PREP", "PROPN")),
    ("pron_verb_det_noun_prep_det_noun", ("PRON", "VERB", "DET", "NOUN", "PREP", "DET", "NOUN")),
    ("det_noun_verb_conj_det_noun_propn", ("DET", "NOUN", "VERB", "CONJ", "DET", "NOUN", "PROPN")),
)


def content_words(words: tuple[str, ...], role_banks: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    function = set(role_banks["DET"] + role_banks["PRON"] + role_banks["NUM"]
                   + role_banks["PREP"] + role_banks["ADV"])
    return tuple(word for word in words if word not in function)


def search_pair(
    left_pattern: tuple[str, ...],
    right_pattern: tuple[str, ...],
    role_banks: dict[str, tuple[str, ...]],
    *,
    node_limit: int = 300_000,
    result_limit: int = 20,
) -> tuple[list[dict], dict]:
    """Enumerate typed exact intersections with memoized residual states."""
    results: list[dict] = []
    visited: set[tuple[tuple[str, ...], tuple[str, ...], str, str]] = set()
    cache: dict[tuple[str, str, str], tuple[tuple[str, str, bool], ...]] = {}
    nodes = 0

    def matches(role: str, debt: str, side: str) -> tuple[tuple[str, str, bool], ...]:
        key = (role, debt, side)
        if key in cache:
            return cache[key]
        rows = []
        for word in role_banks[role]:
            result = consume(word[::-1] if side == "R" else word, debt)
            if result is not None:
                remainder, flipped = result
                rows.append((word, remainder, flipped))
        cache[key] = tuple(rows)
        return cache[key]

    repeatable = set(
        role_banks["DET"] + role_banks["PRON"] + role_banks["NUM"]
        + role_banks["PREP"] + role_banks["ADV"] + role_banks["CONJ"]
    )

    def visit(left: tuple[str, ...], right: tuple[str, ...], debt: str, side: str) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > node_limit or len(results) >= result_limit:
            return
        state = (left, right, debt, side)
        if state in visited:
            return
        visited.add(state)
        if len(left) == len(left_pattern) and len(right) == len(right_pattern):
            if not debt:
                rendered = " ".join(left) + "; " + " ".join(right) + "."
                checked = audit(rendered)
                admission = mechanical_admission_checks(
                    rendered, min_letters=39, max_letters=180
                )
                results.append({
                    "rendered": rendered,
                    "left_pattern": left_pattern,
                    "right_pattern": right_pattern,
                    "left_words": left,
                    "right_words": right,
                    "audit": checked,
                    "admission": admission,
                    "mechanically_admitted": checked["exact"] and all(admission.values()),
                    "provenance": {
                        "lane": "typed-grammar-equation-search-20260919",
                        "construction": "independently typed clause slots with live reverse-tape obligations",
                        "catalogue_seed": False,
                        "finished_tape_reversal": False,
                    },
                })
            return
        if len(left) >= len(left_pattern) and len(right) >= len(right_pattern):
            return

        # The established exact-search invariant: when the right side owns
        # the debt, append the next left slot; otherwise prepend the next right
        # slot in final reading order.
        if side == "L" or not debt:
            if len(right) < len(right_pattern):
                role = right_pattern[len(right_pattern) - len(right) - 1]
                for word, remainder, flipped in matches(role, debt, "R"):
                    new_right = (word,) + right
                    all_words = left + new_right
                    content = content_words(all_words, role_banks)
                    if len(content) != len(set(content)):
                        continue
                    visit(left, new_right, remainder, "R" if flipped else "L")
        if side == "R" and debt and len(left) < len(left_pattern):
            role = left_pattern[len(left)]
            for word, remainder, flipped in matches(role, debt, "L"):
                new_left = left + (word,)
                all_words = new_left + right
                content = content_words(all_words, role_banks)
                if len(content) != len(set(content)):
                    continue
                visit(new_left, right, remainder, "L" if flipped else "R")

    visit((), (), "", "L")
    return results, {
        "nodes": nodes,
        "visited": len(visited),
        "cached_match_queries": len(cache),
    }


def main() -> None:
    role_banks = banks()
    records: list[dict] = []
    stats: list[dict] = []
    for (left_name, left_pattern), (right_name, right_pattern) in itertools.product(TEMPLATES, TEMPLATES):
        rows, info = search_pair(left_pattern, right_pattern, role_banks)
        stats.append({
            "left_template": left_name,
            "right_template": right_name,
            "left_slots": len(left_pattern),
            "right_slots": len(right_pattern),
            "closures": len(rows),
            **info,
        })
        records.extend(rows)
    unique = {row["audit"]["sha256"]: row for row in records}
    exact = [row for row in unique.values() if row["audit"]["exact"]]
    admitted = [row for row in exact if row["mechanically_admitted"]]
    longest = sorted(unique.values(), key=lambda row: (-row["audit"]["letters"], row["rendered"]))[:20]
    payload = {
        "experiment": "typed-grammar-equation-search-20260919",
        "method": "character-indexed exact intersection over independently authored typed clause templates",
        "role_bank_sizes": {key: len(value) for key, value in role_banks.items()},
        "template_pairs": len(TEMPLATES) ** 2,
        "stats": stats,
        "candidate_count": len(unique),
        "exact_count": len(exact),
        "mechanically_admitted_count": len(admitted),
        "exact_admitted": admitted,
        "longest_rendered_candidates": longest,
        "next_repair": "add agreement-carrying inflection slots and a boundary transition that resegments a residual across two adjacent clause roles; keep exact and admission checks live",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "candidate_count": payload["candidate_count"],
        "exact_count": payload["exact_count"],
        "mechanically_admitted_count": payload["mechanically_admitted_count"],
        "longest_letters": max((row["audit"]["letters"] for row in unique.values()), default=0),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
