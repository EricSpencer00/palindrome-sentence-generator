"""Mechanical screen for jointly proposed, two-sided English mirror pairs.

Unlike the rejected pair-bank selector and independent phrase-list experiment,
one proposal here designs both English lexicalizations around the same implicit
letter tape.  The language model may propose; this module decides every
admission property mechanically: exact reversal, lexical form, length,
nondegeneracy, and novelty against both the catalogue and existing v3 pairs.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.paragraphs import is_novel_palindrome
from llm_palindrome.admission import mechanical_admission_checks
from llm_palindrome.validator import is_palindrome, normalize
from server.v3 import harvest_pair, real_words


INTENTS = (
    "a person notices a warning near a door",
    "a traveler sees a signal beside a quiet road",
    "a friend gives a brief instruction before a task",
    "a person observes a small change in a familiar room",
    "two people react to a sound in a calm place",
    "a speaker makes a concise warning about an object",
    "a person recognizes a pattern during an ordinary errand",
    "a watcher describes a small event at night",
)


def pair_identity(left: str, right: str) -> frozenset[str]:
    return frozenset((normalize(left), normalize(right)))


def existing_v3_pairs(bank_path: Path = ROOT / "data" / "v3_bank.json") -> set[frozenset[str]]:
    """Pair-level novelty is stricter than a complete-palindrome catalogue check."""
    out = set()
    for row in json.loads(bank_path.read_text()):
        got = harvest_pair(row["text"].split())
        if got:
            out.add(pair_identity(" ".join(got[0]), " ".join(got[1])))
    return out


def lexical_form(phrase: str) -> bool:
    return bool(re.fullmatch(r"[a-z]+(?: [a-z]+)*", phrase))


def mechanical_checks(left: str, right: str, *, existing_pairs: set[frozenset[str]],
                      word_checker: Callable[[list[str]], bool] = real_words,
                      novel_checker: Callable[[str], bool] = is_novel_palindrome) -> dict[str, bool]:
    """Return every hard predicate; no scalar score can offset a failure."""
    left_norm, right_norm = normalize(left), normalize(right)
    text = f"{left} {right}".strip()
    shape = lexical_form(left) and lexical_form(right)
    words = left.split() + right.split() if shape else []
    shared = mechanical_admission_checks(
        text,
        local_catalogue=set(json.loads((ROOT / "data" / "known_palindromes.json").read_text())),
        min_letters=30,
        max_letters=60,
    )
    return shared | {
        "ascii_word_form": shape,
        "reverse_match": bool(left_norm and left_norm == right_norm[::-1]),
        "half_length_band": 15 <= len(left_norm) <= 30 and 15 <= len(right_norm) <= 30,
        "word_band": 3 <= len(left.split()) <= 8 and 3 <= len(right.split()) <= 8,
        "lexicon_words": bool(shape and word_checker(words)),
        "nondegenerate": bool(left_norm and left_norm != right_norm),
        "exact_palindrome": shared["exact_letter_palindrome"],
        "novel_catalogue": shared["local_catalogue_absent"] and novel_checker(text),
        "novel_v3_bank_pair": pair_identity(left, right) not in existing_pairs,
    }


def screen_candidate(left: str, right: str, link: str, *,
                     existing_pairs: set[frozenset[str]]) -> dict:
    checks = mechanical_checks(left, right, existing_pairs=existing_pairs)
    return {
        "left": left,
        "right": right,
        "link": link,
        "left_normalized": normalize(left),
        "right_normalized": normalize(right),
        "text": f"{left} {right}".strip(),
        "checks": checks,
        "rejection_codes": [key for key, value in checks.items() if not value],
    }


def accepted(row: dict) -> bool:
    return not row["rejection_codes"]
