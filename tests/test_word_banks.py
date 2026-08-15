"""Invariants every shipped word bank must satisfy.

Banks are data, not code, and they will grow — a new one is a JSON edit, not a
review. These are the properties the assembler and the endpoint silently rely
on, so a malformed bank should fail here rather than at a visitor.
"""
import json
import re
from pathlib import Path

import pytest

from llm_palindrome.paragraphs import is_word_palindrome, word_assemble

BANKS = json.loads(Path("data/word_banks.json").read_text())
NAMES = sorted(BANKS)


def words(s):
    return re.findall(r"[A-Za-z]+", s)


@pytest.mark.parametrize("name", NAMES)
class TestBankInvariants:
    def test_centre_is_self_palindromic(self, name):
        """The centre sits at the mirror point and must reverse to itself."""
        assert is_word_palindrome(BANKS[name]["center"])

    def test_units_are_three_word_svo(self, name):
        """Multi-clause units were judged 'scrambled syntax'; 3-word SVO reads."""
        bad = [u for u in BANKS[name]["outer"] if len(words(u)) != 3]
        assert not bad, bad

    def test_no_duplicate_units(self, name):
        outer = [u.lower() for u in BANKS[name]["outer"]]
        assert len(set(outer)) == len(outer)

    def test_bank_is_large_enough_to_vary(self, name):
        """A bank smaller than the default request is a static page."""
        assert len(BANKS[name]["outer"]) >= 12

    def test_assembles_to_a_word_palindrome(self, name):
        out = word_assemble(BANKS[name]["outer"], BANKS[name]["center"])
        assert is_word_palindrome(out)

    def test_every_unit_reverses_to_different_words(self, name):
        """A unit whose reversal equals itself adds no turn to the paragraph."""
        same = [u for u in BANKS[name]["outer"]
                if words(u) == list(reversed(words(u)))]
        assert not same, same


class TestAcrossBanks:
    def test_no_unit_appears_in_two_banks(self):
        seen = {}
        for name in NAMES:
            for u in BANKS[name]["outer"]:
                seen.setdefault(u.lower(), []).append(name)
        shared = {u: b for u, b in seen.items() if len(set(b)) > 1}
        assert not shared, shared

    def test_bank_names_are_matchable_words(self):
        """`pick_bank` scores the prompt against the name, so it must be a word."""
        assert all(re.fullmatch(r"[a-z]+", n) for n in NAMES)

    def test_at_least_two_banks_so_selection_means_something(self):
        assert len(NAMES) >= 2
