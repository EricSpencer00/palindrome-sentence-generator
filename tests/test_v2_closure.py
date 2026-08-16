"""The search has to close, and when it does not the endpoint still answers.

`/api/v2/generate` shipped with `candidate_limit=800` and returned "no
palindrome closed; try again" to every request at every budget. The cause is
not obvious from the symptom, which is why it survived: a bigger candidate
limit means the pool is 800 children of whichever state scored best, the beam
fills with sixty near-identical descendants of one parent, and they dead-end
together. Measured over 8 seeds at the shipped floor, closure ran 5/8 at a
limit of 50 and 0/8 at 300.

So one test pins the closure rate and one pins the fallback, and neither is
allowed to pass because the other is doing the work.
"""
import json
import os
from pathlib import Path

import pytest

from llm_palindrome.validator import is_palindrome


@pytest.fixture(scope="module")
def warm():
    os.environ["PALINDROME_NO_WARM"] = "1"
    import server.v2 as v2
    if v2._tries is None:
        v2._warm()
    if v2._tries is None:                      # pragma: no cover - setup guard
        pytest.skip(f"inventory did not load: {v2._load_error}")
    return v2


class TestTheSearchCloses:
    def test_the_candidate_limit_is_in_the_range_that_closes(self):
        """The measurement, as a bound rather than as a comment."""
        from server.v2 import CANDIDATE_LIMIT
        assert CANDIDATE_LIMIT <= 150

    @pytest.mark.slow
    def test_it_closes_on_most_seeds(self, warm):
        closed = sum(warm._search("", 8.0, None, nonce=n) is not None
                     for n in range(4))
        assert closed >= 3, f"{closed}/4 searches closed"

    @pytest.mark.slow
    def test_a_real_result_says_it_is_not_a_fallback(self, warm):
        """Absent and false are the same to a reader and not to a page."""
        found = warm._search("", 8.0, None, nonce=0)
        assert found is not None and found["fallback"] is False

    @pytest.mark.slow
    def test_what_closes_is_a_palindrome(self, warm):
        found = warm._search("", 8.0, None, nonce=0)
        assert found is not None
        assert is_palindrome(" ".join(found["left"] + found["right"]))


class TestTheFallback:
    def test_the_bank_is_shipped(self):
        bank = json.loads(Path("data/fallback_texts.json").read_text())
        assert len(bank) >= 4

    def test_every_banked_text_is_a_palindrome(self):
        """It reaches a visitor without a search having looked at it."""
        for entry in json.loads(Path("data/fallback_texts.json").read_text()):
            assert is_palindrome(" ".join(entry["words"])), entry["words"][:6]

    def test_it_answers_when_the_search_does_not(self, warm):
        out = warm.fallback_result("cats", nonce=1)
        assert out is not None
        assert is_palindrome(" ".join(out["left"] + out["right"]))

    def test_it_says_it_is_a_fallback(self, warm):
        """A canned answer presented as a fresh one is the dishonest version."""
        out = warm.fallback_result("cats", nonce=1)
        assert out["fallback"] is True
        assert out["usedPrompt"] is False
        assert out["promptWordsPlaced"] == []

    def test_it_is_not_always_the_same_one(self, warm):
        picks = {warm.fallback_result("", nonce=n)["letters"] for n in range(8)}
        assert len(picks) > 1

    def test_an_empty_bank_is_not_an_answer(self, warm, monkeypatch):
        """With nothing banked the endpoint must go back to erroring, not
        invent something."""
        monkeypatch.setattr(warm, "_fallbacks", [])
        assert warm.fallback_result("") is None
