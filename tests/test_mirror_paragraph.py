"""The paragraph must mirror into DIFFERENT text, not into itself.

The endpoint was serving a refrain: a mirrored sequence of sentences that are
each individually palindromic. It passes `is_palindrome`, and the mirror does
no work — reverse it and every sentence comes back as itself. Nothing is
discovered by reading it backwards, which is the whole point of a palindrome
at paragraph length.

The real construction nests mirror-PAIRS, where the right half is different
text that happens to spell the left half reversed:

    Lived on decaf.  ->  Faced no devil.
    Go hang a salami. -> Ima lasagna hog.
    Live on time.    ->  Emit no evil.

This reads worse than the refrain, and a blind judge preferred the refrain.
That preference is not a reason to serve the refrain: it scored better by not
attempting the constraint. These tests hold the endpoint to the constraint.
"""
import json
from pathlib import Path

import pytest

from llm_palindrome.validator import is_palindrome

UNITS = json.loads(Path("data/mirror_units.json").read_text())


class TestShippedMirrorUnits:
    def test_each_pair_closes_into_a_palindrome(self):
        for unit in UNITS:
            joined = " ".join(unit["left"] + unit["right"])
            assert is_palindrome(joined), joined

    def test_no_half_is_its_own_mirror(self):
        """A pair whose halves are identical is a refrain unit in disguise."""
        for unit in UNITS:
            assert unit["left"] != unit["right"], unit

    def test_the_right_half_is_the_left_half_reversed(self):
        for unit in UNITS:
            left = "".join(unit["left"])
            right = "".join(unit["right"])
            assert right == left[::-1], unit

    def test_there_are_enough_to_build_a_paragraph(self):
        assert len(UNITS) >= 15


class TestMirrorParagraph:
    def test_the_whole_paragraph_is_a_palindrome(self):
        from server.v2 import letter_paragraph

        out = letter_paragraph(sentences=9)
        letters = "".join(c.lower() for c in out["text"] if c.isalpha())
        assert letters == letters[::-1]

    def test_no_sentence_is_repeated(self):
        """The refrain's tell: every sentence appearing exactly twice."""
        from server.v2 import letter_paragraph

        out = letter_paragraph(sentences=9)
        said = [s.strip().lower() for s in out["text"].split(".") if s.strip()]
        assert len(set(said)) == len(said), said

    def test_the_second_half_is_different_text(self):
        """The property the refrain did not have and this exists for."""
        from server.v2 import letter_paragraph

        out = letter_paragraph(sentences=9)
        said = [s.strip().lower() for s in out["text"].split(".") if s.strip()]
        first, second = said[:len(said) // 2], said[len(said) // 2 + 1:]
        assert not (set(first) & set(second)), set(first) & set(second)

    def test_most_sentences_are_not_palindromes_on_their_own(self):
        """"Lived on decaf" is not a palindrome; only the whole is.

        If every sentence were self-palindromic we would be back to the
        refrain, whatever else the payload said.
        """
        from server.v2 import letter_paragraph

        out = letter_paragraph(sentences=9)
        said = [s.strip() for s in out["text"].split(".") if s.strip()]
        selfish = [s for s in said if is_palindrome(s)]
        assert len(selfish) <= 2, selfish

    def test_it_reports_the_construction(self):
        from server.v2 import letter_paragraph

        out = letter_paragraph(sentences=9)
        assert out["mode"] == "letter"
        assert out["letterPalindrome"] is True
        assert out["pairs"] >= 4


class TestTheEndpoint:
    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient

        from server.app import app
        return TestClient(app)

    def test_the_default_mirrors_into_different_text(self, client):
        body = client.get("/api/v2/paragraph").json()
        letters = "".join(c.lower() for c in body["text"] if c.isalpha())
        assert letters == letters[::-1]
        said = [s.strip().lower() for s in body["text"].split(".") if s.strip()]
        assert len(set(said)) == len(said)

    def test_the_refrain_is_still_reachable_and_labelled(self, client):
        """It is a real form — mirrored canonical sentences — but it is not
        what "palindromic paragraph" should return by default."""
        body = client.get("/api/v2/paragraph",
                          params={"mode": "refrain"}).json()
        assert body["mode"] == "refrain"
        letters = "".join(c.lower() for c in body["text"] if c.isalpha())
        assert letters == letters[::-1]
