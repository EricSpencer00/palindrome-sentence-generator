"""The endpoint serving generated pairs rather than catalogued ones.

Criterion 9 of docs/NORTH-STAR.md is the one v2 fails outright: every readable
unit it places is a catalogued palindrome, so the mirror is ours and the
sentences are not. These tests cover the path that answers from a generated
bank — including the part that is easy to get wrong, which is what happens
when the generated bank is too thin to carry a paragraph on its own.
"""
import json

import pytest

from llm_palindrome.validator import is_palindrome

# Real mirror-pairs, mechanically checked below, standing in for a bank.
BANK = [
    {"left": ["draw", "no"], "right": ["on", "ward"]},
    {"left": ["step", "on"], "right": ["no", "pets"]},
    {"left": ["live", "not", "on"], "right": ["no", "ton", "evil"]},
    {"left": ["star", "was"], "right": ["saw", "rats"]},
    {"left": ["spot", "a"], "right": ["a", "tops"]},
    {"left": ["time", "not"], "right": ["ton", "emit"]},
    {"left": ["dog", "a"], "right": ["a", "god"]},
    {"left": ["moor", "a"], "right": ["a", "room"]},
    {"left": ["drawer", "no"], "right": ["on", "reward"]},
    {"left": ["stop", "no"], "right": ["on", "pots"]},
]


@pytest.fixture
def bank(monkeypatch):
    """A bank fat enough to reach the word floor without the catalogue."""
    import server.v2 as v2
    pairs = [p for p in BANK
             if "".join(p["right"]) == "".join(p["left"])[::-1]]
    monkeypatch.setattr(v2, "_novel", pairs * 6)
    return pairs


class TestTheBankIsWellFormed:
    def test_every_pair_in_the_fixture_really_mirrors(self):
        """Otherwise the tests below would pass on a broken construction."""
        for pair in BANK:
            joined = " ".join(pair["left"] + pair["right"])
            assert is_palindrome(joined), joined


class TestServingGeneratedPairs:
    def test_it_reaches_the_word_floor(self, bank):
        import re

        from server.v2 import letter_paragraph
        out = letter_paragraph(sentences=4, min_words=100)
        assert len(re.findall(r"[A-Za-z]+", out["text"])) >= 100

    def test_it_says_the_material_is_not_borrowed(self, bank):
        from server.v2 import letter_paragraph
        out = letter_paragraph(sentences=4)
        assert out["borrowed"] is False
        assert out["source"] == "generated"

    def test_it_drops_the_catalogued_centre(self, bank):
        """A centre out of the canon is one more borrowed sentence, and the
        construction does not need one."""
        from server.v2 import letter_paragraph
        assert letter_paragraph(sentences=4)["centre"] is None

    def test_the_whole_text_still_mirrors(self, bank):
        from server.v2 import letter_paragraph
        letters = "".join(c.lower() for c in letter_paragraph()["text"]
                          if c.isalpha())
        assert letters == letters[::-1]

    def test_no_sentence_is_a_palindrome_on_its_own(self, bank):
        """With no centre, criterion 3's allowance is not even spent."""
        from server.v2 import letter_paragraph
        said = [s.strip() for s in letter_paragraph()["text"].split(".")
                if s.strip()]
        assert not [s for s in said if is_palindrome(s)]

    def test_asking_for_the_catalogue_still_gets_it(self, bank):
        from server.v2 import letter_paragraph
        out = letter_paragraph(sentences=9, source="catalogue")
        assert out["borrowed"] is True


class TestAThinBankDoesNotGetUsed:
    """Half a paragraph of our own finished off with canon is still canon."""

    @pytest.fixture
    def thin(self, monkeypatch):
        import server.v2 as v2
        monkeypatch.setattr(v2, "_novel", BANK[:2])

    def test_it_falls_back_to_the_catalogue(self, thin):
        from server.v2 import letter_paragraph
        assert letter_paragraph(sentences=9)["borrowed"] is True

    def test_and_says_so(self, thin):
        from server.v2 import letter_paragraph
        assert letter_paragraph(sentences=9)["source"] == "catalogue"

    def test_it_can_still_be_asked_for_by_name(self, thin):
        from server.v2 import letter_paragraph
        out = letter_paragraph(sentences=2, source="novel")
        assert out["borrowed"] is False


class TestTheShippedBank:
    """data/novel_pairs.json is what the endpoint answers with."""

    @pytest.fixture(scope="class")
    def shipped(self):
        from pathlib import Path
        return json.loads(Path("data/novel_pairs.json").read_text())

    def test_it_can_carry_a_hundred_words_alone(self, shipped):
        """The floor the endpoint checks before preferring it — criterion 1,
        and the reason a thin bank falls back rather than mixing."""
        from llm_palindrome.paragraphs import paragraph_words
        pairs = [(p["left"], p["right"]) for p in shipped]
        assert paragraph_words(pairs) >= 100

    def test_no_half_is_a_palindrome_on_its_own(self, shipped):
        for pair in shipped:
            for half in (pair["left"], pair["right"]):
                assert not is_palindrome(" ".join(half)), half

    def test_no_word_appears_on_both_sides_of_a_pair(self, shipped):
        for pair in shipped:
            assert not set(pair["left"]) & set(pair["right"]), pair

    def test_every_word_survives_the_public_vocabulary_filter(self, shipped):
        """The bank reaches the same endpoint the search does, and the search
        is not allowed to place these words either."""
        from llm_palindrome.generate import build_vocab
        from llm_palindrome.spelling import CONTRACTIONS
        safe = set(build_vocab(60000)) | set(CONTRACTIONS)
        for pair in shipped:
            for word in pair["left"] + pair["right"]:
                assert word in safe, word

    def test_every_shipped_pair_mirrors(self):
        from pathlib import Path
        path = Path("data/novel_pairs.json")
        if not path.exists():
            pytest.skip("no generated bank yet")
        for pair in json.loads(path.read_text()):
            left, right = "".join(pair["left"]), "".join(pair["right"])
            assert right == left[::-1], pair

    def test_no_shipped_pair_is_catalogued(self):
        from pathlib import Path

        from llm_palindrome.paragraphs import is_novel_palindrome
        path = Path("data/novel_pairs.json")
        if not path.exists():
            pytest.skip("no generated bank yet")
        for pair in json.loads(path.read_text()):
            joined = " ".join(pair["left"] + pair["right"])
            assert is_novel_palindrome(joined), joined
