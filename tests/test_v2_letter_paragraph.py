"""The letter-level paragraph endpoint.

`/api/v2/paragraph` has been serving the WORD-ORDER mode: the sentence sequence
mirrors and the letters do not. That is a different and much easier constraint —
it pays no per-letter cost — and it was adopted at a point where the letter-level
paragraph was a list of fragments ("No it cab action. Names abandon.").

It is not that any more. Stored canon spellings, blind-judged centres,
`themes.best_cluster` and `themes.order_for_refrain` produce a paragraph whose
LETTERS mirror and which a blind judge reads as sentences on a subject. That is
what the endpoint should serve.

The word mode stays reachable, labelled, because it is a real curiosity — but it
stops being the answer to "give me a palindromic paragraph".
"""
import json
from pathlib import Path

import pytest

from llm_palindrome.validator import is_palindrome

CENTRES = json.loads(Path("data/centres.json").read_text())


class TestShippedCentres:
    def test_every_centre_is_itself_a_palindrome(self):
        """The refrain is only a palindrome because each unit is one."""
        for centre in CENTRES:
            assert is_palindrome(centre), centre

    def test_there_are_enough_to_build_a_theme_from(self):
        assert len(CENTRES) >= 20

    def test_none_is_a_fragment(self):
        for centre in CENTRES:
            assert len(centre.split()) >= 4, centre

    def test_all_lowercase_and_alphabetic(self):
        for centre in CENTRES:
            assert centre == centre.lower(), centre
            assert all(w.isalpha() for w in centre.split()), centre


class TestLetterParagraph:
    def test_the_text_is_a_letter_palindrome(self):
        from server.v2 import letter_paragraph

        out = letter_paragraph(sentences=7)
        assert is_palindrome(out["text"]), out["text"]

    def test_it_says_which_constraint_it_satisfied(self):
        """A visitor told "palindrome" deserves to know which one."""
        from server.v2 import letter_paragraph

        out = letter_paragraph(sentences=7)
        assert out["mode"] == "letter"
        assert out["letterPalindrome"] is True

    def test_the_sentences_share_a_subject(self):
        """The whole point of themes.best_cluster reaching the endpoint."""
        from llm_palindrome.themes import cohesion
        from server.v2 import letter_paragraph

        out = letter_paragraph(sentences=7)
        assert cohesion(out["units"]) > 0.5, out["units"]

    def test_it_mirrors_the_unit_sequence(self):
        from server.v2 import letter_paragraph

        out = letter_paragraph(sentences=5)
        said = [s.strip().lower() for s in out["text"].split(".") if s.strip()]
        assert said == said[::-1]

    def test_a_question_opens_and_a_statement_turns(self):
        """order_for_refrain's rule, checked where it ships."""
        from llm_palindrome.themes import is_question
        from server.v2 import letter_paragraph

        out = letter_paragraph(sentences=7)
        assert is_question(out["units"][0]), out["units"]
        assert not is_question(out["units"][-1]), out["units"]

    def test_a_prompt_steers_the_theme(self):
        from server.v2 import letter_paragraph

        out = letter_paragraph(sentences=5, prompt="devil santa")
        joined = " ".join(out["units"])
        assert "devil" in joined or "santa" in joined, out["units"]

    def test_asking_for_one_sentence_still_works(self):
        from server.v2 import letter_paragraph

        out = letter_paragraph(sentences=1)
        assert is_palindrome(out["text"])

    def test_the_word_count_is_reported_and_right(self):
        import re

        from server.v2 import letter_paragraph

        out = letter_paragraph(sentences=6)
        assert out["words"] == len(re.findall(r"[A-Za-z]+", out["text"]))


class TestTheEndpoint:
    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient

        from server.app import app
        return TestClient(app)

    def test_paragraph_now_serves_the_letter_mode(self, client):
        body = client.get("/api/v2/paragraph").json()
        assert body["mode"] == "letter"
        assert is_palindrome(body["text"])

    def test_the_word_mode_is_still_reachable(self, client):
        """It is a real curiosity; it is just not the default answer."""
        body = client.get("/api/v2/paragraph", params={"mode": "word"}).json()
        assert body["mode"] == "word"
        assert body["letterPalindrome"] is False

    def test_an_unknown_mode_is_rejected(self, client):
        assert client.get("/api/v2/paragraph",
                          params={"mode": "sideways"}).status_code == 422
