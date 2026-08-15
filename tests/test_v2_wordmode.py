"""Tests for the word-order paragraph mode on the v2 service.

Letter-level generation is capped by the mirror cost at 3.3 bits/letter — its
paragraphs are refrain poetry built from canonical material. Word-order
palindromes pay no per-letter cost, hold a subject across 207 words, and are
judged INTENTIONAL against controls. Both are palindromes; they are different
constraints, and the endpoint has to say which one it served.
"""
import os

os.environ.setdefault("PALINDROME_NO_WARM", "1")

from server.v2 import word_paragraph


BANK = {
    "outer": ["gulls circled harbour", "sailors coiled ropes", "waves took sons"],
    "center": "spring follows spring",
}


class TestWordParagraph:
    def test_result_is_word_palindromic(self):
        from llm_palindrome.paragraphs import is_word_palindrome
        assert is_word_palindrome(word_paragraph(BANK, sentences=3)["text"])

    def test_reports_the_mode_it_served(self):
        assert word_paragraph(BANK, sentences=3)["mode"] == "word"

    def test_sentence_count_controls_length(self):
        short = word_paragraph(BANK, sentences=1)
        long = word_paragraph(BANK, sentences=3)
        assert long["words"] > short["words"]

    def test_never_exceeds_the_bank(self):
        out = word_paragraph(BANK, sentences=99)
        assert out["pairs"] == len(BANK["outer"])

    def test_reports_that_it_is_not_letter_palindromic(self):
        """The honest disclosure: this is a different constraint."""
        assert word_paragraph(BANK, sentences=3)["letterPalindrome"] is False

    def test_carries_the_turn(self):
        out = word_paragraph(BANK, sentences=3)
        assert "Waves took sons." in out["text"]
        assert "Sons took waves." in out["text"]


class TestVariety:
    """Every request returned the identical paragraph.

    The same flaw the letter-level mode had with a pinned seed: a bank of 34
    units always read out in bank order, so the endpoint was a static page
    that took a round trip. The bank supports far more paragraphs than it has
    units — which subset, and in what order, is the variation.
    """

    def test_different_nonces_give_different_paragraphs(self):
        from server.v2 import word_paragraph
        bank = {"outer": [f"{a} taught {b}" for a, b in
                          zip("alpha bravo delta echo gamma hotel india juliet "
                              "kilo lima mike november".split(),
                              "oscar papa quebec romeo sierra tango uniform "
                              "victor whiskey xray yankee zulu".split())],
                "center": "spring follows spring"}
        seen = {word_paragraph(bank, sentences=5, nonce=n)["text"] for n in range(5)}
        assert len(seen) == 5

    def test_every_variant_is_still_word_palindromic(self):
        from llm_palindrome.paragraphs import is_word_palindrome
        from server.v2 import word_paragraph
        bank = {"outer": [f"{a} taught {b}" for a, b in
                          zip("alpha bravo delta echo gamma hotel india juliet "
                              "kilo lima mike november".split(),
                              "oscar papa quebec romeo sierra tango uniform "
                              "victor whiskey xray yankee zulu".split())],
                "center": "spring follows spring"}
        for n in range(6):
            assert is_word_palindrome(word_paragraph(bank, sentences=5, nonce=n)["text"])

    def test_same_nonce_is_reproducible(self):
        from server.v2 import word_paragraph
        bank = {"outer": [f"{a} taught {b}" for a, b in
                          zip("alpha bravo delta echo gamma hotel india juliet "
                              "kilo lima mike november".split(),
                              "oscar papa quebec romeo sierra tango uniform "
                              "victor whiskey xray yankee zulu".split())],
                "center": "spring follows spring"}
        a = word_paragraph(bank, sentences=5, nonce=42)["text"]
        b = word_paragraph(bank, sentences=5, nonce=42)["text"]
        assert a == b


class TestBankSelection:
    """The word mode ignored the visitor's prompt.

    The letter mode centres a palindrome on the words you type; the word mode
    served the same bank whatever you asked for. Banks are themed, so the
    prompt can choose between them.
    """

    BANKS = {"harbour": {"outer": ["gulls circled harbour"], "center": "spring follows spring"},
             "river": {"outer": ["rivers fed villages"], "center": "water remembers water"}}

    def test_prompt_word_selects_its_bank(self):
        from server.v2 import pick_bank
        assert pick_bank(self.BANKS, "tell me about the river")[0] == "river"

    def test_bank_content_words_also_match(self):
        from server.v2 import pick_bank
        assert pick_bank(self.BANKS, "gulls")[0] == "harbour"

    def test_unmatched_prompt_still_returns_a_bank(self):
        from server.v2 import pick_bank
        name, bank = pick_bank(self.BANKS, "quantum chromodynamics")
        assert name in self.BANKS and bank["outer"]

    def test_empty_prompt_returns_a_bank(self):
        from server.v2 import pick_bank
        assert pick_bank(self.BANKS, "")[0] in self.BANKS


class TestArcAwareSelection:
    """Keep the stakes-bearing units, and put them near the centre.

    A judge held a 147-word paragraph to hold a subject and rejected the 33-,
    63- and 99-word cuts of the same bank: only the long one contained "doubt,
    fatigue, theories, dawn" — the abstract stakes — while the others were
    "pure formal reversals without thematic progression". A uniform shuffle
    drops those units at random, so the endpoint sometimes serves a paragraph
    with no through-line at all.
    """

    BANK = {"outer": ["students loaded telescopes", "cables fed instruments",
                      "frost coated glass", "wind rocked domes",
                      "doubt shadowed certainty", "fatigue blurred readings",
                      "dawn erased darkness", "mirrors dimmed dust"],
            "center": "night follows night"}

    def test_arc_units_survive_a_short_request(self):
        from server.v2 import select_units
        picked = select_units(self.BANK["outer"], 4, nonce=0)
        assert any("doubt" in u or "fatigue" in u or "dawn" in u for u in picked)

    def test_arc_units_sit_near_the_centre(self):
        """The centre is the turn; the stakes belong beside it, not at the rim."""
        from server.v2 import select_units
        picked = select_units(self.BANK["outer"], 6, nonce=1)
        arc = [i for i, u in enumerate(picked)
               if any(k in u for k in ("doubt", "fatigue", "dawn"))]
        assert arc and min(arc) >= len(picked) // 3

    def test_returns_the_requested_count(self):
        from server.v2 import select_units
        assert len(select_units(self.BANK["outer"], 5, nonce=2)) == 5

    def test_never_exceeds_the_bank(self):
        from server.v2 import select_units
        assert len(select_units(self.BANK["outer"], 99, nonce=3)) == len(self.BANK["outer"])

    def test_still_varies_between_requests(self):
        from server.v2 import select_units
        seen = {tuple(select_units(self.BANK["outer"], 5, nonce=n)) for n in range(6)}
        assert len(seen) > 1


class TestDefaultLengthClearsThreshold:
    """The default must produce a paragraph that holds a subject.

    A length ladder judged three times (docs/training.md) put the through-line
    at ~105 words, or about 17 units: below it the paragraph stops before
    anything happens to anyone and reads as scenery in mirror form. The old
    default of 12 units sat under that line, so the default request returned a
    formally correct palindrome with no story in it.
    """

    def test_the_word_mode_is_floored_at_the_threshold(self):
        """The threshold is the WORD mode's, so it belongs on that path.

        `sentences` used to serve only this mode and its default carried the
        finding. The endpoint now defaults to the letter mode, where 7 whole
        sentences is a paragraph and 17 would exhaust the centre inventory —
        so a shared default cannot express both. The word path floors its own
        length instead, and a request for fewer still clears 105 words.
        """
        from fastapi.testclient import TestClient
        from server.app import app
        client = TestClient(app)
        body = client.get("/api/v2/paragraph",
                          params={"mode": "word", "sentences": 3}).json()
        assert body["mode"] == "word"
        assert body["words"] >= 105, body["words"]

    def test_default_request_clears_105_words(self):
        import json
        from pathlib import Path
        from server.v2 import word_paragraph
        banks = json.loads(Path("data/word_banks.json").read_text())
        for name, bank in banks.items():
            out = word_paragraph(bank, sentences=18, nonce=0)
            assert out["words"] >= 105, (name, out["words"])
