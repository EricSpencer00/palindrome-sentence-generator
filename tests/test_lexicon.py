"""Which strings are words, as distinct from which strings are frequent.

Mining takes the left half from attested English, so it reads. The right half
is whatever the mirrored letters segment into, and the frequency list will
happily supply "utc", "ips", "evo", "rpm", "iot" and "csi" — all common, none
words. A trial run produced "has pictures || ser utc ips ah".

A dictionary separates them, but a lemma dictionary alone is too strict: web2
has "pet" and not "pets", so filtering on it directly loses "step on no pets",
which is the canonical example of the thing being built. Acceptance therefore
covers regular inflections of a dictionary headword.

The list is built once by training/build_lexicon.py and shipped, so nothing at
runtime depends on a system file that exists only on some machines.
"""
from pathlib import Path

import pytest

from llm_palindrome.lexicon import inflections, is_real_word, load_lexicon

LEXICON_PATH = Path("data/lexicon.txt")


class TestInflections:
    def test_offers_the_bare_stem(self):
        assert "pet" in inflections("pets")

    def test_handles_es_plurals(self):
        assert "box" in inflections("boxes")

    def test_handles_ies_plurals(self):
        assert "party" in inflections("parties")

    def test_handles_past_and_progressive(self):
        assert "walk" in inflections("walked")
        assert "walk" in inflections("walking")

    def test_handles_a_dropped_silent_e(self):
        assert "erase" in inflections("erased")
        assert "live" in inflections("living")

    def test_a_short_word_yields_nothing_spurious(self):
        assert "" not in inflections("is")

    def test_it_does_not_pluralise_a_two_letter_word(self):
        """"ats" is not a word, but "at" is a headword and "s" is a rule.

        Closed-class words do not inflect, and every two-letter headword is
        closed-class or an abbreviation, so a stem that short is never a base.
        """
        assert "at" not in inflections("ats")
        assert not is_real_word("ats", frozenset({"at"}))


class TestIsRealWord:
    HEADWORDS = frozenset({"pet", "erase", "step", "lemon", "walk", "party"})

    def test_accepts_a_headword(self):
        assert is_real_word("lemon", self.HEADWORDS)

    def test_accepts_a_regular_inflection(self):
        assert is_real_word("pets", self.HEADWORDS)
        assert is_real_word("erased", self.HEADWORDS)
        assert is_real_word("parties", self.HEADWORDS)

    def test_rejects_an_abbreviation(self):
        for abbrev in ("utc", "ips", "evo", "rpm", "iot", "csi"):
            assert not is_real_word(abbrev, self.HEADWORDS), abbrev

    def test_rejects_the_empty_string(self):
        assert not is_real_word("", self.HEADWORDS)


class TestTheShippedLexicon:
    @pytest.fixture(scope="class")
    def lexicon(self):
        if not LEXICON_PATH.exists():
            pytest.fail(f"{LEXICON_PATH} missing — run training/build_lexicon.py")
        return load_lexicon(str(LEXICON_PATH))

    def test_it_is_large_enough_to_be_a_dictionary(self, lexicon):
        assert len(lexicon) > 5000

    def test_it_keeps_the_words_the_canon_needs(self, lexicon):
        for w in ("step", "pets", "lemon", "melon", "rats", "star", "evil",
                  "emit", "live", "erase"):
            assert is_real_word(w, lexicon), w

    def test_it_drops_the_abbreviations_that_polluted_mining(self, lexicon):
        for w in ("utc", "ips", "evo", "rpm", "iot", "csi", "erp", "ats"):
            assert not is_real_word(w, lexicon), w

    def test_every_entry_is_alphabetic_and_lowercase(self, lexicon):
        for w in list(lexicon)[:2000]:
            assert w.isalpha() and w.islower(), w
