"""Spelling adds nothing the mirror can see, and it must not."""
import pytest

from llm_palindrome.spelling import BARE, CONTRACTIONS, letters, spell, spell_word


class TestSpellWord:
    def test_the_pronoun_is_capitalised(self):
        assert spell_word("i") == "I"

    def test_a_contraction_gets_its_apostrophe(self):
        assert spell_word("dont") == "don't"
        assert spell_word("im") == "I'm"

    def test_a_word_that_is_also_a_contraction_is_left_alone(self):
        """"its" is a word; rewriting it as "it's" changes the sentence."""
        assert spell_word("its") == "its"
        assert spell_word("ill") == "ill"

    def test_contraction_can_be_switched_off(self):
        assert spell_word("dont", contract=False) == "dont"

    def test_an_ordinary_word_is_untouched(self):
        assert spell_word("janitor") == "janitor"


class TestSpell:
    def test_the_sentence_opens_with_a_capital_and_closes_with_a_stop(self):
        assert spell(["lived", "on", "decaf"]) == "Lived on decaf."

    def test_it_does_not_lowercase_what_it_already_capitalised(self):
        """`str.capitalize` would return "I'm no devil" as "I'm no devil"
        only by luck; on "im no devil" it returns "Im no devil"."""
        assert spell(["im", "no", "devil"]) == "I'm no devil."

    def test_a_mid_sentence_pronoun_is_capitalised_too(self):
        assert spell(["on", "taxes", "i", "moan"]) == "On taxes I moan."

    def test_nothing_from_nothing(self):
        assert spell([]) == ""

    def test_the_stop_is_optional(self):
        assert spell(["go", "hang"], period=False) == "Go hang"


class TestTheMirrorCannotSeeIt:
    """The property the whole module rests on."""

    @pytest.mark.parametrize("words", [
        ["im", "no", "devil"],
        ["i", "dont", "recall"],
        ["on", "taxes", "i", "moan"],
        ["youre", "a", "star"],
    ])
    def test_spelling_leaves_the_letters_alone(self, words):
        assert letters(spell(words)) == "".join(words)

    def test_every_contraction_preserves_its_letters(self):
        for plain, shown in CONTRACTIONS.items():
            assert letters(shown) == plain, (plain, shown)

    def test_bare_words_are_all_listed_as_contractions_too(self):
        """BARE only means anything for a word CONTRACTIONS would rewrite."""
        assert BARE <= set(CONTRACTIONS) | {"its", "lets", "were", "well",
                                            "hell", "shell", "id", "wed"}
