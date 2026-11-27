from llm_palindrome.contractions import JOIN, SPELL
from llm_palindrome.paragraph import render
from llm_palindrome.prose_lm import _join_contractions


def test_brown_and_gutenberg_spellings_join_to_letters():
    assert _join_contractions(["I", "don't", "know"]) == ["I", "dont", "know"]
    assert _join_contractions(["I", "don", "'", "t", "know"]) == ["I", "dont", "know"]
    assert _join_contractions(["'", "Tis", "late"]) == ["tis", "late"]
    assert _join_contractions(["o", "'", "er", "the", "sea"]) == ["oer", "the", "sea"]


def test_possessives_and_homograph_contractions_are_left_alone():
    assert _join_contractions(["Emma", "'", "s", "sister"]) == ["Emma", "'", "s", "sister"]
    assert "were" not in SPELL and "ill" not in SPELL and "its" not in SPELL


def test_render_prints_the_apostrophe_and_capitalises_after_it():
    assert render(["tis", "late", ".", "im", "here", "."]) == "'Tis late. I'm here."
    assert render(["we", "dont", "go", "oer", "it"]) == "We don't go o'er it."


def test_spellings_round_trip():
    assert all(JOIN[v.lower()] == k for k, v in SPELL.items())
    assert all(k == "".join(c for c in v.lower() if c.isalpha()) for k, v in SPELL.items())
