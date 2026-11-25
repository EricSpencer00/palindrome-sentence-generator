"""Centre-out paragraph growth: every closure is an exact letter palindrome,
with free punctuation, crossing debts and palindromes set into both sides."""
import os

import pytest

from llm_palindrome import spectrum
from llm_palindrome.validator import is_palindrome

pytestmark = pytest.mark.skipif(not os.path.exists(spectrum.BIGRAMS),
                                reason="needs data/count_2w.txt and the NLTK Brown and Gutenberg corpora")


@pytest.fixture(scope="module")
def parts():
    from llm_palindrome import paragraph as P
    from llm_palindrome.prose_lm import ProseLM
    lm, t = ProseLM.load(), spectrum.Tables.load()
    return P, lm, P.Lexicon(lm, t), spectrum.Spectrum(t)


def test_every_closure_is_exact(parts):
    P, lm, lex, sp = parts
    closed = P.grow(lm, lex, 40, beam=60, spectrum=sp)
    assert closed
    for s in closed:
        assert is_palindrome(P.render(s.seq))


def test_a_palindrome_centre_stays_exact(parts):
    P, lm, lex, sp = parts
    for s in P.grow(lm, lex, 40, beam=40, spectrum=sp, centre=("racecar",))[:20]:
        text = P.render(s.seq)
        assert "racecar" in text.lower() and is_palindrome(text)


def test_debts_are_paid_from_the_right_end():
    from llm_palindrome.paragraph import Lexicon
    lex = Lexicon.__new__(Lexicon)
    lex.words = {"draw", "a", "ward", "war", "d"}
    lex.freq = {w: 1 for w in lex.words}
    lex.prefix = {"dra": ["draw"]}
    lex.suffix = {"ard": ["ward"]}
    # Right owes "draw": "draw" settles it, "dra" is not a word.
    assert ("draw", "", False) in lex.append_matches.__wrapped__(lex, "draw")
    # Left owes "...ard" at its end: "ward" overruns by "w", flipped to the right.
    assert ("ward", "w", True) in lex.prepend_matches.__wrapped__(lex, "ard")


def test_render_places_free_punctuation():
    from llm_palindrome.paragraph import render
    assert render(("no", ",", "i", "did", ".", "so")) == "No, I did. So."
