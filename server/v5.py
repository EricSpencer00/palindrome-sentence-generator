"""v5 engine behind /api/generate: centre-out paragraphs with crossing debts.

Selected with PALINDROME_ENGINE=v5; the route, events and payload shape are
v1's, so the existing page works unchanged.
"""
from __future__ import annotations

import os
import random
import re
import threading

from llm_palindrome import paragraph as P
from llm_palindrome.prose_lm import ProseLM, theme_bonus, theme_vectors
from llm_palindrome.spectrum import Spectrum, Tables

LETTERS = int(os.environ.get("PALINDROME_V5_LETTERS", "300"))
BEAM = int(os.environ.get("PALINDROME_V5_BEAM", "200"))
THEME_W = float(os.environ.get("PALINDROME_V5_THEME_W", "15"))

# An empty prompt draws one of these. Without a theme every run falls into the
# same phrases ("I saw one" in 24 of 24); with one, pairwise trigram overlap
# drops from 0.21 to 0.05 at the same readable share. Each kept theme read at
# 60% or better over two runs (runs/themes/trial-20261125.jsonl).
THEMES = [
    "forest wolf tree", "doctor fever nurse", "love heart kiss", "mother home child",
    "fire smoke ash", "gold silver coin", "farm field harvest", "sea ship storm",
    "winter snow cold", "city street crowd", "dog cat bird", "desert sand sun",
    "train station travel", "money bank debt", "letter ink pen", "garden flower rain",
    "island shore wave", "mountain stone climb", "kitchen bread salt", "river boat water",
    "wine glass dinner",
]

_m: dict = {}
_ready = threading.Event()


def warm() -> None:
    lm, tb = ProseLM.load(), Tables.load()
    _m.update(lm=lm, lex=P.Lexicon(lm, tb), sp=Spectrum(tb), vec=theme_vectors())
    _ready.set()


def search(prompt: str, centre: list[str]) -> list[str]:
    """Printed words of the best closed paragraph; [] if none closed.
    A palindromic prompt sits at the centre; its words set the theme."""
    _ready.wait()
    m = _m
    theme = [w for w in re.findall(r"[a-z]+", prompt.lower()) if w in m["vec"]][:6]
    theme = theme or random.choice(THEMES).split()
    closed = P.grow(m["lm"], m["lex"], LETTERS, beam=BEAM, spectrum=m["sp"], sent_w=2.0,
                    content_bonus=1.6, theme=theme_bonus(m["vec"], theme) if theme else None,
                    theme_w=THEME_W, start_noise=4.0, slack=40,
                    seed=int.from_bytes(os.urandom(2), "big"),
                    centre=tuple(centre))
    return P.render(closed[0].seq).split() if closed else []
