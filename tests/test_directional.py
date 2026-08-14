"""Direction bookkeeping for the backward scorer.

These are the details that fail silently: pick the wrong token for a word and
the backward model is scored on a token it only ever sees in second place, and
every number downstream is quiet nonsense.
"""
from __future__ import annotations

from llm_palindrome.directional import leading_token


def test_forward_reaches_a_word_by_its_first_token():
    diverged = [4614, 2004]  # [" diver", "ged"]
    assert leading_token(diverged, reversed_order=False) == 4614


def test_backward_reaches_a_word_by_its_last_token():
    diverged = [4614, 2004]
    assert leading_token(diverged, reversed_order=True) == 2004


def test_single_token_words_are_the_same_either_way():
    star = [3491]
    assert leading_token(star, False) == leading_token(star, True) == 3491
