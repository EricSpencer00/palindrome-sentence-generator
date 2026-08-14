"""Direction bookkeeping for the backward scorer.

These are the details that fail silently: pick the wrong token for a word and
the backward model is scored on a token it only ever sees in second place, and
every number downstream is quiet nonsense.
"""
from __future__ import annotations

from llm_palindrome.directional import context_key, leading_token


def test_forward_reaches_a_word_by_its_first_token():
    diverged = [4614, 2004]  # [" diver", "ged"]
    assert leading_token(diverged, reversed_order=False) == 4614


def test_backward_reaches_a_word_by_its_last_token():
    diverged = [4614, 2004]
    assert leading_token(diverged, reversed_order=True) == 2004


def test_single_token_words_are_the_same_either_way():
    star = [3491]
    assert leading_token(star, False) == leading_token(star, True) == 3491


# `prepare` keys distributions off the states in the beam; `word_delta` looks
# them up from the state that results once a word is added. If those two keys
# disagree the cache misses every time and the whole amortization is gone, with
# nothing raised to say so.

def test_appending_looks_up_what_prepare_stored():
    parent_left = ("rats", "live", "on")
    prepared = context_key(parent_left, "append", max_context=8)

    child_left = parent_left + ("no",)
    looked_up = context_key(child_left[:-1], "append", max_context=8)
    assert looked_up == prepared


def test_prepending_looks_up_what_prepare_stored():
    parent_right = ("no", "evil", "star")
    prepared = context_key(parent_right, "prepend", max_context=8)

    child_right = ("on",) + parent_right
    looked_up = context_key(child_right[1:], "prepend", max_context=8)
    assert looked_up == prepared


def test_truncation_keeps_the_end_nearest_the_scored_word():
    words = ("a", "b", "c", "d", "e")
    assert context_key(words, "append", 2) == ("append", ("d", "e"))
    assert context_key(words, "prepend", 2) == ("prepend", ("a", "b"))


def test_truncation_still_agrees_when_the_context_is_longer_than_the_window():
    parent = ("a", "b", "c", "d", "e", "f")
    child = parent + ("g",)
    assert (context_key(child[:-1], "append", 3)
            == context_key(parent, "append", 3))
