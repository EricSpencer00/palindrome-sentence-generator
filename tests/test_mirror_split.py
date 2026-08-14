"""Tests for the cut the page draws its caret at.

The mirror belongs to the letters, not to the words. Everything here is about
the gap between those two ways of counting: a palindrome's halves hold the same
number of LETTERS by construction and no particular number of words, so any
split that counts words drifts off the mirror as soon as the halves are worded
differently. The straddling word need not be a palindrome itself — the mirror
of "a man a plan a canal panama" runs through the `c` of `canal`.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["PALINDROME_NO_WARM"] = "1"   # these are pure functions; no model needed

from server.app import _display_pivot, _shape, _split_at_mirror
from llm_palindrome.validator import is_palindrome, normalize


class TestSplitAtMirror:
    def test_mirror_in_a_gap_between_words(self):
        # 28 letters, mirror at 14, which is exactly where `web` ends.
        words = "wrote to lay a web be way a lot et or w".split()
        left, center, right, pivot = _split_at_mirror(words)
        assert left == ["wrote", "to", "lay", "a", "web"]
        assert center == ""
        assert right == ["be", "way", "a", "lot", "et", "or", "w"]
        assert pivot == 0

    def test_unequal_word_counts_do_not_move_the_mirror(self):
        """The regression itself: 5 words left, 7 right, so counting words put
        the cut two words late even though the letters split evenly."""
        words = "wrote to lay a web be way a lot et or w".split()
        left, _, right, _ = _split_at_mirror(words)
        assert (len(left), len(right)) == (5, 7)
        assert sum(len(w) for w in left) == sum(len(w) for w in right)

    def test_mirror_inside_a_word_odd_letters(self):
        # 5 letters: the middle letter IS the pivot.
        left, center, right, pivot = _split_at_mirror(["level"])
        assert (left, center, right) == ([], "level", [])
        assert center[pivot] == "v"

    def test_mirror_inside_a_word_even_letters(self):
        # 6 letters, mirror at 3, two letters into `noon`.
        assert _split_at_mirror(["a", "noon", "a"]) == (["a"], "noon", ["a"], 2)

    def test_straddling_word_need_not_be_a_palindrome(self):
        words = "a man a plan a canal panama".split()
        left, center, right, pivot = _split_at_mirror(words)
        assert center == "canal"
        assert center[pivot] == "c"     # 21 letters; index 10 of the whole text

    def test_single_letter_text(self):
        assert _split_at_mirror(["w"]) == ([], "w", [], 0)

    def test_empty(self):
        assert _split_at_mirror([]) == ([], "", [], 0)

    @pytest.mark.parametrize("text", [
        "never odd or even",
        "a man a plan a canal panama",
        "was it a car or a cat i saw",
        "step on no pets",
        "rats live on no evil star",
        "no lemon no melon",
        "wrote to lay a web be way a lot et or w",
    ])
    def test_cut_lands_on_the_mirror(self, text):
        """One property covers every case: the words come back whole, and the
        mirror lies inside the centre word, or in the gap when there is none."""
        words = text.split()
        assert is_palindrome(text)
        left, center, right, pivot = _split_at_mirror(words)

        assert left + ([center] if center else []) + right == words
        n = len(normalize(text))
        half, before = n // 2, sum(len(w) for w in left)
        if center:
            assert before <= half < before + len(center)
            assert pivot == half - before
        else:
            assert n % 2 == 0 and before == half and pivot == 0


class TestDisplayPivot:
    def test_no_spaces(self):
        assert _display_pivot("level", 2) == 2

    def test_spaces_shift_the_index(self):
        # 14 letters; the mirror is between the two d's of "odd".
        assert _display_pivot("never odd or even", 7) == 8

    def test_past_the_end(self):
        assert _display_pivot("abc", 9) == 3


class TestShape:
    def test_prompt_is_echoed_at_the_mirror_as_typed(self):
        shape = _shape(["a", "neveroddoreven", "a"], "Never Odd Or Even")
        assert shape["promptCenter"] is True
        assert shape["centerDisplay"] == "Never Odd Or Even"
        assert shape["pivotOdd"] is False     # 16 letters: the pivot is a gap
        d, p = shape["centerDisplay"], shape["pivot"]
        assert d[p - 1:p + 1] == "dd"         # the caret goes between the d's

    def test_unrelated_prompt_is_not_echoed(self):
        shape = _shape("step on no pets".split(), "dogs")
        assert shape["promptCenter"] is False
        assert shape["centerDisplay"] == shape["center"]

    def test_odd_letter_count_marks_a_single_letter(self):
        shape = _shape("a man a plan a canal panama".split(), "")
        assert shape["pivotOdd"] is True
        assert shape["centerDisplay"][shape["pivot"]] == "c"

    def test_counts(self):
        shape = _shape("step on no pets".split(), "")
        assert (shape["letters"], shape["words"]) == (12, 4)
        assert shape["center"] == ""          # 12 letters, mirror after `on`
