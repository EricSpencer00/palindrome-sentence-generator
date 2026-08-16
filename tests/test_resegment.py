"""The closing half can be re-spelled: the mirror holds letters, not spaces."""
import pytest

from llm_palindrome.paragraphs import cut_at_letters, render, resegment
from llm_palindrome.validator import is_palindrome

PAIRS = [(["draw", "no"], ["on", "ward"]),
         (["step", "on"], ["no", "pets"])]

VOCAB = {"a", "i", "draw", "no", "on", "ward", "step", "pets", "note",
         "stop", "sit", "stops", "it", "sword", "words", "pet", "sword"}


class TestCutAtLetters:
    def test_it_cuts_on_word_boundaries(self):
        assert cut_at_letters(["ab", "cd", "ef"], [4, 2]) == [
            ["ab", "cd"], ["ef"]]

    def test_a_target_inside_a_word_goes_to_the_nearer_boundary(self):
        # Target 3 falls inside "cd" (letters 3-4); its far edge is 1 past,
        # its near edge 1 short, so the word is taken.
        assert cut_at_letters(["ab", "cd", "ef"], [3, 3]) == [
            ["ab", "cd"], ["ef"]]

    def test_the_last_group_takes_the_remainder(self):
        assert cut_at_letters(["a", "b", "c", "d"], [1, 1]) == [
            ["a"], ["b", "c", "d"]]

    def test_no_group_comes_back_empty(self):
        assert all(cut_at_letters(["abcdef"], [1, 1, 1]))

    def test_nothing_is_lost_or_added(self):
        words = ["one", "two", "three", "four", "five"]
        out = cut_at_letters(words, [6, 8, 5])
        assert [w for g in out for w in g] == words


class TestResegment:
    def test_it_returns_a_reading_of_the_same_letters(self):
        out = resegment(PAIRS, VOCAB)
        assert out is not None
        letters = "".join(w for group in out for w in group)
        assert letters == "".join("".join(r) for _, r in reversed(PAIRS))

    def test_it_returns_one_group_per_pair(self):
        assert len(resegment(PAIRS, VOCAB)) == len(PAIRS)

    def test_no_reading_gives_nothing(self):
        assert resegment(PAIRS, {"zzz"}) is None

    def test_nothing_from_nothing(self):
        assert resegment([], VOCAB) is None

    def test_the_paragraph_still_mirrors_after_respelling(self):
        """The property the whole idea rests on: spaces are not letters."""
        groups = resegment(PAIRS, VOCAB)
        text = " ".join(
            [" ".join(l) for l, _ in PAIRS] + [" ".join(g) for g in groups])
        assert is_palindrome(text)

    def test_a_scorer_decides_between_readings(self):
        """`respace_k`'s own order is a unigram model; the caller may know
        better, and passing a scorer is how it says so."""
        picked = resegment(PAIRS, VOCAB, k=8,
                           score=lambda words: -len(words))
        assert picked is not None


class TestAgainstRender:
    def test_render_is_unaffected_when_nothing_is_passed(self):
        assert render(PAIRS) == "Draw no. Step on. No pets. On ward."


class TestRenderWithATail:
    def test_a_tail_replaces_the_per_pair_spellings(self):
        text = render(PAIRS, tail=[["no", "pet"], ["son", "ward"]])
        assert text == "Draw no. Step on. No pet. Son ward."

    def test_a_tail_that_changes_the_letters_is_refused(self):
        """The assertion in render is what makes a free tail safe to allow."""
        with pytest.raises(AssertionError):
            render(PAIRS, tail=[["no", "pets"]])

    def test_the_resegmented_tail_renders(self):
        groups = resegment(PAIRS, VOCAB)
        assert render(PAIRS, tail=groups)
