"""Tests for the append-only stream the poster is built on.

The page places each word outward from the mirror, so a word's position depends
only on the words between it and the centre. That is what lets text stay where
it was written — but only for as long as every frame CONTAINS its predecessor.
The moment a frame drops a word the reader has already seen, the poster has to
rewrite itself, and the guarantee is gone. So it is worth a test.

`min_letters` is set out of reach in most of these so the deadline is what stops
the search: the usual early stop fires within milliseconds on a dictionary this
small, and a search that finishes before its first publish exercises the fallback
rather than the streaming.
"""
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["PALINDROME_NO_WARM"] = "1"

from server.app import _split_at_mirror
from llm_palindrome.centerout import centerout_search
from llm_palindrome.search import WordTries
from llm_palindrome.scoring import FreqScorer
from llm_palindrome.validator import is_palindrome, normalize

TINY_DICT = [
    "stop", "pots", "spot", "tops", "live", "evil", "rats", "star",
    "no", "on", "a", "madam", "was", "saw", "draw", "ward", "dog", "god",
    "never", "even", "now", "won", "net", "ten", "i", "am", "level", "deed",
]
CENTER = "level"


def collect(commit_every=0.05, floor=4000, budget=0.8, seed=1):
    frames: list[list[str]] = []
    words = centerout_search(
        WordTries(TINY_DICT), FreqScorer(TINY_DICT), min_letters=floor,
        beam_width=30, center=CENTER, seed=seed, max_steps=10**6,
        maximize="letters", deadline=time.monotonic() + budget,
        on_closed=frames.append, commit_every=commit_every,
    )
    return frames, words


def halves(words):
    """Split where the page splits — at the mirror.

    Not at `words.index(CENTER)`: "level" is in this dictionary, so the search
    can and does place another copy of it out in the left half, and taking the
    first occurrence cuts the frame in the wrong place.
    """
    left, center, right, _ = _split_at_mirror(words)
    assert center == CENTER, f"mirror landed on {center!r}, not the centre"
    return left, right


def letters(words):
    return len(normalize(" ".join(words)))


class TestCommittedFrames:
    def test_every_frame_contains_the_one_before(self):
        frames, _ = collect()
        assert len(frames) >= 3, f"expected a stream, got {len(frames)} frame(s)"
        for older, newer in zip(frames, frames[1:]):
            lo, ro = halves(older)
            ln, rn = halves(newer)
            assert ln[len(ln) - len(lo):] == lo, (
                f"left half stopped containing its predecessor:\n{lo}\n{ln}")
            assert rn[:len(ro)] == ro, (
                f"right half stopped containing its predecessor:\n{ro}\n{rn}")

    def test_frames_only_ever_grow(self):
        frames, _ = collect()
        sizes = [letters(f) for f in frames]
        assert sizes == sorted(sizes)
        assert sizes[-1] > sizes[0]

    def test_every_frame_is_itself_a_palindrome(self):
        """Frames are shown to the reader, so one that does not read the same
        backwards is a lie on screen, not merely an intermediate value."""
        for f in collect()[0]:
            assert is_palindrome(" ".join(f)), f"frame is not a palindrome: {f}"

    def test_the_answer_is_the_last_frame(self):
        """Returning anything better than what was published would force the page
        to rewrite at the very end — the one moment it looks most like a bug."""
        frames, words = collect()
        assert words == frames[-1]

    @pytest.mark.parametrize("seed", [1, 2, 3])
    @pytest.mark.parametrize("commit_every", [0.02, 0.05, 0.1])
    def test_holds_across_intervals_and_seeds(self, commit_every, seed):
        frames, words = collect(commit_every=commit_every, seed=seed)
        sizes = [letters(f) for f in frames]
        assert len(frames) >= 2 and sizes == sorted(sizes)
        assert words == frames[-1]


class TestFallbacks:
    def test_a_search_that_ends_before_publishing_still_returns(self):
        """The early stop can fire inside the first interval. Nothing was shown,
        so nothing can be contradicted: the free best stands, and it is sent as
        the one and only frame rather than being dropped on the floor."""
        frames, words = collect(commit_every=5.0, floor=40, budget=1.0)
        assert words, "a search that never published returned nothing at all"
        assert is_palindrome(" ".join(words))
        assert frames == [words]

    def test_uncommitted_search_still_reports_every_closure(self):
        """Without an interval the stream is not append-only and the caller sees
        everything, which is what the non-streaming callers still expect."""
        frames, words = collect(commit_every=None, floor=40, budget=0.6)
        sizes = [letters(f) for f in frames]
        assert len(frames) > 3
        assert sizes != sorted(sizes), (
            "a free beam wanders; sorted sizes mean this is not exercising the "
            "uncommitted path any more")
        assert is_palindrome(" ".join(words))

    def test_no_callback_is_fine(self):
        words = centerout_search(
            WordTries(TINY_DICT), FreqScorer(TINY_DICT), min_letters=40,
            beam_width=30, center=CENTER, seed=1, max_steps=10**6,
            deadline=time.monotonic() + 0.5, commit_every=0.05)
        assert words and is_palindrome(" ".join(words))
