"""What can be checked rather than judged.

Reinforcement learning against a learned judge invites the policy to find the
judge's blind spots. The defence here is unusually strong, because most of what
this task demands is decidable: whether the text is a palindrome, whether every
word is a real word, whether it closed, how long it is. Only readability needs
a model's opinion.

So the reward splits in two. The verifiable part is computed exactly and, where
the search already guarantees it, *asserted* — a violation is a bug in the
search, not a low score, and `verify` says so by raising. The judged part is
the only place a policy can chase an artifact, and it is bounded by the
verifiable part's requirements.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from .textify import textify
from .validator import is_palindrome, normalize


class InvariantViolation(AssertionError):
    """The search produced something it is supposed to be unable to produce."""


@dataclass(frozen=True)
class Verified:
    """Exact, checkable properties of a finished candidate."""

    closed: bool
    letters: int
    words: int
    distinct_words: int
    adjacent_repeats: int
    max_word_uses: int
    short_word_rate: float          # words of 1-2 letters
    is_palindrome: bool             # asserted, never merely scored
    all_in_vocabulary: bool         # asserted

    def reward(self, target_letters: int,
               repeat_penalty: float = 2.0,
               short_penalty: float = 1.0,
               length_weight: float = 1.0) -> float:
        """Verifiable reward in [-inf, length_weight].

        Length saturates at the target: past it, more letters are not better,
        and rewarding them is what turns the search into a filler generator.
        """
        if not self.closed:
            return -10.0
        length = min(1.0, self.letters / max(1, target_letters))
        repeats = self.adjacent_repeats / max(1, self.words - 1)
        return (length_weight * length
                - repeat_penalty * repeats
                - short_penalty * self.short_word_rate)


def verify(words: Sequence[str], vocabulary: Optional[Iterable[str]] = None,
           strict: bool = True) -> Verified:
    """Check a finished candidate. Raises on a broken invariant when strict.

    `is_palindrome` and `all_in_vocabulary` are guarantees of the search, not
    achievements of the policy. Returning them as a low reward would let a
    policy trade them away; raising makes them non-negotiable.
    """
    words = list(words)
    if not words:
        return Verified(False, 0, 0, 0, 0, 0, 0.0, True, True)

    text = textify(words)
    pal = is_palindrome(text)
    vocab = set(vocabulary) if vocabulary is not None else None
    in_vocab = True if vocab is None else all(w in vocab for w in words)

    if strict and not pal:
        raise InvariantViolation(
            f"search returned a non-palindrome: {normalize(text)[:80]!r}")
    if strict and not in_vocab:
        missing = sorted({w for w in words if w not in vocab})[:5]
        raise InvariantViolation(f"words outside the vocabulary: {missing}")

    counts: dict[str, int] = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1

    return Verified(
        closed=True,
        letters=sum(len(w) for w in words),
        words=len(words),
        distinct_words=len(counts),
        adjacent_repeats=sum(1 for a, b in zip(words, words[1:]) if a == b),
        max_word_uses=max(counts.values()),
        short_word_rate=sum(1 for w in words if len(w) <= 2) / len(words),
        is_palindrome=pal,
        all_in_vocabulary=in_vocab,
    )
