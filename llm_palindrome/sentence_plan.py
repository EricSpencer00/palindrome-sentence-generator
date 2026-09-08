"""Incremental grammatical plans for exhaustive mirror-pair search.

The exhaustive palindrome walk grows away from its mirror: words are
prepended to the left half and appended to the right.  A completed left half
therefore has the current left words as a suffix, while a completed right half
has the current right words as a prefix.  Testing those two facts during the
walk avoids spending nearly all of the node budget on halves which can never
be sentence-shaped.

This is deliberately only a structural gate.  Corpus POS shapes do not decide
whether a sentence is meaningful; surviving pairs still need diversity and
semantic evaluation.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import Iterable, Mapping, Sequence

from .syntax import OPENING_TAGS


class SentencePlan:
    """Fast prefix/suffix feasibility over attested sentence tag shapes."""

    def __init__(self, table: Mapping[str, Iterable[str]],
                 shapes: Iterable[Sequence[str]], min_words: int = 3,
                 max_words: int = 9):
        self.table = {word: frozenset(tags) for word, tags in table.items()}
        kept = {tuple(shape) for shape in shapes
                if min_words <= len(shape) <= max_words
                and shape and shape[0] in OPENING_TAGS and "VERB" in shape}
        self.shapes = kept
        by_length: dict[int, list[tuple[str, ...]]] = defaultdict(list)
        for shape in kept:
            by_length[len(shape)].append(shape)
        self.by_length = dict(by_length)
        self.complete_shapes = {length: set(items)
                                for length, items in self.by_length.items()}
        self.prefixes: dict[int, set[tuple[str, ...]]] = defaultdict(set)
        self.suffixes: dict[int, set[tuple[str, ...]]] = defaultdict(set)
        for shape in kept:
            for length in range(1, len(shape) + 1):
                self.prefixes[length].add(shape[:length])
                self.suffixes[length].add(shape[-length:])
        self.min_words = min_words
        self.max_words = max_words

    def _tags(self, words: Sequence[str]):
        pools = []
        for word in words:
            tags = self.table.get(word.lower())
            if not tags:
                return None
            pools.append(tags)
        return pools

    @staticmethod
    def _any_reading(pools, allowed: set[tuple[str, ...]]) -> bool:
        return any(tuple(reading) in allowed for reading in product(*pools))

    def prefix_possible(self, words: Sequence[str]) -> bool:
        """Can `words` begin some complete sentence plan?"""
        if len(words) > self.max_words:
            return False
        if not words:
            return bool(self.shapes)
        pools = self._tags(words)
        if pools is None:
            return False
        return self._any_reading(pools, self.prefixes[len(words)])

    def suffix_possible(self, words: Sequence[str]) -> bool:
        """Can `words` end some complete sentence plan?"""
        if len(words) > self.max_words:
            return False
        if not words:
            return bool(self.shapes)
        pools = self._tags(words)
        if pools is None:
            return False
        return self._any_reading(pools, self.suffixes[len(words)])

    def complete(self, words: Sequence[str]) -> bool:
        """Do the words realize one complete subject-and-verb plan?"""
        pools = self._tags(words)
        if pools is None:
            return False
        return self._any_reading(pools, self.complete_shapes.get(len(words), set()))

    def state_possible(self, left: Sequence[str], right: Sequence[str]) -> bool:
        return self.suffix_possible(left) and self.prefix_possible(right)
