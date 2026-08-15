"""Norvig-style two-sided palindrome search, guided by a pluggable scorer.

The palindrome grows from the outside in. At every step one half "owes" the
other a run of letters — the overhang. Words added to the left are matched
forward against the overhang; words added to the right are matched with their
letters reversed. The search closes when the overhang is itself a palindrome,
which becomes the center of the final text.

This is the algorithm behind Peter Norvig's palindrome program (2002), itself
building on Dan Hoey's 1984 one; the scorer is what's new — it lets a language
model decide which of the letter-valid branches read as English.
"""
from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field
from typing import Optional, Sequence


def unit_letters(unit: str) -> str:
    """The letters a unit contributes to the palindrome.

    A unit used to be a single word, so its spelling and its letters were the
    same string and the search could use one for the other. A phrase separates
    them: "new york" occupies eight letters of the mirror, not nine, and its
    reflection is "kroywen". Every place the search reverses, measures or
    matches a unit goes through here; every place it PRINTS one does not.
    """
    return unit.replace(" ", "")


def consume(letters: str, overhang: str) -> Optional[tuple[str, bool]]:
    """Match a word's letters against the current overhang.

    Returns (new_overhang, flipped) where flipped means the remainder is now
    owed by the opposite side, or None if the letters don't line up.
    """
    if overhang.startswith(letters):
        return overhang[len(letters):], False
    if letters.startswith(overhang):
        return letters[len(overhang):], True
    return None


class _TrieNode:
    __slots__ = ("children", "words")

    def __init__(self):
        self.children: dict[str, _TrieNode] = {}
        self.words: list[str] = []  # words terminating exactly here


class _Trie:
    def __init__(self, keyed_words: Sequence[tuple[str, str]]):
        self.root = _TrieNode()
        for key, word in keyed_words:
            node = self.root
            for ch in key:
                node = node.children.setdefault(ch, _TrieNode())
            node.words.append(word)

    def candidates(self, overhang: str, limit: int = 200) -> list[str]:
        """Words whose key is a prefix of overhang, or has overhang as prefix."""
        out: list[str] = []
        node = self.root
        # keys that are prefixes of the overhang (word swallowed by overhang)
        for ch in overhang:
            if node.words:
                out.extend(node.words)
            node = node.children.get(ch)
            if node is None:
                return out[:limit]
        # keys that begin with the whole overhang (word overruns it)
        stack = [node]
        while stack and len(out) < limit:
            n = stack.pop()
            out.extend(n.words)
            stack.extend(n.children.values())
        return out[:limit]


class WordTries:
    """Forward trie for left-side matches, reversed trie for right-side.

    Units may be single words or multi-word phrases. Both tries are keyed on a
    unit's LETTERS, so a phrase is reachable by the run of letters it would
    fill and never by its spaced spelling — the overhang has no spaces in it.
    """

    def __init__(self, words: Sequence[str]):
        seen = sorted({w.lower() for w in words
                       if w and unit_letters(w).isalpha()})
        self.words = seen
        self._fwd = _Trie([(unit_letters(w), w) for w in seen])
        self._rev = _Trie([(unit_letters(w)[::-1], w) for w in seen])

    def left_candidates(self, overhang: str, limit: int = 200) -> list[str]:
        return self._fwd.candidates(overhang, limit)

    def right_candidates(self, overhang: str, limit: int = 200) -> list[str]:
        return self._rev.candidates(overhang, limit)


@dataclass(order=True)
class State:
    sort_key: float
    left: tuple[str, ...] = field(compare=False)
    right: tuple[str, ...] = field(compare=False)  # final order; grown by prepending
    overhang: str = field(compare=False)
    side: str = field(compare=False)  # 'L': left owes letters; 'R': right owes
    score: float = field(compare=False, default=0.0)

    @property
    def letters(self) -> int:
        return (sum(len(unit_letters(w)) for w in self.left)
                + sum(len(unit_letters(w)) for w in self.right))


def _expand(state: State, tries: WordTries, limit: int) -> list[tuple[str, str, str, str]]:
    """Yield (placement, word, new_overhang, new_side) for legal extensions.

    side 'L' means the LEFT half has unmatched letters the right must mirror;
    we then add words to the RIGHT (matched reversed). side 'R' is symmetric.
    An empty overhang allows growth on the right by convention (either would do).
    """
    out = []
    if state.side == "L" or not state.overhang:
        for w in tries.right_candidates(state.overhang, limit):
            res = consume(unit_letters(w)[::-1], state.overhang)
            if res is not None:
                new_over, flipped = res
                out.append(("R", w, new_over, "R" if flipped else "L"))
    if state.side == "R" and state.overhang:
        for w in tries.left_candidates(state.overhang, limit):
            res = consume(unit_letters(w), state.overhang)
            if res is not None:
                new_over, flipped = res
                out.append(("L", w, new_over, "L" if flipped else "R"))
    return out


def beam_search(
    tries: WordTries,
    scorer,
    min_letters: int = 60,
    beam_width: int = 50,
    max_steps: int = 400,
    candidate_limit: int = 200,
    seed: Optional[int] = None,
    diversity: float = 0.4,
    prune=None,
    prune_every: int = 8,
) -> list[str]:
    """Beam search for a word sequence whose letters form a palindrome.

    Returns the words of the best closed palindrome found, [] on failure.
    A state can close when its overhang is itself a palindrome (the center).

    `prune(states) -> states` is called every `prune_every` steps; a language
    model uses it to drop branches that are letter-valid but not fluent. It may
    reorder or filter but must not fabricate states, so correctness is unaffected.
    """
    rng = random.Random(seed)
    start = State(sort_key=0.0, left=(), right=(), overhang="", side="L")
    beam = [start]
    best: Optional[tuple[float, list[str]]] = None

    for step in range(max_steps):
        if not beam:
            break
        # A scorer may amortize model calls across a whole beam step; see
        # DirectionalScorer.prepare. Optional, so plain scorers stay simple.
        if hasattr(scorer, "prepare"):
            scorer.prepare(beam)
        pool: list[State] = []
        for state in beam:
            over = state.overhang
            closable = over == over[::-1]
            if closable and state.letters >= min_letters:
                words = list(state.left) + list(state.right)
                per_letter = state.score / max(1, state.letters)
                if best is None or per_letter > best[0]:
                    best = (per_letter, words)
            for placement, w, new_over, new_side in _expand(state, tries, candidate_limit):
                if len(new_over) > 24:  # unmatchable overhangs stall the search
                    continue
                if placement == "L":
                    left, right = state.left + (w,), state.right
                else:
                    left, right = state.left, (w,) + state.right
                # Outside-in: the left half is appended to, the right prepended.
                growth = "append" if placement == "L" else "prepend"
                sc = state.score + scorer.word_delta(left, right, placement, w,
                                                     growth)
                sc += rng.random() * diversity  # jitter so seeds explore differently
                pool.append(State(sort_key=-sc, left=left, right=right,
                                  overhang=new_over, side=new_side, score=sc))
        if best is not None and not pool:
            break
        beam = heapq.nsmallest(beam_width, pool)
        if prune is not None and beam and step % prune_every == prune_every - 1:
            beam = list(prune(beam))
        if best is not None and all(s.letters > 3 * min_letters for s in beam):
            break

    return best[1] if best else []
