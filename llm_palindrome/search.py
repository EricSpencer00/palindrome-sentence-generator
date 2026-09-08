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
    __slots__ = ("children", "words", "ranked")

    def __init__(self):
        self.children: dict[str, _TrieNode] = {}
        self.words: list[str] = []  # words terminating exactly here
        # Filled once the trie has been built.  It is deliberately ordered by
        # the caller's vocabulary rank, never by traversal order.
        self.ranked: list[str] = []


class _Trie:
    def __init__(self, keyed_words: Sequence[tuple[str, str, int]]):
        self.root = _TrieNode()
        self.rank: dict[str, int] = {}
        for key, word, rank in keyed_words:
            self.rank[word] = rank
            node = self.root
            for ch in key:
                node = node.children.setdefault(ch, _TrieNode())
            node.words.append(word)
        self._rank_descendants(self.root)

    def _rank_descendants(self, node: _TrieNode) -> list[str]:
        """Cache every subtree's words in vocabulary-rank order.

        A trie is an index for legal words, not a policy for choosing them.
        The old breadth-first walk accidentally made word length, then spelling,
        the policy whenever a candidate limit was applied.  Caching this list
        keeps lookup cheap without letting the shape of the trie decide which
        English words the scorer is allowed to see.
        """
        words = list(node.words)
        for child in node.children.values():
            words.extend(self._rank_descendants(child))
        words.sort(key=self.rank.__getitem__)
        node.ranked = words
        return words

    @staticmethod
    def _length_bucket(word: str) -> int:
        n = len(unit_letters(word))
        if n <= 2:
            return 0
        if n <= 4:
            return 1
        if n <= 7:
            return 2
        return 3

    def _limit(self, words: list[str], limit: int) -> list[str]:
        """Keep ranked candidates without making a short-word-only menu.

        The initial menu needs common function words *and* normal content
        words.  Round-robinning four length bands gives the scorer both while
        retaining vocabulary order inside every band.  No work is done when the
        caller requests the complete legal set (the graph/enumerator path).
        """
        if limit <= 0:
            return []
        if len(words) <= limit:
            return words
        bands = [[] for _ in range(4)]
        for word in words:
            bands[self._length_bucket(word)].append(word)
        out: list[str] = []
        at = [0] * len(bands)
        while len(out) < limit:
            progressed = False
            for i, band in enumerate(bands):
                if at[i] >= len(band):
                    continue
                out.append(band[at[i]])
                at[i] += 1
                progressed = True
                if len(out) == limit:
                    break
            if not progressed:
                break
        return out

    def candidates(self, overhang: str, limit: int = 200) -> list[str]:
        """Words whose key is a prefix of overhang, or has overhang as prefix.

        Candidate truncation is a proposal policy.  It must not inherit a
        breadth-first traversal's shortest-word ordering: that made an empty
        30k-word trie offer only one-to-three-letter words.  Legal candidates
        are recovered in the input vocabulary's rank order and, when limited,
        are balanced across word lengths before a scorer ranks them.
        """
        consumed: list[str] = []
        node = self.root
        # keys that are prefixes of the overhang (word swallowed by overhang)
        for ch in overhang:
            if node.words:
                consumed.extend(node.words)
            node = node.children.get(ch)
            if node is None:
                consumed.sort(key=self.rank.__getitem__)
                return consumed[:limit]
        # A word that consumes all or part of the debt is structurally special:
        # withholding it because a length bucket is full can remove the only
        # route to closure.  Offer all such prefix matches before stratifying
        # the words that overrun the debt.
        consumed.extend(node.words)
        consumed.sort(key=self.rank.__getitem__)
        if len(consumed) >= limit:
            return consumed[:limit]
        consumed_set = set(consumed)
        overruns = [word for word in node.ranked if word not in consumed_set]
        return consumed + self._limit(overruns, limit - len(consumed))


class WordTries:
    """Forward trie for left-side matches, reversed trie for right-side.

    Units may be single words or multi-word phrases. Both tries are keyed on a
    unit's LETTERS, so a phrase is reachable by the run of letters it would
    fill and never by its spaced spelling — the overhang has no spaces in it.
    """

    def __init__(self, words: Sequence[str]):
        # `build_vocab` is frequency-ranked.  Preserve that useful prior and
        # make duplicate removal stable instead of sorting the vocabulary into
        # an unrelated alphabetic order.
        seen: list[str] = []
        known: set[str] = set()
        for raw in words:
            w = raw.lower()
            if not w or not unit_letters(w).isalpha() or w in known:
                continue
            known.add(w)
            seen.append(w)
        self.words = seen
        self._fwd = _Trie([(unit_letters(w), w, i) for i, w in enumerate(seen)])
        self._rev = _Trie([(unit_letters(w)[::-1], w, i) for i, w in enumerate(seen)])

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


def _score_choices(scorer, choices: Sequence[tuple[tuple[str, ...], tuple[str, ...],
                                                   str, str, str]],
                   overhangs: Optional[Sequence[str]] = None) -> list[float]:
    """Score a parent's legal children together when the scorer supports it."""
    if hasattr(scorer, "word_deltas"):
        if getattr(scorer, "wants_overhang", False):
            assert overhangs is not None
            return list(scorer.word_deltas(choices, overhangs))
        return list(scorer.word_deltas(choices))
    if getattr(scorer, "wants_overhang", False):
        assert overhangs is not None
        return [scorer.word_delta(left, right, placement, word, growth, overhang=overhang)
                for (left, right, placement, word, growth), overhang
                in zip(choices, overhangs)]
    return [scorer.word_delta(left, right, placement, word, growth)
            for left, right, placement, word, growth in choices]


def _parent_width(beam_width: int, parents: int, configured: Optional[int]) -> int:
    """Prevent one parent from spending the whole beam on sibling variants."""
    if configured is not None:
        return max(1, configured)
    # The first state needs a whole beam's worth of starts; thereafter every
    # surviving parent gets at least two chances before global selection.
    return max(2, (beam_width + max(1, parents) - 1) // max(1, parents))


def beam_search(
    tries: WordTries,
    scorer,
    min_letters: int = 60,
    beam_width: int = 50,
    max_steps: int = 400,
    candidate_limit: int = 200,
    per_parent: Optional[int] = None,
    seed: Optional[int] = None,
    diversity: float = 0.4,
    prune=None,
    prune_every: int = 8,
    opening_words: Optional[set[str]] = None,
    max_word_uses: Optional[int] = None,
) -> list[str]:
    """Beam search for a word sequence whose letters form a palindrome.

    Returns the words of the best closed palindrome found, [] on failure.
    A state can close when its overhang is itself a palindrome (the center).

    `opening_words` is a hard constraint on the first unit of the finished
    text. It is applied when the left half first appears, not to the initial
    search move (which becomes the text's final unit). `max_word_uses` is a
    hard cap across individual words, including words inside phrase units.

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
        # A scorer may prepare state-level caches.  Candidate-level scorers get
        # the complete legal menu for each parent below.
        if hasattr(scorer, "prepare"):
            scorer.prepare(beam)
        pool: list[State] = []
        parent_limit = _parent_width(beam_width, len(beam), per_parent)
        for state in beam:
            over = state.overhang
            closable = over == over[::-1]
            if closable and state.letters >= min_letters:
                words = list(state.left) + list(state.right)
                per_letter = state.score / max(1, state.letters)
                if best is None or per_letter > best[0]:
                    best = (per_letter, words)
            child_specs = []
            choices = []
            for placement, w, new_over, new_side in _expand(state, tries, candidate_limit):
                if len(new_over) > 24:  # unmatchable overhangs stall the search
                    continue
                if (opening_words is not None and placement == "L"
                        and not state.left and w not in opening_words):
                    continue
                if max_word_uses is not None:
                    existing = [part for unit in state.left + state.right
                                for part in unit.split()]
                    added = w.split()
                    if any(existing.count(part) + added.count(part) > max_word_uses
                           for part in set(added)):
                        continue
                if placement == "L":
                    left, right = state.left + (w,), state.right
                else:
                    left, right = state.left, (w,) + state.right
                # Outside-in: the left half is appended to, the right prepended.
                growth = "append" if placement == "L" else "prepend"
                child_specs.append((left, right, placement, w, new_over, new_side, growth))
                choices.append((left, right, placement, w, growth))
            deltas = _score_choices(scorer, choices,
                                    [spec[4] for spec in child_specs])
            children: list[State] = []
            for spec, delta in zip(child_specs, deltas):
                left, right, placement, w, new_over, new_side, growth = spec
                semantic = state.score + delta
                # Exploration controls ranking only.  It must never become
                # part of a completed palindrome's reported objective.
                priority = semantic + rng.random() * diversity
                children.append(State(sort_key=-priority, left=left, right=right,
                                      overhang=new_over, side=new_side, score=semantic))
            pool.extend(heapq.nsmallest(parent_limit, children))
        if best is not None and not pool:
            break
        beam = heapq.nsmallest(beam_width, pool)
        if prune is not None and beam and step % prune_every == prune_every - 1:
            beam = list(prune(beam))
        if best is not None and all(s.letters > 3 * min_letters for s in beam):
            break

    return best[1] if best else []
