"""Walk the whole space of short palindromes instead of steering through it.

Every other search here is a beam. A beam is the right tool when the target is
long, because the space is unwalkable and the only question is which corner of
it to explore. It is the wrong tool when the target is short, and the record
says the readable palindromes are short: "A man, a plan, a canal: Panama" is 24
letters, "Sir, I demand, I am a maid named Iris" is 30. Nothing longer than a
732-character poem is considered readable by anyone, and the 90,439-letter
record holder is a noun list its own author calls nonsense.

At 24 letters the space can be walked. That changes what a language model is
for: not steering the search — which `experiments/` shows repeatedly does not
work — but choosing among everything that exists. A best-of-N over an
exhaustive enumeration is not the same object as a best-of-N over a beam's
output, and the oracle bound measured in `oracle_bound.py` says nothing about
it, because that bound was over a fixed proposal distribution and this has no
proposal distribution at all.

The state is the overhang, exactly as in `centerout`. The difference is that
nothing is discarded: every branch is followed until it closes or exceeds the
letter budget.
"""
from __future__ import annotations

from typing import Iterator, Optional, Sequence

from .centerout import _expand, COState
from .search import WordTries, unit_letters


def enumerate_palindromes(tries: WordTries, max_letters: int = 30,
                          min_letters: int = 0, max_overhang: int = 20,
                          shard: int = 0, shards: int = 1,
                          node_budget: int = 10 ** 9,
                          max_units: int = 12,
                          deadline: Optional[float] = None,
                          shuffle_seed: Optional[int] = None,
                          allow_join=None,
                          join_slack: int = 0) -> Iterator[list[str]]:
    """Every palindrome the vocabulary admits within `max_letters`.

    Sharded on the OPENING unit so that ranks partition the space exactly and
    never duplicate: the first placement determines a disjoint subtree, and a
    rank that takes every k-th opening takes a disjoint set of them.

    `node_budget` bounds the walk. Exhaustive within a budget is honest;
    exhaustive without one is a promise the space may not keep.

    `allow_join(before, after)` is asked about every adjacency the walk is
    about to create, and a False prunes that whole subtree. It is where a
    requirement like "every join is one English has been seen to make" belongs:
    applied afterwards it is a filter that rejects almost everything the walk
    produced, and applied here it is a constraint that stops the walk producing
    it. The left half grows by prepending and the right by appending, so the
    new adjacency is (w, left[0]) on one side and (right[-1], w) on the other.
    The junction between the halves is never asked about — the two halves are
    different sentences, and English does not have to join them.

    `join_slack` is how many refused joins a branch may take anyway. Requiring
    every join to be attested is severe — English makes joins it has not made
    before all day — and a budget of one turns "every adjacency is idiomatic"
    into "all but one is", which is the difference between a phrase book and a
    sentence. The budget is per branch and is spent, not refreshed.
    """
    import random as _random
    rng = _random.Random(shuffle_seed) if shuffle_seed is not None else None

    root = COState(sort_key=0.0, left=(), right=(), overhang="", owner="R",
                   center_len=0)
    nodes = 0

    # The opening placements, in a fixed order, so sharding is deterministic.
    openings = _expand(root, tries, limit=10 ** 6)
    openings = [o for i, o in enumerate(openings) if i % shards == shard]
    # A LIFO stack over a sorted trie drills into whatever sorts first, and a
    # time budget then expires inside that one corner: 2.55M results contained
    # none of the 27 canonical palindromes, several of which this enumerator
    # produces instantly on a small vocabulary. Shuffling the frontier turns
    # the walk back into a sample of the space.
    if rng is not None:
        rng.shuffle(openings)

    # Each entry carries the slack its branch has left, because a budget that
    # lived on the state would be shared by siblings that never met.
    stack: list[tuple[COState, int]] = []
    for placement, w, new_over, new_owner in openings:
        if len(new_over) > max_overhang or len(unit_letters(w)) > max_letters:
            continue
        stack.append((COState(sort_key=0.0, left=(w,), right=(), overhang=new_over,
                              owner=new_owner, center_len=0), join_slack))

    import time as _time
    while stack:
        if nodes >= node_budget:
            return
        # Checked coarsely: a syscall per node would dominate the walk.
        if deadline is not None and nodes % 4096 == 0 and _time.time() > deadline:
            return
        nodes += 1
        state, slack = stack.pop()

        if not state.overhang:
            if state.letters >= min_letters:
                yield list(state.left) + list(state.right)
            # A closed state can still be extended, so it is not a leaf.

        if state.letters >= max_letters or len(state.left) + len(state.right) >= max_units:
            continue

        expansions = _expand(state, tries, limit=10 ** 6)
        if rng is not None:
            rng.shuffle(expansions)
        for placement, w, new_over, new_owner in expansions:
            if len(new_over) > max_overhang:
                continue
            join = None
            if placement == "L":
                if allow_join is not None and state.left:
                    join = (w, state.left[0])
                left, right = (w,) + state.left, state.right
            else:
                if allow_join is not None and state.right:
                    join = (state.right[-1], w)
                left, right = state.left, state.right + (w,)
            left_slack = slack
            if join is not None and not allow_join(*join):
                if left_slack <= 0:
                    continue
                left_slack -= 1
            nxt = COState(sort_key=0.0, left=left, right=right, overhang=new_over,
                          owner=new_owner, center_len=0)
            if nxt.letters > max_letters:
                continue
            stack.append((nxt, left_slack))


def acceptable_words(words, min_mean_len: float = 3.0) -> bool:
    """Structural filters an exhaustive walk cannot do without.

    Walking the whole space means finding every degenerate closure too. "aaa"
    is in the frequency list because the web contains it, it fits any overhang,
    and a first run returned "ann aaa aaron nora aaa anna" as its best result.
    A word that is one letter repeated is filler wearing a word's clothes.
    """
    if not words:
        return False
    if any(len(set(w)) == 1 and len(w) >= 2 for w in words):
        return False
    mean_len = sum(len(w) for w in words) / len(words)
    return mean_len >= min_mean_len


def hunt_vocabulary(words, zipf, min_zipf: float = 3.5) -> list[str]:
    """The units the walk is allowed to build from.

    Filtering closures after the fact does not work: the trie sorts its units
    and the walk is depth-first, so a degenerate unit near the front of the
    alphabet — "aaa" — absorbs the entire node budget and every result is
    discarded. A word that could never survive the acceptance filter has to be
    withheld from the trie instead.
    """
    return [w for w in words
            if not (len(set(w)) == 1 and len(w) >= 2) and zipf(w) >= min_zipf]
