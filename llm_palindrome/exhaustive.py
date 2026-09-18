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

from collections import deque
import hashlib
from typing import Iterator, Optional, Sequence

from .centerout import _expand, COState
from .search import WordTries, _score_choices, unit_letters


def enumerate_palindromes(tries: WordTries, max_letters: int = 30,
                          min_letters: int = 0, max_overhang: int = 20,
                          shard: int = 0, shards: int = 1,
                          node_budget: int = 10 ** 9,
                          max_units: int = 12,
                          deadline: Optional[float] = None,
                          shuffle_seed: Optional[int] = None,
                          allow_join=None,
                          join_slack: int = 0,
                          allow_state=None,
                          stats: Optional[dict] = None,
                          scorer=None,
                          order_seed: Optional[int] = None,
                          traversal: str = "dfs",
                          reverse_order: bool = False) -> Iterator[list[str]]:
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

    `allow_state(left, right)` can prune using properties of both partial
    halves. It is evaluated on openings and every expanded state. `stats`, if
    supplied, receives candidate, generated, pushed, popped, closure,
    state-prune, frontier, and yield counts for experiments.
    `scorer` orders siblings but never deletes one. This lets a language model
    spend a finite node budget in promising subtrees without turning the walk
    back into a beam whose discarded lineage can never close.

    `order_seed` gives every vocabulary item a stable seeded rank and applies
    that same ordering at every expansion. Unlike `shuffle_seed`, it consumes
    no traversal-dependent random state: a partial state reached by two search
    configurations therefore receives the same sibling order in both. This is
    intended for controlled comparisons. The two ordering modes are mutually
    exclusive.

    `traversal` selects a depth-first stack (the default) or a breadth-first
    queue. `reverse_order` reverses the ordered sibling list before inserting
    it into that frontier. They exist chiefly for bounded-search sensitivity
    experiments; neither changes the set of solutions when the walk exhausts.
    """
    import random as _random
    if shuffle_seed is not None and order_seed is not None:
        raise ValueError("shuffle_seed and order_seed are mutually exclusive")
    if traversal not in {"dfs", "bfs"}:
        raise ValueError("traversal must be 'dfs' or 'bfs'")
    rng = _random.Random(shuffle_seed) if shuffle_seed is not None else None
    order_rank = None
    if order_seed is not None:
        prefix = str(order_seed).encode("ascii") + b"\0"
        ranked_words = sorted(
            tries.words,
            key=lambda word: (hashlib.sha256(prefix + word.encode("utf-8")).digest(),
                              word),
        )
        order_rank = {word: rank for rank, word in enumerate(ranked_words)}

    def order(expansions):
        if rng is not None:
            rng.shuffle(expansions)
        elif order_rank is not None:
            expansions.sort(key=lambda row: order_rank[row[1]])
        if reverse_order:
            expansions.reverse()

    root = COState(sort_key=0.0, left=(), right=(), overhang="", owner="R",
                   center_len=0)
    nodes = 0
    if stats is not None:
        stats.update(nodes=0, states_popped=0, expansion_calls=0,
                     candidate_expansions=0, states_generated=0,
                     states_pushed=0, state_pruned=0, closed_states=0,
                     yielded=0, peak_frontier=0)

    # The opening placements, in a fixed order, so sharding is deterministic.
    openings = _expand(root, tries, limit=10 ** 6)
    if stats is not None:
        stats["expansion_calls"] += 1
        stats["candidate_expansions"] += len(openings)
    openings = [o for i, o in enumerate(openings) if i % shards == shard]
    # A LIFO stack over a sorted trie drills into whatever sorts first, and a
    # time budget then expires inside that one corner: 2.55M results contained
    # none of the 27 canonical palindromes, several of which this enumerator
    # produces instantly on a small vocabulary. Shuffling the frontier turns
    # the walk back into a sample of the space.
    order(openings)
    if scorer is not None:
        choices = [((w,), (), "L", w, "prepend")
                   for _, w, _, _ in openings]
        scores = _score_choices(scorer, choices,
                                [new_over for _, _, new_over, _ in openings])
        openings = [row for _, row in sorted(zip(scores, openings),
                                             key=lambda item: item[0])]

    # Each entry carries the slack its branch has left, because a budget that
    # lived on the state would be shared by siblings that never met.
    frontier = ([] if traversal == "dfs" else deque())
    for placement, w, new_over, new_owner in openings:
        if len(new_over) > max_overhang or len(unit_letters(w)) > max_letters:
            continue
        opening = COState(sort_key=0.0, left=(w,), right=(), overhang=new_over,
                          owner=new_owner, center_len=0)
        if stats is not None:
            stats["states_generated"] += 1
        if allow_state is not None and not allow_state(opening.left, opening.right):
            if stats is not None:
                stats["state_pruned"] += 1
            continue
        frontier.append((opening, join_slack))
        if stats is not None:
            stats["states_pushed"] += 1
    if stats is not None:
        stats["peak_frontier"] = len(frontier)

    import time as _time
    while frontier:
        if nodes >= node_budget:
            if stats is not None:
                stats["stop_reason"] = "node_budget"
            return
        # Checked coarsely: a syscall per node would dominate the walk.
        if deadline is not None and nodes % 4096 == 0 and _time.time() > deadline:
            if stats is not None:
                stats["stop_reason"] = "deadline"
            return
        nodes += 1
        if stats is not None:
            stats["nodes"] = nodes
            stats["states_popped"] = nodes
        state, slack = (frontier.pop() if traversal == "dfs"
                        else frontier.popleft())

        if not state.overhang:
            if stats is not None:
                stats["closed_states"] += 1
            if state.letters >= min_letters:
                if stats is not None:
                    stats["yielded"] += 1
                yield list(state.left) + list(state.right)
            # A closed state can still be extended, so it is not a leaf.

        if state.letters >= max_letters or len(state.left) + len(state.right) >= max_units:
            continue

        expansions = _expand(state, tries, limit=10 ** 6)
        if stats is not None:
            stats["expansion_calls"] += 1
            stats["candidate_expansions"] += len(expansions)
        order(expansions)
        children = []
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
            if stats is not None:
                stats["states_generated"] += 1
            if allow_state is not None and not allow_state(left, right):
                if stats is not None:
                    stats["state_pruned"] += 1
                continue
            growth = "prepend" if placement == "L" else "append"
            children.append((nxt, left_slack,
                             (left, right, placement, w, growth), new_over))
        if scorer is not None and children:
            scores = _score_choices(scorer, [row[2] for row in children],
                                    [row[3] for row in children])
            children = [row for _, row in sorted(zip(scores, children),
                                                  key=lambda item: item[0])]
        for nxt, left_slack, _, _ in children:
            frontier.append((nxt, left_slack))
            if stats is not None:
                stats["states_pushed"] += 1
        if stats is not None:
            stats["peak_frontier"] = max(stats["peak_frontier"], len(frontier))
    if stats is not None:
        stats["stop_reason"] = "exhausted"


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
