"""Center-out palindrome growth: the mirror image of the outside-in search.

Both directions satisfy the constraint; they differ in where the slack lives
and what it takes to finish.

  outside-in  the two ends are fixed first, slack accumulates in the MIDDLE,
              and the text closes when that slack is itself a palindrome.
  center-out  the middle is fixed first, slack accumulates at the OUTER EDGES,
              and the text closes only when that slack is exactly empty.

"Palindromic" is a far weaker closing condition than "empty" — every single
letter satisfies it — which is the reason to expect outside-in to close more
often. direction_study.py measures whether it actually does.
"""
from __future__ import annotations

import heapq
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .search import WordTries, consume


def consume_suffix(letters: str, overhang: str) -> Optional[tuple[str, bool]]:
    """Match a word against the END of the owed run.

    Left-side words are prepended, so they sit immediately before the existing
    text and therefore fill the tail of the overhang, not its head.
    """
    res = consume(letters[::-1], overhang[::-1])
    if res is None:
        return None
    remainder, flipped = res
    return remainder[::-1], flipped


@dataclass(order=True)
class COState:
    sort_key: float
    left: tuple[str, ...] = field(compare=False)   # final reading order
    right: tuple[str, ...] = field(compare=False)  # final reading order
    overhang: str = field(compare=False)
    owner: str = field(compare=False)  # 'R': right owes; 'L': left owes
    score: float = field(compare=False, default=0.0)
    center_len: int = field(compare=False, default=0)

    @property
    def letters(self) -> int:
        return (sum(len(w) for w in self.left)
                + sum(len(w) for w in self.right) + self.center_len)


def _expand(state: COState, tries: WordTries, limit: int):
    """Yield (placement, word, new_overhang, new_owner)."""
    out = []
    if not state.overhang:
        # Free choice: grow the left edge outward; the right then owes its mirror.
        for w in tries.left_candidates("", limit):
            out.append(("L", w, w[::-1], "R"))
        return out

    if state.owner == "R":
        # Right appends words that consume the owed run from its front.
        for w in tries.left_candidates(state.overhang, limit):
            res = consume(w, state.overhang)
            if res is None:
                continue
            rem, flipped = res
            # On overrun the debt crosses to the other side, mirrored.
            out.append(("R", w, rem[::-1] if flipped else rem,
                        "L" if flipped else "R"))
    else:
        # Left prepends words that consume the owed run from its back.
        for w in tries.right_candidates(state.overhang[::-1], limit):
            res = consume_suffix(w, state.overhang)
            if res is None:
                continue
            rem, flipped = res
            out.append(("L", w, rem[::-1] if flipped else rem,
                        "R" if flipped else "L"))
    return out


def centerout_search(
    tries: WordTries,
    scorer,
    min_letters: int = 60,
    beam_width: int = 50,
    center: str = "",
    max_steps: int = 400,
    candidate_limit: int = 200,
    seed: Optional[int] = None,
    diversity: float = 0.4,
    max_overhang: int = 24,
    deadline: Optional[float] = None,
    maximize: str = "score",
    on_closed: Optional[Callable[[list[str]], None]] = None,
) -> list[str]:
    """Beam search outward from a fixed palindromic center.

    Returns the full word sequence, [] if nothing closed. `center` must read the
    same both ways; it is emitted verbatim between the two halves.

    `deadline` is a time.monotonic() value. Past it the search stops expanding
    and returns the longest palindrome it has already closed, which is what
    lets a public endpoint promise "as long as fits in N seconds".

    `maximize` picks which closure wins: "score" for the best-reading one,
    "letters" for the longest. Length only becomes the right objective when a
    deadline is doing the stopping.

    `on_closed` is called with every palindrome that closes, so a caller can
    watch the search rather than wait on it. It fires below `min_letters` too —
    the search closes on short texts almost immediately and spends the rest of
    its budget lengthening them, so the floor is exactly the part worth watching.
    It runs inside the loop, so it must be cheap, and anything it raises is the
    caller's to contain.
    """
    if center != center[::-1]:
        raise ValueError(f"center {center!r} is not itself a palindrome")

    rng = random.Random(seed)
    start = COState(sort_key=0.0, left=(), right=(), overhang="", owner="R",
                    center_len=len(center))
    beam = [start]
    best: Optional[tuple[float, list[str]]] = None

    def assemble(s: COState) -> list[str]:
        mid = [center] if center else []
        return list(s.left) + mid + list(s.right)

    for _ in range(max_steps):
        if not beam:
            break
        if deadline is not None and time.monotonic() > deadline:
            break
        if hasattr(scorer, "prepare"):
            scorer.prepare(beam)
        pool: list[COState] = []
        for state in beam:
            # Center-out closes only on an exactly empty overhang.
            if not state.overhang:
                if on_closed is not None:
                    on_closed(assemble(state))
                if state.letters >= min_letters:
                    key = (state.letters if maximize == "letters"
                           else state.score / max(1, state.letters))
                    if best is None or key > best[0]:
                        best = (key, assemble(state))
            for placement, w, new_over, new_owner in _expand(state, tries, candidate_limit):
                if len(new_over) > max_overhang:
                    continue
                if placement == "L":
                    left, right = (w,) + state.left, state.right
                else:
                    left, right = state.left, state.right + (w,)
                # Center-out grows outward: the left half is prepended to, the
                # right appended to — the opposite of the outside-in search.
                growth = "prepend" if placement == "L" else "append"
                sc = state.score + scorer.word_delta(left, right, placement, w,
                                                     growth)
                sc += rng.random() * diversity
                pool.append(COState(sort_key=-sc, left=left, right=right,
                                    overhang=new_over, owner=new_owner,
                                    score=sc, center_len=len(center)))
        if not pool:
            break
        beam = heapq.nsmallest(beam_width, pool)
        # Stop once the whole beam has comfortably cleared the floor. This is
        # what makes min_letters the length dial: the search reliably overshoots
        # it and then stops, instead of wandering into states that never close.
        if best is not None and all(s.letters > 3 * min_letters for s in beam):
            break

    return best[1] if best else []
