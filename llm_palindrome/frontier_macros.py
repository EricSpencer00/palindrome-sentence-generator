"""Bounded exact frontier moves with an editable English completion witness.

The witness orders proposals only. Accepted letters always come from the
existing exact lexical lattice; an English completion is never a candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Callable

from .admission import (
    REPEATABLE_FUNCTION_WORDS, has_distinct_content_words,
    has_only_ordinary_short_words, normalize_letters, tokenize,
)
from .search import State, WordTries, _expand


@dataclass(frozen=True)
class MacroMove:
    state: State
    trace: tuple[tuple[str, str], ...]
    witness_matches: int
    rank_score: float

    @property
    def move_id(self) -> str:
        payload = [self.state.left, self.state.right, self.trace]
        return sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()[:16]

    def menu_item(self) -> dict:
        return {
            "id": self.move_id,
            "prefix": " ".join(self.state.left),
            "suffix": " ".join(self.state.right),
            "fixed_letters": self.state.letters,
            "added_words": sum(len(unit.split()) for _, unit in self.trace),
        }


def state_from_anchors(prefix: str, suffix: str) -> State:
    left, right = tokenize(prefix), tokenize(suffix)
    if not left or not right:
        raise ValueError("both_anchors_required")
    a, b = normalize_letters(prefix), normalize_letters(suffix)[::-1]
    if a.startswith(b):
        overhang, side = a[len(b):], "L"
    elif b.startswith(a):
        overhang, side = b[len(a):], "R"
    else:
        raise ValueError("incompatible_anchor_letters")
    return State(0, left, right, overhang, side)


def witness_preserves_anchors(state: State, text: str, *, allow_closed: bool = False) -> bool:
    words = tokenize(text)
    left = tuple(w for unit in state.left for w in unit.split())
    right = tuple(w for unit in state.right for w in unit.split())
    minimum = len(left) + len(right) + (0 if allow_closed else 1)
    return len(words) >= minimum and words[:len(left)] == left and words[-len(right):] == right


def _matching_edges(state: State, witness: tuple[str, ...]) -> int:
    left = tuple(w for unit in state.left for w in unit.split())
    right = tuple(w for unit in state.right for w in unit.split())
    count = 0
    for a, b in zip(left, witness):
        if a != b:
            break
        count += 1
    for a, b in zip(reversed(right), reversed(witness)):
        if a != b:
            break
        count += 1
    return count


def enumerate_macros(
    parent: State, tries: WordTries, *, witness: str = "",
    min_words: int = 2, max_words: int = 4, frontier_width: int = 256,
    menu_size: int = 16, candidate_limit: int = 1200,
    score_word: Callable[[State, str, str], float] | None = None,
) -> list[MacroMove]:
    """Enumerate legal multiword moves without narrowing the lexical grammar.

    Round-robin selection by first transition keeps the cheap proposal ranking
    from spending the whole model menu on near-identical siblings. All depths
    in the requested band are eligible; one-sided moves remain available when
    paying a long overhang requires several words before the other side grows.
    """
    if not 1 <= min_words <= max_words or min(frontier_width, menu_size, candidate_limit) < 1:
        raise ValueError("invalid_macro_budget")
    witness_words = tokenize(witness)
    initial_matches = _matching_edges(parent, witness_words)
    frontier = [MacroMove(parent, (), 0, 0.0)]
    leaves: dict[tuple, MacroMove] = {}

    def diverse(rows: list[MacroMove], limit: int) -> list[MacroMove]:
        buckets: dict[tuple, list[MacroMove]] = {}
        for row in sorted(rows, key=lambda x: (-x.witness_matches, -x.rank_score / max(1, len(x.trace)), x.move_id)):
            buckets.setdefault((len(row.trace), row.trace[0]), []).append(row)
        # Rotate depth bands as well as first moves. Raw accumulated log
        # probabilities otherwise make almost every menu item the shortest
        # possible move, removing the intended lookahead.
        depths = sorted({key[0] for key in buckets})
        ordered_keys = []
        by_depth = {depth: [key for key in buckets if key[0] == depth] for depth in depths}
        while any(by_depth.values()):
            for depth in depths:
                if by_depth[depth]:
                    ordered_keys.append(by_depth[depth].pop(0))
        buckets = {key: buckets[key] for key in ordered_keys}
        out = []
        while buckets and len(out) < limit:
            for key in list(buckets):
                out.append(buckets[key].pop(0))
                if not buckets[key]:
                    del buckets[key]
                if len(out) == limit:
                    break
        return out

    for _ in range(max_words):
        children = []
        for row in frontier:
            for side, unit, overhang, next_side in _expand(row.state, tries, candidate_limit):
                trace = row.trace + ((side, unit),)
                count = sum(len(word.split()) for _, word in trace)
                if count > max_words:
                    continue
                left = row.state.left + (unit,) if side == "L" else row.state.left
                right = (unit,) + row.state.right if side == "R" else row.state.right
                words = tuple(word for block in left + right for word in block.split())
                if not has_distinct_content_words(words) or not has_only_ordinary_short_words(words):
                    continue
                if any(w == w[::-1] and w not in REPEATABLE_FUNCTION_WORDS for w in words):
                    continue
                state = State(0, left, right, overhang, next_side)
                delta = score_word(state, side, unit) if score_word else 0.0
                child = MacroMove(state, trace, _matching_edges(state, witness_words) - initial_matches, row.rank_score + delta)
                children.append(child)
                if count >= min_words:
                    key = (left, right, overhang, next_side)
                    prior = leaves.get(key)
                    if prior is None or child.rank_score > prior.rank_score:
                        leaves[key] = child
        frontier = diverse(children, frontier_width)
        if not frontier:
            break
    return diverse(list(leaves.values()), menu_size)
