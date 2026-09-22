"""Exact intersection of two differently segmented English surface lattices.

An exact letter palindrome always has a left character tape ``H`` and the
reverse tape ``H[::-1]`` on its right.  The useful search problem is therefore
not to hide that equality; it is to require *independent linguistic parses* of
the two tapes.  This module intersects a left grammar with a right grammar
backwards, one character at a time, before either complete surface is chosen.

The lattices retain semantic role labels and lexical choice identities.  That
lets paragraph experiments express an A/B discourse on the left and a B'/A'
discourse on the right without requiring any sentence or word to be a reversed
unit.  Different word-boundary signatures are reported explicitly.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import re


def letter_tape(text: str) -> str:
    """Return the lowercase ASCII letter tape used by exact validation."""
    return "".join(re.findall(r"[a-z]", text.casefold()))


def word_boundaries(text: str) -> tuple[int, ...]:
    """Return internal word boundaries measured on the normalized tape."""
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", text.casefold())
    offsets: list[int] = []
    consumed = 0
    for word in words[:-1]:
        consumed += len(letter_tape(word))
        offsets.append(consumed)
    return tuple(offsets)


@dataclass(frozen=True)
class Choice:
    id: int
    role: str
    surface: str
    tape: str


@dataclass(frozen=True)
class Edge:
    source: int
    target: int
    char: str
    choice_id: int
    offset: int
    choice_length: int


class SurfaceLattice:
    """A linear sequence of semantic slots with lexical alternatives."""

    def __init__(self) -> None:
        self.start = 0
        self.finish = 0
        self._next_state = 1
        self.choices: dict[int, Choice] = {}
        self.edges: list[Edge] = []
        self.outgoing: dict[int, list[int]] = defaultdict(list)
        self.incoming: dict[int, list[int]] = defaultdict(list)

    def _state(self) -> int:
        state = self._next_state
        self._next_state += 1
        return state

    def slot(self, role: str, alternatives: tuple[str, ...] | list[str]) -> None:
        """Append one required role slot to the lattice."""
        if not alternatives:
            raise ValueError(f"slot {role!r} has no alternatives")
        before, after = self.finish, self._state()
        for surface in alternatives:
            tape = letter_tape(surface)
            if not tape:
                raise ValueError(f"slot {role!r} contains an empty letter tape")
            choice_id = len(self.choices)
            self.choices[choice_id] = Choice(choice_id, role, surface, tape)
            current = before
            for offset, char in enumerate(tape):
                following = after if offset == len(tape) - 1 else self._state()
                edge = Edge(current, following, char, choice_id, offset, len(tape))
                edge_id = len(self.edges)
                self.edges.append(edge)
                self.outgoing[current].append(edge_id)
                self.incoming[following].append(edge_id)
                current = following
        self.finish = after


def _append_choice(path: tuple[int, ...], edge: Edge, *, backward: bool) -> tuple[int, ...]:
    entering_choice = edge.offset == (edge.choice_length - 1 if backward else 0)
    return path + (edge.choice_id,) if entering_choice else path


def intersect_surfaces(
    left: SurfaceLattice,
    right: SurfaceLattice,
    *,
    max_states: int = 250_000,
    max_results: int = 100,
) -> dict:
    """Intersect ``left`` with the character reversal of ``right``.

    Search starts at the left grammar's first character and the right grammar's
    final character.  Only equal character transitions enter the queue, so an
    accepting state is exact by construction.  Lexical choices are part of the
    state key; different segmentations that reconverge at a slot boundary are
    not silently collapsed.
    """
    start = (left.start, right.finish, (), ())
    queue = deque([start])
    seen: set[tuple[int, int, tuple[int, ...], tuple[int, ...]]] = set()
    results: list[dict] = []
    transitions = 0
    dead_frontiers: list[dict] = []

    while queue and len(seen) < max_states and len(results) < max_results:
        left_state, right_state, left_choices, right_choices_reverse = queue.popleft()
        state_key = (left_state, right_state, left_choices, right_choices_reverse)
        if state_key in seen:
            continue
        seen.add(state_key)

        if left_state == left.finish and right_state == right.start:
            right_choices = tuple(reversed(right_choices_reverse))
            left_surfaces = [left.choices[c].surface for c in left_choices]
            right_surfaces = [right.choices[c].surface for c in right_choices]
            left_text = " ".join(left_surfaces)
            right_text = " ".join(right_surfaces)
            left_letters, right_letters = letter_tape(left_text), letter_tape(right_text)
            left_breaks = word_boundaries(left_text)
            reflected_right_breaks = tuple(
                sorted(len(right_letters) - boundary for boundary in word_boundaries(right_text))
            )
            results.append({
                "left": left_text,
                "right": right_text,
                "rendered": f"{left_text} {right_text}",
                "letters_per_half": len(left_letters),
                "exact_half_equation": left_letters == right_letters[::-1],
                "left_roles": [left.choices[c].role for c in left_choices],
                "right_roles": [right.choices[c].role for c in right_choices],
                "left_choice_ids": list(left_choices),
                "right_choice_ids": list(right_choices),
                "left_word_boundaries": list(left_breaks),
                "reflected_right_word_boundaries": list(reflected_right_breaks),
                "different_word_segmentation": left_breaks != reflected_right_breaks,
            })
            continue

        left_by_char: dict[str, list[int]] = defaultdict(list)
        right_by_char: dict[str, list[int]] = defaultdict(list)
        for edge_id in left.outgoing.get(left_state, ()):
            left_by_char[left.edges[edge_id].char].append(edge_id)
        for edge_id in right.incoming.get(right_state, ()):
            right_by_char[right.edges[edge_id].char].append(edge_id)
        common = left_by_char.keys() & right_by_char.keys()
        if not common:
            dead_frontiers.append({
                "matched_choices_left": list(left_choices),
                "matched_choices_right_reverse": list(right_choices_reverse),
                "left_next": sorted(left_by_char),
                "right_next": sorted(right_by_char),
            })
            dead_frontiers = dead_frontiers[-32:]
            continue
        for char in sorted(common):
            for left_edge_id in left_by_char[char]:
                left_edge = left.edges[left_edge_id]
                next_left_choices = _append_choice(left_choices, left_edge, backward=False)
                for right_edge_id in right_by_char[char]:
                    right_edge = right.edges[right_edge_id]
                    next_right_choices = _append_choice(
                        right_choices_reverse, right_edge, backward=True
                    )
                    queue.append((
                        left_edge.target,
                        right_edge.source,
                        next_left_choices,
                        next_right_choices,
                    ))
                    transitions += 1

    return {
        "results": results,
        "states": len(seen),
        "transitions": transitions,
        "cap_reached": bool(queue),
        "dead_frontiers": dead_frontiers,
    }
