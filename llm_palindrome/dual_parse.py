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
from collections.abc import Callable


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


def word_residual_search(
    left_slots: tuple[tuple[str, tuple[str, ...]], ...],
    right_slots: tuple[tuple[str, tuple[str, ...]], ...],
    *, max_states: int = 50_000, max_results: int = 100,
    allow_choice: Callable[[str, str, str | None, str], bool] | None = None,
    allow_partial: Callable[[tuple[str, ...], tuple[str, ...]], bool] | None = None,
    reject_intermediate_closure: bool = False,
) -> dict:
    """Search two POS/role plans while carrying the live unmatched tape.

    A state contains both grammar cursors, both role traces, and the residual
    characters exposed by the most recently selected word.  Words are selected
    only when their side is needed; no completed sentence or reversed word is
    materialized as a search primitive.
    """
    # ``li`` advances through the left parse in reading order. ``ri`` moves
    # backwards through the right parse because the paragraph's final word is
    # the first surface exposed by the palindrome equation.
    queue = deque([(0, len(right_slots) - 1, (), (), "", "", (), ())])
    seen = set(); results = []; transitions = 0; dead_frontiers = []
    intermediate_closure_rejections = 0
    while queue and len(seen) < max_states and len(results) < max_results:
        li, ri, lw, rw_reverse, owner, residual, lroles, rroles_reverse = queue.popleft()
        key = (li, ri, lw, rw_reverse, owner, residual, lroles, rroles_reverse)
        if key in seen: continue
        seen.add(key)
        if li == len(left_slots) and ri < 0 and not residual:
            left, right = " ".join(lw), " ".join(reversed(rw_reverse))
            results.append({"left": left, "right": right,
                            "rendered": (left + " " + right).strip(),
                            "left_roles": list(lroles), "right_roles": list(reversed(rroles_reverse)),
                            "residual": "", "exact_half_equation": True})
            continue
        # If one side owns unmatched characters, only the other may advance.
        # With no debt either outer edge may open the equation.
        sides = ("right",) if owner == "left" else (("left",) if owner == "right" else ("left", "right"))
        before_transitions = transitions
        for side in sides:
            slots, index = (left_slots, li) if side == "left" else (right_slots, ri)
            if index < 0 or index >= len(slots): continue
            role, alternatives = slots[index]
            for word in alternatives:
                neighbor = (lw[-1] if side == "left" and lw else
                            rw_reverse[-1] if side == "right" and rw_reverse else None)
                if allow_choice is not None and not allow_choice(side, word, neighbor, role):
                    continue
                tape = letter_tape(word)
                if not tape: continue
                # Both streams are compared in the left-to-right orientation:
                # the right word is exposed from its last character first.
                exposed = tape if side == "left" else tape[::-1]
                if residual:
                    common = min(len(residual), len(exposed))
                    if residual[:common] != exposed[:common]: continue
                    if len(residual) > len(exposed):
                        next_owner, next_residual = owner, residual[common:]
                    elif len(exposed) > len(residual):
                        next_owner = side
                        next_residual = exposed[common:]
                    else:
                        next_owner, next_residual = "", ""
                else:
                    next_owner, next_residual = side, exposed
                next_lw = lw + ((word,) if side == "left" else ())
                next_rw_reverse = rw_reverse + ((word,) if side == "right" else ())
                if (allow_partial is not None
                        and not allow_partial(next_lw, tuple(reversed(next_rw_reverse)))):
                    continue
                next_li = li + (side == "left")
                next_ri = ri - (side == "right")
                complete_after = next_li == len(left_slots) and next_ri < 0
                if (reject_intermediate_closure and not next_residual
                        and not complete_after and next_lw and next_rw_reverse):
                    intermediate_closure_rejections += 1
                    continue
                queue.append((next_li, next_ri,
                              next_lw,
                              next_rw_reverse,
                              next_owner, next_residual,
                              lroles + ((role,) if side == "left" else ()),
                              rroles_reverse + ((role,) if side == "right" else ())))
                transitions += 1
        if transitions == before_transitions:
            left_letters = sum(len(letter_tape(word)) for word in lw)
            right_letters = sum(len(letter_tape(word)) for word in rw_reverse)
            dead_frontiers.append({
                "matched_letters": min(left_letters, right_letters),
                "left_words": list(lw),
                "right_words_reverse": list(rw_reverse),
                "owner": owner,
                "residual": residual,
                "next_left_role": left_slots[li][0] if li < len(left_slots) else None,
                "next_right_role": right_slots[ri][0] if ri >= 0 else None,
            })
            dead_frontiers.sort(key=lambda row: (-row["matched_letters"], len(row["residual"])))
            del dead_frontiers[32:]
    return {"results": results, "states": len(seen), "transitions": transitions,
            "cap_reached": bool(queue), "dead_frontiers": dead_frontiers,
            "intermediate_closure_rejections": intermediate_closure_rejections}


@dataclass(frozen=True)
class Morphology:
    """Features carried by a lexical choice, rather than repaired afterwards."""
    agreement: str = "any"
    tense: str = "any"
    determiner: str = "any"
    noun_number: str = "any"
    clitic_boundary: str = "none"


def productive_lattice(slots: list[tuple[str, list[tuple[str, Morphology]]]]) -> SurfaceLattice:
    """Build a lattice while retaining productive inflectional features."""
    lattice = SurfaceLattice()
    lattice.morphology: dict[int, Morphology] = {}
    lattice.slot_features: list[str] = []
    for role, alternatives in slots:
        lattice.slot_features.append(role)
        ids = lattice.slot(role, [surface for surface, _ in alternatives])
        for choice_id, (_, features) in zip(ids, alternatives):
            lattice.morphology[choice_id] = features
    return lattice


def _morphology_compatible(left: SurfaceLattice, right: SurfaceLattice,
                           left_ids: tuple[int, ...], right_ids: tuple[int, ...]) -> bool:
    """Check agreement/tense and boundary features at admission only."""
    lm = getattr(left, "morphology", {})
    rm = getattr(right, "morphology", {})
    # Opposing parses may use different lexical realizations, but their clause
    # features must agree.  This is deliberately independent of tape equality.
    for a, b in zip((lm.get(i) for i in left_ids), (rm.get(i) for i in right_ids)):
        if a is None or b is None:
            continue
        for field in ("agreement", "tense", "determiner", "noun_number"):
            av, bv = getattr(a, field), getattr(b, field)
            if av != "any" and bv != "any" and av != bv:
                return False
    return True


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

    def slot(self, role: str, alternatives: tuple[str, ...] | list[str]) -> tuple[int, ...]:
        """Append one required role slot to the lattice."""
        if not alternatives:
            raise ValueError(f"slot {role!r} has no alternatives")
        before, after = self.finish, self._state()
        created: list[int] = []
        for surface in alternatives:
            tape = letter_tape(surface)
            if not tape:
                raise ValueError(f"slot {role!r} contains an empty letter tape")
            choice_id = len(self.choices)
            created.append(choice_id)
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
        return tuple(created)


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
    morphology_rejections = 0
    dead_frontiers: list[dict] = []

    while queue and len(seen) < max_states and len(results) < max_results:
        left_state, right_state, left_choices, right_choices_reverse = queue.popleft()
        state_key = (left_state, right_state, left_choices, right_choices_reverse)
        if state_key in seen:
            continue
        seen.add(state_key)

        if left_state == left.finish and right_state == right.start:
            right_choices = tuple(reversed(right_choices_reverse))
            if not _morphology_compatible(left, right, left_choices, right_choices):
                morphology_rejections += 1
                continue
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
                "central_admission": {"exact_half_equation": left_letters == right_letters[::-1],
                                      "morphology_compatible": True,
                                      "post_render_repair": False},
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
        "morphology_rejections": morphology_rejections,
        "cap_reached": bool(queue),
        "dead_frontiers": dead_frontiers,
    }
