"""Length-indexed exact palindrome construction.

This module is the constructive core for arbitrary requested lengths.  It
does not score a completed string and call that readability: a state is legal
only when its two independently segmented tapes cancel exactly.  The search
is length-bounded, memoized, and can be run in a strict lexical mode or with
an explicit one-letter fallback for lengths the supplied lexicon cannot tile.

The fallback is useful for testing the length machinery only.  It is never a
reader-study candidate; callers must run the shared admission gate and then
the blinded-reader gate before making an English claim.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Callable, Iterable, Optional, Sequence

from .admission import mechanical_admission_checks, normalize_letters
from .centerout import COState, _expand
from .search import WordTries, unit_letters
from .validator import is_palindrome


@dataclass(frozen=True)
class ExactResult:
    status: str
    target_letters: int
    text: str | None = None
    letters: int = 0
    nodes: int = 0
    center: str = ""
    fallback: bool = False
    admission: dict[str, bool] | None = None

    def as_dict(self) -> dict:
        out = {
            "status": self.status,
            "target_letters": self.target_letters,
            "letters": self.letters,
            "nodes": self.nodes,
            "center": self.center,
            "fallback": self.fallback,
            # No automatic constructor result is reader evidence.  A later
            # frozen package may promote a strict row only after intact-prose
            # and shuffled-control human ratings.
            "reader_candidate": False,
        }
        if self.text is not None:
            out["text"] = self.text
            out["independent_exact_validation"] = (
                is_palindrome(self.text)
                and len(normalize_letters(self.text)) == self.target_letters
            )
            out["admission"] = self.admission or mechanical_admission_checks(
                self.text, min_letters=1, max_letters=max(1, self.target_letters)
            )
        return out


def _assemble(state: COState, center: str) -> str:
    units = list(state.left)
    if center:
        units.append(center)
    units.extend(state.right)
    return " ".join(units)


def _search_target(
    tries: WordTries,
    target: int,
    center: str,
    *,
    max_nodes: int,
    max_overhang: int,
    candidate_limit: int,
    require_admitted: bool,
    allow_state: Optional[Callable[[tuple[str, ...], tuple[str, ...]], bool]] = None,
) -> ExactResult:
    """Best-first exact-length search for one fixed centre.

    The priority is the remaining letter budget plus the live debt.  Unlike a
    beam, a state is never discarded merely because its local words score
    poorly; memoization only removes the same lexical state reached twice.
    """
    center_len = len(unit_letters(center))
    if center_len > target or (center_len and center_len % 2 != target % 2):
        return ExactResult("unreachable_center_parity", target, center=center)
    start = COState(0.0, (), (), "", "R", center_len=center_len)
    heap: list[tuple[int, int, COState]] = [(target - center_len, 0, start)]
    seen: set[tuple[tuple[str, ...], tuple[str, ...], str, str]] = set()
    serial = 0
    nodes = 0
    while heap and nodes < max_nodes:
        _priority, _serial, state = heapq.heappop(heap)
        nodes += 1
        key = (state.left, state.right, state.overhang, state.owner)
        if key in seen:
            continue
        seen.add(key)
        if not state.overhang and state.letters == target:
            text = _assemble(state, center)
            tape = normalize_letters(text)
            if tape == tape[::-1] and len(tape) == target:
                admission = mechanical_admission_checks(
                    text, min_letters=1, max_letters=max(1, target)
                )
                if require_admitted and not all(admission.values()):
                    continue
                return ExactResult("exact", target, text, target, nodes,
                                   center, False, admission)
        if state.letters >= target:
            continue
        for placement, word, debt, owner in _expand(state, tries, limit=candidate_limit):
            if len(debt) > max_overhang:
                continue
            left = ((word,) + state.left if placement == "L" else state.left)
            right = (state.right if placement == "L" else state.right + (word,))
            if allow_state is not None and not allow_state(left, right):
                continue
            nxt = COState(0.0, left, right, debt, owner, center_len=center_len)
            if nxt.letters > target:
                continue
            serial += 1
            # Prefer states nearest the requested length, with shorter debts
            # first so closures are found quickly without pruning alternatives.
            priority = (target - nxt.letters) + 2 * len(debt)
            heapq.heappush(heap, (priority, serial, nxt))
    status = "node_budget" if nodes >= max_nodes else "no_construction"
    return ExactResult(status, target, nodes=nodes, center=center)


def construct_exact(
    target_letters: int,
    vocabulary: Sequence[str],
    *,
    centers: Sequence[str] = ("", "a", "i"),
    max_nodes: int = 200_000,
    max_overhang: int = 24,
    candidate_limit: int = 256,
    require_admitted: bool = False,
    fallback_one_letters: bool = False,
    allow_state: Optional[Callable[[tuple[str, ...], tuple[str, ...]], bool]] = None,
) -> dict:
    """Construct an exact palindrome at ``target_letters`` if reachable.

    ``vocabulary`` is shared by both independently segmented sides.  A
    caller may add a broader lexicon or phrase units without changing the
    algorithm.  If strict lexical search cannot tile a requested length,
    ``fallback_one_letters`` appends ordinary one-letter tokens solely to
    demonstrate that the length engine itself is total; the result is marked
    ``fallback`` and is not readable-output evidence.
    """
    if target_letters < 1:
        return ExactResult("invalid_target", target_letters).as_dict()
    words = list(dict.fromkeys(w.lower() for w in vocabulary if unit_letters(w).isalpha()))
    tries = WordTries(words)
    attempts: list[ExactResult] = []
    for center in centers:
        if normalize_letters(center) != normalize_letters(center)[::-1]:
            continue
        attempts.append(_search_target(
            tries, target_letters, center, max_nodes=max_nodes,
            max_overhang=max_overhang, candidate_limit=candidate_limit,
            require_admitted=require_admitted,
            allow_state=allow_state
        ))
        if attempts[-1].status == "exact":
            return attempts[-1].as_dict()
    if fallback_one_letters:
        # The fallback is deliberately explicit and auditable.  It gives the
        # exact-length core a total construction for both parities, but its
        # output is not admitted as English prose and is never reader-facing.
        center = "a" if target_letters % 2 else ""
        half = (target_letters - len(center)) // 2
        text = " ".join(["a"] * half + ([center] if center else []) + ["a"] * half)
        tape = normalize_letters(text)
        row = ExactResult(
            "exact_fallback", target_letters, text, len(tape),
            sum(r.nodes for r in attempts), center, True,
            mechanical_admission_checks(text, min_letters=1,
                                        max_letters=max(1, target_letters)),
        )
        return row.as_dict()
    # Preserve the most useful diagnostic from the attempted centres.
    best = max(attempts, key=lambda r: r.nodes, default=ExactResult("no_construction", target_letters))
    return best.as_dict()


def construct_many(targets: Iterable[int], vocabulary: Sequence[str], **kwargs) -> list[dict]:
    """Run a reproducible length sweep with the same lexical inventory."""
    return [construct_exact(int(target), vocabulary, **kwargs) for target in targets]
