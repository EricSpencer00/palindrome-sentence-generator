"""Exact-by-construction role graph for semordnilap phrase edges.

This lane is intentionally not a center-out search or a repair pass.  A graph
edge owns both sides of one mirrored lexical unit; paths are grown by choosing
the next edge from the current semantic role.  Consequently every emitted
path is exact before rendering.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence
import re

_NONLETTERS = re.compile(r"[^a-z]")


def tape(text: str) -> str:
    return _NONLETTERS.sub("", text.casefold())


@dataclass(frozen=True)
class RoleEdge:
    left: str
    right: str
    role: str
    next_role: str
    provenance: str = "authored"
    seam_shift: bool = True

    def exact(self) -> bool:
        # A one-token semordnilap is a catalogue shortcut, not a sentence
        # construction.  Require a lexical boundary on each side so every
        # admitted edge crosses a word seam during composition.
        return (self.seam_shift and len(self.left.split()) >= 2 and len(self.right.split()) >= 2
                and tape(self.left) == tape(self.right)[::-1])


@dataclass(frozen=True)
class GraphPath:
    edges: tuple[RoleEdge, ...]

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(e.role for e in self.edges)

    def render(self) -> str:
        # The right half is reversed in edge order, as well as character tape.
        return " ".join([*(e.left for e in self.edges), *(e.right for e in reversed(self.edges))])

    def exact(self) -> bool:
        return tape(self.render()) == tape(self.render())[::-1]

    def admissible_frame(self) -> bool:
        """Require a complete SVO or imperative role chain on both mirrors."""
        roles = self.roles
        return roles in (("subject", "verb", "object"), ("verb", "object"))

    def audit(self) -> dict[str, bool]:
        units = [tape(e.left) for e in self.edges]
        return {
            "exact": self.exact(),
            "complete_frame": self.admissible_frame(),
            "boundary_shift": any(e.seam_shift for e in self.edges),
            "no_repeated_units": len(units) == len(set(units)),
            "no_repeated_edges": len(self.edges) == len({(e.left, e.right) for e in self.edges}),
        }


def build_graph(edges: Iterable[RoleEdge]) -> Mapping[str, tuple[RoleEdge, ...]]:
    """Index only exact, cross-word edges; malformed edges never enter."""
    graph: dict[str, list[RoleEdge]] = {}
    for edge in edges:
        if edge.exact():
            graph.setdefault(edge.role, []).append(edge)
    return {role: tuple(items) for role, items in graph.items()}


def search_paths(
    graph: Mapping[str, Sequence[RoleEdge]],
    start_roles: Iterable[str],
    max_edges: int = 4,
    min_letters: int = 1,
    limit: int = 100,
) -> list[GraphPath]:
    """Enumerate role-compatible paths, growing both sides online.

    No candidate is repaired or filtered after rendering: each extension is
    checked as an edge and the returned path is exact by construction.
    """
    out: list[GraphPath] = []

    def walk(role: str, path: tuple[RoleEdge, ...]) -> None:
        if len(out) >= limit:
            return
        if path and sum(len(tape(e.left)) for e in path) >= min_letters:
            out.append(GraphPath(path))
        if len(path) >= max_edges:
            return
        for edge in graph.get(role, ()):
            walk(edge.next_role, path + (edge,))

    for role in start_roles:
        walk(role, ())
    return out


def novelty_preflight(paths: Iterable[GraphPath], catalogue: Iterable[str]) -> list[GraphPath]:
    """Remove normalized renders already present in a catalogue."""
    known = {tape(item) for item in catalogue}
    return [p for p in paths if tape(p.render()) not in known]
