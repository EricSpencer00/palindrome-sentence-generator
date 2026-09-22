"""Finite recursive dual-grammar residual product.

This module is deliberately independent of the prose inventories.  Grammars
are finite maps of states to word-labelled edges; recursion is represented by
edges back to an earlier state.  The left grammar is traversed in reading
order.  The right grammar is supplied and traversed from the paragraph's
outside edge toward its center; each right word is therefore exposed in
reverse character order and the selected right words are reversed for normal
rendering.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import re


def tape(s: str) -> str:
    return "".join(re.findall(r"[a-z]", s.casefold()))


@dataclass(frozen=True)
class Edge:
    target: str
    word: str
    phase: str = ""
    role: str = ""

    @property
    def chars(self) -> str:
        return tape(self.word)


@dataclass(frozen=True)
class State:
    left: str
    right: str
    owner: str = ""
    residual: str = ""
    phase: str = ""


@dataclass(frozen=True)
class Witness:
    left_words: tuple[str, ...]
    right_words: tuple[str, ...]
    phase_trace: tuple[str, ...]
    left_boundaries: tuple[int, ...]
    reflected_right_boundaries: tuple[int, ...]


@dataclass(frozen=True)
class PumpCycle:
    """A reachable/coaccessible nonempty-debt cycle and its witness paths."""
    states: tuple[State, ...]
    prefix: tuple[tuple[str, Edge], ...]
    cycle: tuple[tuple[str, Edge], ...]
    suffix: tuple[tuple[str, Edge], ...]


@dataclass
class Report:
    witnesses: list[Witness]
    reachable: set[State]
    coaccessible: set[State]
    pumpable_cycles: list[PumpCycle]
    intermediate_empty_closures: int


def _step(s: State, side: str, edge: Edge) -> tuple[State, bool] | None:
    exposed = edge.chars if side == "L" else edge.chars[::-1]
    if not exposed:
        return None
    if s.residual:
        n = min(len(s.residual), len(exposed))
        if s.residual[:n] != exposed[:n]:
            return None
        a, b = s.residual[n:], exposed[n:]
        owner = s.owner if a else (side if b else "")
        residual = a or b
    else:
        owner, residual = side, exposed
    if side == "L":
        ns = State(edge.target, s.right, owner, residual, edge.phase or s.phase)
    else:
        ns = State(s.left, edge.target, owner, residual, edge.phase or s.phase)
    return ns, not residual


def _witness_from_steps(steps: tuple[tuple[str, Edge], ...]) -> Witness:
    lw = tuple(edge.word for side, edge in steps if side == "L")
    rw_outside_in = tuple(edge.word for side, edge in steps if side == "R")
    rw = tuple(reversed(rw_outside_in))
    phases = tuple(edge.phase for _side, edge in steps if edge.phase)
    left_offsets, consumed = [], 0
    for word in lw[:-1]:
        consumed += len(tape(word)); left_offsets.append(consumed)
    right_offsets, consumed = [], 0
    right_letters = sum(len(tape(word)) for word in rw)
    for word in rw[:-1]:
        consumed += len(tape(word)); right_offsets.append(right_letters - consumed)
    return Witness(lw, rw, phases, tuple(left_offsets),
                   tuple(sorted(right_offsets)))


def materialize_pump(pump: PumpCycle, repetitions: int) -> Witness:
    """Replay a certified cycle ``repetitions`` times into one exact witness."""
    if repetitions < 0:
        raise ValueError("repetitions must be nonnegative")
    return _witness_from_steps(pump.prefix + pump.cycle * repetitions + pump.suffix)


def search(left: dict[str, tuple[Edge, ...]], right: dict[str, tuple[Edge, ...]], *,
           start_left: str = "S", start_right: str = "S", terminals=("F",),
           max_states: int = 20_000, max_results: int = 20,
           reject_intermediate_closure: bool = True,
           allowed_phases: dict[str, tuple[str, ...]] | None = None) -> Report:
    start = State(start_left, start_right)
    q = deque([start]); reachable = {start}; pred: dict[State, tuple[State, str, Edge]] = {}
    graph: dict[State, set[State]] = defaultdict(set)
    labels: dict[tuple[State, State], list[tuple[str, Edge]]] = defaultdict(list)
    empty = 0
    while q and len(reachable) <= max_states:
        s = q.popleft()
        # ``owner`` names the side that has emitted unmatched characters, so
        # only the opposite side can consume the live debt.
        sides = (("R",) if s.owner == "L" else
                 (("L",) if s.owner == "R" else ("L", "R")))
        for side in sides:
            edges = left.get(s.left, ()) if side == "L" else right.get(s.right, ())
            for e in sorted(edges, key=lambda x: (x.target, x.word, x.phase, x.role)):
                got = _step(s, side, e)
                if got is None: continue
                ns, closed = got
                if allowed_phases is not None and s.phase and e.phase and e.phase not in allowed_phases.get(s.phase, ()):
                    continue
                complete = ns.left in terminals and ns.right in terminals
                if closed and reject_intermediate_closure and not complete:
                    empty += 1; continue
                graph[s].add(ns)
                labels[(s, ns)].append((side, e))
                if ns not in reachable:
                    reachable.add(ns); pred[ns] = (s, side, e); q.append(ns)
    accepting = {s for s in reachable if s.left in terminals and s.right in terminals and not s.residual}
    witnesses = []
    for end in sorted(accepting, key=repr):
        path=[]; cur=end
        while cur != start:
            p, side, e = pred[cur]; path.append((side, e)); cur=p
        path.reverse()
        witnesses.append(_witness_from_steps(tuple(path)))
    reverse: dict[State, set[State]] = defaultdict(set)
    for a, bs in graph.items():
        for b in bs: reverse[b].add(a)
    co = set(accepting); q = deque(accepting)
    while q:
        x = q.popleft()
        for p in reverse[x]:
            if p not in co: co.add(p); q.append(p)
    # Tarjan SCC, restricted to reachable/coaccessible states.
    nodes = reachable & co; index = 0; stack=[]; on=set(); ix={}; low={}; cycle_states=[]

    def concrete_cycle(component: set[State]) -> tuple[State, ...] | None:
        """Return one deterministic directed cycle, including its repeated start."""
        for anchor in sorted(component, key=repr):
            for successor in sorted(graph[anchor] & component, key=repr):
                if successor == anchor:
                    return (anchor, anchor)
                todo = deque([(successor, (anchor, successor))])
                visited = {successor}
                while todo:
                    node, path = todo.popleft()
                    for following in sorted(graph[node] & component, key=repr):
                        if following == anchor:
                            return path + (anchor,)
                        if following not in visited:
                            visited.add(following)
                            todo.append((following, path + (following,)))
        return None

    def visit(v):
        nonlocal index
        ix[v]=low[v]=index; index+=1; stack.append(v); on.add(v)
        for w in graph[v] & nodes:
            if w not in ix: visit(w); low[v]=min(low[v],low[w])
            elif w in on: low[v]=min(low[v],ix[w])
        if low[v] == ix[v]:
            comp=[]
            while True:
                w=stack.pop(); on.remove(w); comp.append(w)
                if w==v: break
            component = set(comp)
            cycle = concrete_cycle(component)
            # A pump may never pass through an empty residual state: that
            # would be a smaller, independently closed palindrome boundary.
            if cycle is not None and all(x.residual for x in cycle):
                cycle_states.append(cycle)
    for v in sorted(nodes, key=repr):
        if v not in ix: visit(v)
    def prefix_steps(entry: State) -> tuple[tuple[str, Edge], ...]:
        path: list[tuple[str, Edge]] = []
        current = entry
        while current != start:
            previous, side, edge = pred[current]
            path.append((side, edge)); current = previous
        return tuple(reversed(path))

    def suffix_states(entry: State) -> tuple[State, ...]:
        todo = deque([(entry, (entry,))]); visited = {entry}
        while todo:
            current, path = todo.popleft()
            if current in accepting:
                return path
            for following in sorted(graph[current] & co, key=repr):
                if following not in visited:
                    visited.add(following)
                    todo.append((following, path + (following,)))
        raise AssertionError("coaccessible state lacks accepting path")

    def steps_for(states: tuple[State, ...]) -> tuple[tuple[str, Edge], ...]:
        chosen = []
        for source, target in zip(states, states[1:]):
            chosen.append(sorted(labels[(source, target)],
                                 key=lambda item: (item[0], item[1].target,
                                                   item[1].word, item[1].phase,
                                                   item[1].role))[0])
        return tuple(chosen)

    pumps = []
    for states in cycle_states:
        entry = states[0]
        suffix = suffix_states(entry)
        pumps.append(PumpCycle(states, prefix_steps(entry), steps_for(states),
                               steps_for(suffix)))
    return Report(witnesses[:max_results], reachable, co, pumps, empty)
