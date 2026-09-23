"""Independent word-pair tape graph.

Each edge contributes one ordinary phrase to the left tape and a separately
chosen phrase to the right tape.  Search is character-ledger based: a node is
extendable only when its exposed residual can still be consumed by an edge.
No sentence is reversed or re-ordered by this module.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import re

WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")

def tape(s: str) -> str:
    return ''.join(WORD.findall(s.lower())).replace("'", '')

@dataclass(frozen=True)
class WordPair:
    left: str
    right: str
    pos: tuple[str, ...] = ()
    valency: str = "clause"

    @property
    def left_tape(self): return tape(self.left)
    @property
    def right_tape(self): return tape(self.right)

def _compatible(a: WordPair, b: WordPair) -> bool:
    # Distinct ordinary phrases; preserve clause-level valency at every join.
    return (a != b and a.valency == b.valency and
            not (set(a.pos) & {"proper", "initialism"}) and
            not (set(b.pos) & {"proper", "initialism"}))

def extendable(left: str, right: str) -> bool:
    """Return whether the exposed tapes have a compatible overlap.

    ``right`` is stored in reading order, so its exposed outer characters are
    encountered as ``right[::-1]`` when checking a palindrome from the edges
    inward.  Characters in the shared prefix are already fixed and must agree;
    any unmatched suffix is still live center debt that later edges may fill.
    """
    n = min(len(left), len(right))
    return left[:n] == right[::-1][:n]

def search(pairs: list[WordPair], max_depth: int = 8) -> dict:
    """Enumerate distinct pair paths, retaining longest prose and closures."""
    closures, best = [], []
    calls = 0
    def visit(path, left, right, used):
        nonlocal calls, best
        calls += 1
        if len(path) > len(best): best = path[:]
        if left == right[::-1] and path:
            closures.append(path[:])
        if len(path) >= max_depth: return
        for i, p in enumerate(pairs):
            if i in used or (path and not _compatible(path[-1], p)): continue
            nl, nr = left + p.left_tape, p.right_tape + right
            if extendable(nl, nr): visit(path + [p], nl, nr, used | {i})
    for i, p in enumerate(pairs): visit([p], p.left_tape, p.right_tape, {i})
    return {"closures": closures, "best": best, "expansions": calls}

def render(path: list[WordPair]) -> str:
    return ' '.join(p.left for p in path) + ' ' + ' '.join(p.right for p in reversed(path))

def fingerprint(path: list[WordPair]) -> str:
    return hashlib.sha256(tape(render(path)).encode()).hexdigest()
