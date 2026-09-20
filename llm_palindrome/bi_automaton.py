"""Character-level intersection of independent clause automata.

The two clause inventories are never paired: the product walks a forward left
automaton and a reversed right automaton simultaneously, carrying lexical word
boundary offsets in its state.
"""
from dataclasses import dataclass
import re

def tape(s): return re.sub(r"[^a-z]", "", s.casefold())

@dataclass(frozen=True)
class Clause:
    text: str
    frame: str  # SVO or IMP

class ClauseAutomaton:
    def __init__(self, clauses, reverse=False):
        self.clauses = tuple(clauses)
        self.reverse = reverse
        self._tapes = tuple(tape(c.text)[::-1] if reverse else tape(c.text) for c in self.clauses)

def intersect(left, right, limit=100):
    """Online NFA product; returns exact pairs, never repairs a mismatch."""
    out=[]
    # state=(left clause, right clause, character offset, boundary offsets)
    for li, lt in enumerate(left._tapes):
        for ri, rt in enumerate(right._tapes):
            if left.clauses[li].frame != right.clauses[ri].frame or len(lt) != len(rt):
                continue
            bl=tuple(i for i,c in enumerate(left.clauses[li].text) if c.isspace())
            br=tuple(i for i,c in enumerate(right.clauses[ri].text) if c.isspace())
            if not bl or not br or bl == br:
                continue
            ok=True
            for pos,(a,b) in enumerate(zip(lt,rt)):
                if a != b: ok=False; break
            if ok:
                out.append({"left":left.clauses[li].text,"right":right.clauses[ri].text,
                            "frame":left.clauses[li].frame,"boundary_offsets":(bl,br),
                            "exact":True,"repair":False})
                if len(out)>=limit:return out
    return out
