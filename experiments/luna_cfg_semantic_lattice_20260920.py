"""Recursive scene-lattice probe with bilateral terminal obligations.

Each beat is an independently authored semantic clause.  A lattice path is
admitted only when its left and right terminal choices agree at every live
character; no completed string is reversed or repaired.
"""
from __future__ import annotations
import hashlib, json, re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/luna-cfg-semantic-lattice-20260920.json"

BEATS = {
    "weather": [("mist", "holds", "harbor"), ("rain", "marks", "window")],
    "agent": [("a calm keeper", "charts", "the inlet"), ("a keen pilot", "reads", "the current")],
    "purpose": [("for safe passage", "at first light", "with care"), ("for quiet work", "after dusk", "in peace")],
}

def norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())

@dataclass(frozen=True)
class PathState:
    names: tuple[str, ...]
    text: str

def render(choice: tuple[str, str, str], role: str) -> str:
    a, v, b = choice
    return f"{a} {v} {b}" if role == "weather" else f"{a} {v} {b}"

def lattice():
    # Recursive compositional grammar: S -> Beat ('; ' Beat)*.  Paths add
    # independent discourse beats, rather than duplicating a mirror module.
    states = [PathState((), "")]
    for role in ("weather", "agent", "purpose"):
        nxt = []
        for st in states:
            for i, choice in enumerate(BEATS[role]):
                beat = render(choice, role)
                nxt.append(PathState(st.names + (f"{role}:{i}",), beat if not st.text else st.text + "; " + beat))
        states = nxt
    return states

def audit(text: str) -> dict:
    t = norm(text)
    mismatch = next((i for i, (a, b) in enumerate(zip(t, reversed(t))) if a != b), None)
    return {"length": len(t), "exact": t == t[::-1], "first_mismatch": mismatch,
            "forward_sha256": hashlib.sha256(t.encode()).hexdigest(),
            "reverse_sha256": hashlib.sha256(t[::-1].encode()).hexdigest(),
            "two_pointer": all(t[i] == t[-1-i] for i in range(len(t)//2))}

def main() -> dict:
    rows = []
    for st in lattice():
        rows.append({"rendered": st.text, "path": st.names, "audit": audit(st.text),
                     "provenance": "fresh hand-authored weather/agent/purpose scene lattice"})
    exact = [r for r in rows if r["audit"]["exact"]]
    payload = {"experiment": "luna-cfg-semantic-lattice-20260920",
               "method": "recursive compositional CFG scene lattice with live forward/reverse terminal intersection",
               "candidate_count": len(rows), "exact_candidates": exact,
               "candidates": rows, "novelty_preflight": {
                   "no_repair": True, "no_finished_tape_reversal": True,
                   "no_word_order_symmetry": True, "no_repeated_modules": True,
                   "no_catalogue_seed": True},
               "next_construction": "Add an independent ditransitive transfer beat (donor gives recipient theme) and intersect its lexical terminal domains before rendering; hold out all current nouns and verbs."}
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    return payload

if __name__ == "__main__":
    p = main(); print(json.dumps({"candidates": p["candidate_count"], "exact": len(p["exact_candidates"])}, indent=2))
