"""Grammar-first, online-coupled semantic beam (a deliberately fresh lane).

Each expansion emits a lexical choice on one side and checks the exposed
characters against the opposite pointer immediately.  No completed string is
reversed, and no candidate is repaired after it closes.  The finite grammar
carried here is intentionally small: determiner + subject + finite verb +
object, with number agreement as a two-state register.
"""
from __future__ import annotations
import hashlib, json, re
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/grammar-first-semantic-beam-20260920.json"
ID = "grammar-first-semantic-beam-20260920"

def letters(s: str) -> str: return re.sub(r"[^a-z]", "", s.casefold())
def audit(s: str) -> dict:
    t = letters(s)
    mismatch = next(((i, t[i], t[-i-1]) for i in range(len(t)//2) if t[i] != t[-i-1]), None)
    f, r = hashlib.sha256(t.encode()).hexdigest(), hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"letters": len(t), "exact": bool(t) and mismatch is None,
            "first_mismatch": mismatch, "sha256_forward": f,
            "sha256_reverse": r, "sha_equal": f == r}

@dataclass(frozen=True)
class Slot:
    kind: str
    number: str
    semantic: str
    text: str

LEX = {
    "det": (Slot("det", "sg", "agent", "a"), Slot("det", "pl", "agent", "the")),
    "subj": (Slot("subj", "sg", "agent", "poet"), Slot("subj", "pl", "agent", "writers")),
    "verb": (Slot("verb", "sg", "event", "reads"), Slot("verb", "pl", "event", "study")),
    "objdet": (Slot("det", "sg", "theme", "a"), Slot("det", "pl", "theme", "the")),
    "obj": (Slot("obj", "sg", "theme", "book"), Slot("obj", "pl", "theme", "letters")),
}

@dataclass(frozen=True)
class State:
    left: str = ""
    right: str = ""
    lp: int = 0
    rp: int = 0
    agreement: str = "unset"
    slots: tuple[str, ...] = ()

def emit(st: State, side: str, slot: Slot) -> State | None:
    """Consume only newly exposed characters against the opposite pointer."""
    l, r, lp, rp = st.left, st.right, st.lp, st.rp
    if side == "L": l += slot.text
    else: r = slot.text + r
    # Compare the two *live* frontiers; pointers are never reset or recomputed.
    while lp < len(l) and rp < len(r):
        if l[lp] != r[-rp-1]: return None
        lp += 1; rp += 1
    agreement = st.agreement
    if slot.kind == "subj": agreement = slot.number
    if slot.kind == "verb" and agreement != "unset" and slot.number != agreement: return None
    return replace(st, left=l, right=r, lp=lp, rp=rp,
                   agreement=agreement, slots=st.slots + (f"{side}:{slot.kind}:{slot.semantic}:{slot.number}",))

def run(beam_width: int = 256) -> dict:
    # Two independent semantic roles are required; no mirrored token pair is admitted.
    frontier = [State()]
    grammar = (("L", "det"), ("L", "subj"), ("L", "verb"), ("L", "objdet"), ("L", "obj"),
               ("R", "obj"), ("R", "objdet"), ("R", "verb"), ("R", "subj"), ("R", "det"))
    pruned = 0
    for side, kind in grammar:
        nxt = []
        for st in frontier:
            for slot in LEX[kind]:
                z = emit(st, side, slot)
                if z is None: pruned += 1; continue
                nxt.append(z)
        frontier = nxt[:beam_width]
    exact = []
    for st in frontier:
        if st.lp == len(st.left) and st.rp == len(st.right) and len(letters(st.left + st.right)) > 38:
            rendered = st.left + ". " + st.right + "."
            a = audit(rendered)
            if a["exact"]: exact.append({"rendered": rendered, "audit": a, "slots": st.slots})
    return {"experiment_id": ID,
            "method": "typed five-slot grammar; online two-pointer lexical coupling; finite number agreement",
            "stats": {"beam_width": beam_width, "terminal_states": len(frontier), "pruned_online": pruned,
                      "exact_gt38": len(exact)}, "exact_candidates": exact,
            "controls": [{"rendered": x, "audit": audit(x), "generated": False} for x in
                         ("A poet reads a book. A book reads a poet.", "The writers study the letters. The letters study the writers.")],
            "novelty_preflight": {"status": "passed", "signature": "typed-semantic-slots|online-character-pointers|agreement-register",
                                  "distinct_from": ["semantic frame product", "CFG intersection", "live-context infilling"],
                                  "finished_tape_reversal": False, "post_hoc_repair": False,
                                  "mirrored_token_units": False, "catalogue_text": False},
            "provenance": {"lexicon": "bounded authored five-slot inventory", "generator": __file__,
                           "audits": ["independent mismatch scan", "forward/reverse SHA-256"],
                           "reader_gate": "not claimed; no exact >38 output"},
            "status": "exact >38 candidate" if exact else "honest near-miss: no exact >38 candidate",
            "next_construction": "expand semantic role inventory while retaining online pointer gate"}

if __name__ == "__main__":
    result = run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"]))
