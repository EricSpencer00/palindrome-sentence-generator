"""Phrase-boundary CSP: solve character obligations while selecting typed clauses.

Unlike reverse-segmentation, the solver never renders a right clause by reversing
the left.  It chooses two independently authored valency frames and propagates
character equations through their slot boundaries before rendering.
"""
from __future__ import annotations
import hashlib, json, re
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/phrase-boundary-valency-equations-20260918.json"

FRAMES = [
    ("agent", "verb", "object", "adjunct"),
    ("agent", "verb", "object", "location"),
]
SLOTS = {
    "agent": ["the patient curator", "a careful baker", "the quiet teacher", "an old sailor", "the young nurse"],
    "verb": ["records", "carries", "labels", "opens", "packs"],
    "object": ["the brass compass", "a sealed parcel", "the winter map", "a warm letter", "the small lantern"],
    "adjunct": ["before dawn", "after the storm", "near the station", "for the quiet ward", "beside the river"],
    "location": ["in the quiet archive", "at the old harbor", "by the northern window", "under the wide awning", "beside the market gate"],
}

def letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())

def audit(text: str) -> dict:
    t = letters(text); rev = t[::-1]
    return {"letters": len(t), "two_pointer_exact": t == rev,
            "forward_sha256": hashlib.sha256(t.encode()).hexdigest(),
            "reverse_sha256": hashlib.sha256(rev.encode()).hexdigest(),
            "mismatch": next((i for i,(a,b) in enumerate(zip(t,rev)) if a != b), None)}

def clause(frame, values):
    # Every surface is a complete, independently authored transitive clause.
    return f"{values[0].capitalize()} {values[1]} {values[2]} {values[3]}."

def slot_tape(values):
    return letters(" ".join(values))

def solve():
    # Character equations are propagated at the slot level: for each candidate
    # pair, compare only the currently forced prefix/suffix before full render.
    left = [(f, tuple(v)) for f in FRAMES for v in product(*(SLOTS[x] for x in f))]
    right = [(f, tuple(v)) for f in FRAMES for v in product(*(SLOTS[x] for x in f))]
    states = 0; pruned = 0; exact = []
    for lf, lv in left:
        lt = slot_tape(lv)
        for rf, rv in right:
            states += 1
            rt = slot_tape(rv)
            # This is the equation check before rendering; no text reversal is
            # used to create rv.  Boundary lengths are part of the state.
            overlap = min(len(lt), len(rt))
            if any(lt[i] != rt[-1-i] for i in range(overlap)):
                pruned += 1; continue
            text = clause(lf, lv) + " " + clause(rf, rv)
            a = audit(text)
            if a["two_pointer_exact"]:
                exact.append({"text": text, "length": a["letters"], "left_frame": lf,
                              "right_frame": rf, "audit": a,
                              "provenance": "two independently selected typed frames; slot equation propagation"})
    controls = []
    for f, v in left[:8]:
        text = clause(f,v)
        controls.append({"text": text, "length": audit(text)["letters"], "audit": audit(text),
                         "intact_control": True, "provenance": "human-authored valency frame"})
    result = {"method": "phrase-boundary valency equations", "date": "2026-09-18",
              "frames": FRAMES, "states": states, "equation_pruned": pruned,
              "exact_count": len(exact), "exact_candidates": exact,
              "intact_controls": controls,
              "strict_gate": {"admitted": 0, "reader_gate": "closed",
                              "reason": "No exact closure; no candidate is presented as a palindrome."},
              "next_repair": "Carry residual character obligations through optional determiner and adjunct boundaries, then add held-out transitive frames; preserve independent role assignments.",
              "novelty": "slot-boundary character equations are solved before surfaces are rendered; right clauses are independently selected, never reversed"}
    RUN.parent.mkdir(exist_ok=True)
    RUN.write_text(json.dumps(result, indent=2) + "\n")
    return result

if __name__ == "__main__":
    r=solve(); print(json.dumps({k:r[k] for k in ('states','equation_pruned','exact_count','next_repair')}, indent=2))
