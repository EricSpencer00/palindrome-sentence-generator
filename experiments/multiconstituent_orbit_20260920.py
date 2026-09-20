"""Simultaneous multi-constituent orbit search.

Each side contributes an independently authored constituent at every step;
the shared character obligation is consumed while constituents are selected,
not after either complete sentence has been built.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/multiconstituent-orbit-20260920.json"
EXPERIMENT_ID = "multiconstituent-orbit-20260920"

def letters(s): return re.sub(r"[^a-z]", "", s.casefold())
def audit(s):
    t = letters(s); bad = next(((i, len(t)-1-i) for i in range(len(t)//2) if t[i] != t[-1-i]), None)
    return {"letters": len(t), "exact": bool(t) and bad is None, "first_mismatch": bad,
            "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(t[::-1].encode()).hexdigest(),
            "two_pointer_exact": bool(t) and bad is None}
def shortcut_flags(words):
    toks = [letters(w) for w in words]
    return {"mirrored_token_unit": any(a == b[::-1] for a in toks for b in toks if a != b),
            "embedded_palindrome_span": any(len(t) > 3 and t == t[::-1] for t in toks)}

CONSTITUENTS = {
 "subject": ("the young poet", "a careful nurse", "the quiet baker", "an eager clerk", "the old farmer", "a bright teacher", "the local artist", "one patient scribe"),
 "predicate": ("reads a letter", "writes a poem", "marks the record", "carries a map", "keeps the promise", "names the child", "opens the gate", "tends the garden"),
 "object": ("the small book", "a blue lantern", "the old story", "a fresh memo", "the long road", "one red apple", "the kind visitor", "a quiet room"),
 "pp": ("at the harbor", "by the garden", "in the school", "near the river", "with a friend", "on the quiet lane"),
 "relative": ("who reads well", "that helps the child", "who keeps the record", "that knows the road"),
 "coord": ("and sings softly", "and waits outside", "and tells a story", "and walks home"),
}
SHAPES = (("subject", "predicate", "object", "pp"), ("subject", "predicate", "relative", "object"),
          ("subject", "predicate", "object", "coord"))

def valid_constituent(w):
    t = letters(w)
    return bool(t) and w != w[::-1] and len(t) >= 3

def search(shape, cap=250000):
    left = right = ""; states = rejected = 0; candidates = []
    # Both independent sides are expanded together. Compatibility is only
    # against the currently exposed tape, never a completed clause reversal.
    def walk(i, lp, rp, ls, rs):
        nonlocal states, rejected
        if states >= cap: return
        if i == len(shape):
            surface = " ".join(ls + rs)
            a = audit(surface)
            flags = shortcut_flags(ls + rs)
            if a["exact"] and len(set(ls + rs)) == len(ls + rs) and not any(flags.values()) and len(surface.split()) >= 8:
                candidates.append({"rendered": surface, "audit": a, "provenance": {
                    "construction": "simultaneous multi-constituent orbit",
                    "shape": shape, "finished_tape_reversal": False,
                    "post_hoc_repair": False, "catalogue_text": False,
                    **flags}})
            return
        role = shape[i]
        for l in CONSTITUENTS[role]:
            if not valid_constituent(l): continue
            for r in CONSTITUENTS[role]:
                if l == r or r in ls or l in rs: continue
                states += 1
                # consume the exposed pair of character buffers as soon as
                # each independently authored constituent is selected
                nl, nr = lp + letters(l), letters(r) + rp
                n = min(len(nl), len(nr))
                if nl[:n] == nr[:n][::-1]:
                    walk(i+1, nl[n:], nr[n:], ls+(l,), (r,)+rs)
                else: rejected += 1
    walk(0, left, right, (), ())
    return {"states": states, "rejected": rejected, "candidates": candidates[:20]}

CONTROLS = ("The young poet reads a letter at the harbor.", "A careful nurse writes a poem in the school.",
            "The quiet baker carries a map by the garden.", "An eager clerk marks the record near the river.",
            "The local artist opens the gate and waits outside.", "One patient scribe keeps the promise and tells a story.")

def main():
    searches = {" ".join(s): search(s) for s in SHAPES}
    exact = [c for r in searches.values() for c in r["candidates"] if c["audit"]["exact"]]
    result = {"experiment_id": EXPERIMENT_ID, "method": "simultaneous independent multiword constituents with shared live character obligation", "searches": searches, "exact_candidates": exact, "best_length": max((c["audit"]["letters"] for c in exact), default=0), "controls": [{"surface": s, "audit": audit(s), "natural": True} for s in CONTROLS], "reader_gate": "closed; controls are diagnostic only", "next_repair": "Expand authored constituent inventory with held-out lexicalized clauses; do not repair these states."}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"states": sum(x["states"] for x in searches.values()), "exact": len(exact), "best_length": result["best_length"]}))
if __name__ == "__main__": main()
