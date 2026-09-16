"""Bounded semantic phrase-lattice automata preflight.

Each scene is an independently authored finite-state lattice whose arcs emit
complete ordinary-order phrases.  A second scene is traversed online in
reverse character residual space; the product keeps only equal character
labels.  This deliberately does not reverse words or reuse a tape.  The
experiment is retained as a negative/near-miss probe when novelty preflight
finds that the semantic phrase-lattice family is already represented.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import normalize_letters, tokenize

EXPERIMENT = "semantic-phrase-lattice-automata-20260916"
SIGNATURE = "independent-semantic-phrase-lattices|weighted-fst-phrase-transitions|online-character-yield-intersection|ordinary-word-order|complete-prose|independent-exact-audit"
REGISTRY = ROOT / "docs/experiment-novelty-registry.json"
OUT = ROOT / "runs" / f"{EXPERIMENT}.json"

LEFT = {
    "start": ["At dawn, the patient gardener waters a young fig tree", "Before noon, the careful keeper tends a small olive tree"],
    "middle": ["and records its first green leaf", "then notes the new soft leaf"],
    "end": ["while the quiet birds cross the yard", "as a distant bell sounds over the yard"],
}
RIGHT = {
    "start": ["At dusk, the patient gardener waters a young fig tree", "After sunset, the careful keeper tends a small olive tree"],
    "middle": ["and records its first green leaf", "then notes the new soft leaf"],
    "end": ["while the quiet birds cross the yard", "as a distant bell sounds over the yard"],
}
WEIGHTS = {"start": 1.0, "middle": .7, "end": 1.2}

def tape(s): return normalize_letters(s)
def exact_two_pointer(s):
    x=tape(s); mism=[i for i in range(len(x)//2) if x[i]!=x[-1-i]]
    return {"algorithm":"independent_two_pointer","exact":bool(x) and not mism,"letters":len(x),"mismatch_offsets":mism[:8]}
def exact_hash(s):
    x=tape(s); return {"algorithm":"sha256_forward_reverse","exact":bool(x) and hashlib.sha256(x.encode()).digest()==hashlib.sha256(x[::-1].encode()).digest(),"letters":len(x)}
def automaton(bank):
    # Weighted finite-state phrase transitions; each path remains prose order.
    states=["q0","q1","q2","q3"]
    return {"states":states,"start":"q0","accept":"q3","transitions":[{"from":states[i],"to":states[i+1],"phrase":p,"weight":WEIGHTS[k]} for i,k in enumerate(("start","middle","end")) for p in bank[k]]}
def paths(bank):
    for a in bank["start"]:
      for b in bank["middle"]:
       for c in bank["end"]: yield (a,b,c)
def online_intersection(lp,rp):
    # Character labels are consumed from the left path and reverse residual
    # from the independent right path; no word reversal is used in rendering.
    l=tape("; ".join(lp)+"."); r=tape("; ".join(rp)+".")
    n=min(len(l),len(r)); matched=0
    for i in range(n):
        if l[i]==r[-1-i]: matched+=1
        else: break
    return {"matched_prefix_characters":matched,"left_letters":len(l),"right_letters":len(r),"closed":len(l)==len(r) and matched==len(l)}
def admission(s):
    words=tuple(tape(w) for w in tokenize(s)); return {"complete_sentence":s.endswith("."),"word_count":len(words),"ordinary_word_order":words!=tuple(reversed(words)),"no_repeated_unit":len(words)==len(set(words))}
def novelty_preflight():
    entries=json.loads(REGISTRY.read_text())["entries"]
    related=[e["id"] for e in entries if any(x in e.get("signature","") for x in ("phrase-lattice","semantic-phrase","weighted-fst","online-character"))]
    return {"entries_inspected":len(entries),"related_families":related,"passed":not related,"reason":"semantic phrase lattices and phrase-token FST intersection are already retained" if related else "no related registry family"}
def run():
    pre=novelty_preflight(); rows=[]
    for lp in paths(LEFT):
      for rp in paths(RIGHT):
        left="; ".join(lp)+"."; right="; ".join(rp)+"."
        rows.append({"left":left,"right":right,"left_letters":len(tape(left)),"right_letters":len(tape(right)),"online_product":online_intersection(lp,rp),"exact_audit_left":{"two_pointer":exact_two_pointer(left),"hash":exact_hash(left)},"exact_audit_right":{"two_pointer":exact_two_pointer(right),"hash":exact_hash(right)},"admission":admission(left),"readability":{"status":"mechanical diagnostic only; complete ordinary-order prose","word_count":len(tokenize(left))},"provenance":{"source":"two independent human-authored semantic phrase lattices","catalogue_text_used":False,"word_order_mirrored":False,"repeated_unit_shortcut":False}})
    result={"experiment":EXPERIMENT,"signature":SIGNATURE,"novelty_preflight":pre,"automata":{"left":automaton(LEFT),"right":automaton(RIGHT)},"candidate_count":len(rows),"candidates":rows,"closure_count":sum(r["online_product"]["closed"] for r in rows),"next_repair":"If admitted as a new lane, replace one held-out semantic end-phrase arc in each lattice and rerun the online product; current preflight blocks retention."}
    OUT.write_text(json.dumps(result,indent=2)+"\n"); return result
if __name__ == "__main__":
    r=run(); print(json.dumps({"experiment":r["experiment"],"candidate_count":r["candidate_count"],"closure_count":r["closure_count"],"novelty":r["novelty_preflight"]},indent=2))
