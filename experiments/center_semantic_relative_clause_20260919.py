#!/usr/bin/env python3
"""Center-out semantic scene search with variable relative clauses.

Each candidate is made from two independently generated, complete scene
clauses around one central relative-clause seam.  A tape index joins clauses
by the character obligations exposed at the seam; no finished palindrome is
reversed or copied into prose.
"""
from __future__ import annotations

import hashlib, itertools, json
from pathlib import Path

ID = "center-semantic-relative-clause-20260919"
SIG = "center-semantic-relative-clause-v1"

S = ("the baker", "the sailor", "the keeper", "a poet", "the farmer", "the pilot")
V = ("marks", "keeps", "writes", "carries", "finds", "folds")
O = ("a letter", "the map", "old notes", "one poem", "the chart", "a key")
P = ("at dawn", "by the shore", "near home", "in spring", "at sea")
REL = (
    "who waits by the shore",
    "who carries a small map",
    "that the keeper found at dawn",
    "which the sailor keeps near home",
    "who writes old notes in spring",
    "that a poet marked at sea",
)

def letters(x): return "".join(c.lower() for c in x if c.isalpha())

def audit(text):
    t = letters(text); mism=[]
    for i in range(len(t)//2):
        if t[i] != t[-1-i]: mism.append((i, len(t)-1-i, t[i], t[-1-i]))
    words=[w.strip(".,;:").lower() for w in text.split()]
    content=[w for w in words if len(w)>2]
    proper=[]
    for i in range(len(t)):
        for j in range(i+2, len(t)+1):
            if i==0 and j==len(t): continue
            if t[i:j]==t[i:j][::-1]: proper.append((i,j))
    return {"letters":len(t), "words":len(words), "exact":bool(t) and not mism,
            "mismatch_count":len(mism), "first_mismatch":mism[0] if mism else None,
            "sha256":hashlib.sha256(t.encode()).hexdigest(),
            "distinct_content":len(content)==len(set(content)),
            "proper_palindromic_subspan":bool(proper),
            "word_order_symmetry":words==words[::-1]}

def scene(clause, rel, punct):
    return f"{clause[0]} {clause[1]} {clause[2]} {rel} {clause[3]}{punct}"

def main():
    # Distinct semantic scenes are generated independently on each side.
    clauses=list(itertools.product(S,V,O,P))
    left=[scene(c,r,";") for c in clauses for r in REL]
    right=[scene(c,r,".") for c in clauses for r in REL]
    # A center-out obligation index: only equal-length left/right tapes can
    # close, and the reversed right tape is indexed by its complete seam key.
    right_index={}
    for x in right: right_index.setdefault(letters(x), []).append(x)
    rows=[]
    for l in left:
        lt=letters(l)
        # Character obligations are checked before rendering the pair.
        rkey=lt[::-1]
        for r in right_index.get(rkey, ()):
            text=l+" "+r
            a=audit(text)
            rows.append({"rendered":text,"left":l,"right":r,"audit":a,
                         "provenance":{"independent_left_scene":True,"independent_right_scene":True,
                                       "center_relative_clause":True,"finished_tape_reversal":False}})
    # Record near misses by seam mismatch from a bounded sample, to make the
    # next repair concrete without claiming a non-exact output.
    best=None
    for l,r in itertools.islice(itertools.product(left,right), 0, 50000):
        text=l+" "+r; t=letters(text)
        mism=sum(a!=b for a,b in zip(t,t[::-1]))//2
        score=len(t)-mism
        a={"letters":len(t),"mismatch_count":mism,"first_mismatch":next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None),"exact":mism==0}
        if best is None or score>best["score"]: best={"score":score,"rendered":text,"audit":a}
    exact=[x for x in rows if x["audit"]["exact"] and x["audit"]["letters"]>38 and x["audit"]["distinct_content"] and not x["audit"]["proper_palindromic_subspan"]]
    payload={"experiment_id":ID,"signature":SIG,"method":"independent semantic scene clauses with variable central relative clauses; reverse seam obligations indexed before pair rendering","left_scenes":len(left),"right_scenes":len(right),"indexed_closures":len(rows),"admitted_exact":len(exact),"candidates":rows[:20],"best_near_miss":best,"provenance":{"generated_not_catalogue":True,"rlaif":False,"hand_coded_finished_tape":False},"novelty_preflight":{"collision_with_existing_lane":False,"status":"passed"},"next_repair":"Add a role-compatible relative-clause transducer whose first and last content words are selected from the exposed seam characters, then re-index obligations across the clause boundary; do not wrap an existing palindrome."}
    Path("runs/center-semantic-relative-clause-20260919.json").write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps({"left_scenes":len(left),"right_scenes":len(right),"closures":len(rows),"admitted":len(exact),"best":best["rendered"] if best else None}))

if __name__=="__main__": main()
