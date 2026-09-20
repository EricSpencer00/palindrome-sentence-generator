"""Exact-by-construction two-clause search with unequal live buffers.

The buffers contain only unmatched characters.  A newly selected phrase is
compared against the opposite buffer and consumed immediately; previously
matched characters are never indexed again.  This is a construction search,
not a repair pass.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

SEED = "An aide rips nine memos; some men inspire Diana."

def letters(s): return re.sub(r"[^a-z]", "", s.casefold())

def audit(s):
    t=letters(s); f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"letters":len(t),"exact":bool(t) and t==t[::-1],"sha256_forward":f,"sha256_reverse":r,"sha_equal":f==r}

def consume(left: str, right: str):
    """Append left-side text and right-side text; return unmatched buffers."""
    a,b=left,right
    while a and b and a[0]==b[-1]: a,b=a[1:],b[:-1]
    return a,b

def search(limit=8):
    bank=json.loads(Path("data/brown_pcfg_bank_20260920.json").read_text())["lexicon"]
    def top(tag,n): return tuple(x["word"].casefold() for x in bank.get(tag,[])[:n])
    det=top("DET",8); noun=top("NOUN",32); verb=top("VERB",32); adp=top("ADP",8)
    # Common typed clauses, independently composed from word banks.
    clauses=tuple(f"{d} {n} {v} {d2} {m}" for d in det for n in noun for v in verb for d2 in det[:4] for m in noun[:16])
    pp=tuple(f"{p} {n}" for p in adp for n in noun[:16])
    states=pruned=0; witnesses=[]; exact=[]
    for left in clauses[:256]:
      for right in tuple(reversed(clauses[:256])):
        states+=1
        a,b=consume(letters(left),letters(right))
        if a or b:
            pruned+=1
            if len(witnesses)<limit: witnesses.append({"rendered":left+"; "+right,"audit":audit(left+"; "+right),"provenance":{"complete":False,"unmatched_left":a,"unmatched_right":b,"grammar":"DET NOUN VERB DET NOUN ; DET NOUN VERB DET NOUN","catalogue_text":False}})
            continue
        item={"rendered":left+"; "+right,"audit":audit(left+"; "+right),"provenance":{"complete":True,"grammar":"two independent typed clauses","catalogue_text":False}}
        exact.append(item)
    # A known exact closure is retained only as a regression, never as output.
    seed_regression={"rendered":SEED,"audit":audit(SEED),"provenance":{"regression":True,"generated_by_this_lane":False}}
    return {"run_id":"two-clause-live-buffer-20260919","stats":{"states":states,"pruned":pruned,"exact":len(exact),"over_38":sum(x["audit"]["letters"]>38 for x in exact)},"candidates":witnesses,"exact_candidates":exact,"seed_regression":seed_regression,"next_construction":"increase clause grammar breadth only after adding semantic valency constraints; do not index unequal buffers"}

if __name__=="__main__": print(json.dumps(search(),indent=2))
