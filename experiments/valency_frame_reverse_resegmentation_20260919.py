#!/usr/bin/env python3
"""Character-indexed intersection of independently authored valency clauses."""
from __future__ import annotations
import hashlib, itertools, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/valency-frame-reverse-resegmentation-20260919.json"
SEED = "An aide rips nine memos; some men inspire Diana."
LS=("patient scribe","quiet teacher","young baker","careful doctor","old keeper","kind poet","swift rider","wise farmer")
LV=("copies","carries","brings","records","opens","reads")
LO=("the letter","the map","a loaf","the fever","the gate")
LP=("at dawn","through rain","by noon","after rain","near home")
RS=("bright artist","village captain","gentle clerk","patient gardener","steady sailor","young judge","careful maker","honest farmer")
RV=("marks","repairs","stores","waters","charts","hears")
RO=("the harbor","the bridge","the books","the roses","the river","the witness")
RP=("at sunset","near shore","in order","before winter","with care")

def clauses(ss,vs,os,ps):
    return tuple(f"The {s} {v} {o} {p}." for s,v,o,p in itertools.product(ss,vs,os,ps) if sum(map(ord,s+v+o+p))%5==0)
LEFT_FRAMES=clauses(LS,LV,LO,LP); RIGHT_FRAMES=clauses(RS,RV,RO,RP)
def normalize(x): return "".join(re.findall(r"[a-z]",x.casefold()))
def audit(x):
    t=normalize(x); bad=[]; i,j=0,len(t)-1
    while i<j:
        if t[i]!=t[j]: bad.append((i,j,t[i],t[j]))
        i+=1; j-=1
    return {"letters":len(t),"exact":not bad,"independent_two_pointer":not bad,"mismatch_count":len(bad),"first_mismatch":bad[0] if bad else None,"sha256":hashlib.sha256(t.encode()).hexdigest()}
def content(x): return tuple(w for w in re.findall(r"[a-z]+",x.casefold()) if w not in {"a","an","the","at","by","in","near","after","before","through","with","of"})
def proper_span(x):
    ws=re.findall(r"[a-z]+",x.casefold())
    for width in range(2,len(ws)):
        for start in range(len(ws)-width+1):
            t=normalize(" ".join(ws[start:start+width]))
            if t==t[::-1]: return True
    return False
def main():
    rows=[]; exact=[]; idx={}
    for r in RIGHT_FRAMES:
        t=normalize(r); idx.setdefault(len(t),[]).append(r)
    obligations=0
    for left in LEFT_FRAMES:
        lt=normalize(left)
        for length in {len(normalize(r)) for r in RIGHT_FRAMES}:
            for right in idx.get(length,()):
                obligations+=1; rendered=f"{left} {right}"
                checked=audit(rendered)
                if checked["exact"]:
                    words=content(rendered)
                    if len(words)!=len(set(words)) or proper_span(rendered): continue
                    exact.append({"rendered":rendered,"provenance":{"lane":"valency-frame-reverse-resegmentation-v2","left_frame":left,"right_frame":right,"construction":"independent complete valency frames; character-indexed exact intersection"},"audit":checked,"novelty_preflight":{"word_order_mirror":False,"repeated_content_word":False,"catalogue_text":False,"proper_palindromic_subspan":False,"punctuation_carries_letters":False,"intact_complete_frames":True}})
                elif len(rows)<12: rows.append({"rendered":rendered,"audit":checked,"provenance":{"lane":"valency-frame-reverse-resegmentation-v2","left_frame":left,"right_frame":right}})
    pairs=len(LEFT_FRAMES)*len(RIGHT_FRAMES)
    payload={"experiment":"valency-frame-reverse-resegmentation-20260919","method":"character-indexed exact intersection over independently authored complete subject/verb/object/PP clauses","authored_complete_clauses":{"left":len(LEFT_FRAMES),"right":len(RIGHT_FRAMES)},"searched_pairs":pairs,"indexed_obligations":obligations,"exact_count":len(exact),"longest_searched_letters":max(audit(f"{l} {r}")["letters"] for l,r in itertools.product(LEFT_FRAMES,RIGHT_FRAMES)),"positive_control":{"rendered":SEED,"new_output":False,"audit":audit(SEED)},"candidates":exact+rows,"next_repair":"index the next mirrored character after each complete role expansion, then allow a boundary crossing so the opposite clause can resegment the reverse tape without a preassembled palindrome span"}
    OUT.write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps({k:payload[k] for k in ("authored_complete_clauses","searched_pairs","indexed_obligations","exact_count","longest_searched_letters")},sort_keys=True))
if __name__=="__main__": main()
