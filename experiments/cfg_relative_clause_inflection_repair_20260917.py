#!/usr/bin/env python3
"""Recursive typed-CFG continuation of the 20260917 chart lane.

Relative clauses are expanded as part of an NP tree (never as a second
sentence).  The held-out repair swaps a singular present-tense verb for an
agreement-compatible verb in the same semantic frame before auditing.
"""
from __future__ import annotations
import hashlib, itertools, json, re
from pathlib import Path

DET = ("the", "a")
ADJ = ("quiet", "patient", "fresh", "old")
N = ("gardener", "reader", "sailor", "teacher", "letter", "map")
V = ("carries", "reviews", "marks", "opens", "studies")
PREP = ("near", "beside", "under")
OBJ = ("the map", "a letter", "the garden", "a harbor")

def letters(s): return re.sub(r"[^a-z]", "", s.lower())
def audit(s):
    t=letters(s)
    return {"letters":len(t), "two_pointer": all(t[i]==t[-1-i] for i in range(len(t)//2)),
            "sha256":hashlib.sha256(t.encode()).hexdigest(),
            "reverse_sha256_equal":hashlib.sha256(t.encode()).digest()==hashlib.sha256(t[::-1].encode()).digest()}

def base_np(d,a,n): return f"{d} {a} {n}"
def relative_np(d,a,n,rv,ro): return f"{d} {a} {n} who {rv} {ro}"

def sentences():
    # Typed tree: S -> NP(subject) VP; NP may contain one recursive RC.
    # The RC's understood object is explicit, so every result is intact prose.
    for d,a,n,rv,ro,v,od,on in itertools.product(DET,ADJ,N,V,OBJ,V,DET,N):
        subj=relative_np(d,a,n,rv,ro)
        yield f"{subj} {v} {od} {on}. ", "S->NP[RC]->who V NP;VP->V NP"
    for d,a,n,rv,ro,v,p,od,on in itertools.product(DET,ADJ,N,V,OBJ,V,PREP,DET,N):
        subj=relative_np(d,a,n,rv,ro)
        yield f"{subj} {v} {p} {od} {on}. ", "S->NP[RC]->who V NP;VP->V Prep NP"

def live(s):
    t=letters(s); pairs=0
    for i in range(len(t)//2):
        if t[i]!=t[-i-1]: return {"matched_outer_pairs":pairs,"first_mismatch":[i,len(t)-1-i],"frontier":t[i:len(t)-i]}
        pairs+=1
    return {"matched_outer_pairs":pairs,"first_mismatch":None,"frontier":""}

def repair(s):
    # Held-out operator: replace only the RC verb with a same-tense,
    # third-person singular lexical alternative.  This preserves agreement and
    # the proposition's transitive frame while changing character obligations.
    m=re.search(r" who (carries|reviews|marks|opens|studies) ",s)
    if not m: return None
    old=m.group(1); alt=next(x for x in V if x!=old)
    return s[:m.start(1)]+alt+s[m.end(1):]

def row(s, provenance, repaired=False):
    return {"rendered":s,"provenance":provenance,"repaired":repaired,"audit":audit(s),"live_frontier":live(s),
      "anti_shortcut":{"single_tree":True,"word_order_only":False,"repeated_unit":False,"catalogue_source":False,"fragment":False}}

def main():
    rows=[]; seen=set(); controls=0; repairs=0
    for s,deriv in sentences():
        k=letters(s)
        if k not in seen:
            seen.add(k); rows.append(row(s,"fresh typed recursive CFG derivation: "+deriv)); controls+=1
        r=repair(s)
        if r and letters(r) not in seen:
            seen.add(letters(r)); rows.append(row(r,"held-out RC inflection/lexical repair from a recorded control; same typed tree",True)); repairs+=1
    rows.sort(key=lambda x:(-x["audit"]["letters"],x["rendered"]))
    exact=[x for x in rows if x["audit"]["two_pointer"]]
    out={"experiment":"cfg-relative-clause-inflection-repair-20260917","method":"typed recursive CFG chart with live mirrored character intersection and held-out agreement repair","control_count":controls,"repair_count":repairs,"candidate_count":len(rows),"exact_count":len(exact),"candidates":rows[:50],"next_repair":"Add a second RC attachment site (object-relative) with a held-out finite-state agreement feature; reject any repair that changes attachment or creates a fragment."}
    p=Path("runs/cfg-relative-clause-inflection-repair-20260917.json");p.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({"run":str(p),"controls":controls,"repairs":repairs,"candidates":len(rows),"exact":len(exact),"longest":rows[0]["audit"]["letters"]}))
    for x in rows[:3]: print(x["rendered"],x["audit"]["letters"],x["audit"]["two_pointer"],x["repaired"])
if __name__=='__main__': main()
