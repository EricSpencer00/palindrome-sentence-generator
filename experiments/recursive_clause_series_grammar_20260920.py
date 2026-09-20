"""Recursive compositional clause-series constructor.

Complete semantic clauses are composed recursively to bounded depth.  Each
side lexicalizes its own role path, seeded by compatible exposed character
classes and checked by live residual consumption.  No finished tape is
reversed or repaired.
"""
from __future__ import annotations
import hashlib,json,re
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

# Fresh banks: every VP is a complete transitive predicate, and CONJ makes a
# complete clause-series composition rather than a fragment bridge.
BANK={
 "SUBJ":("the baker","a singer","a dog","Ada","the farmer","a quiet sailor","the young poet","a patient keeper"),
 "VP":("greets the child","keeps warm bread","writes a letter","guides the flock","opens the window","carries fresh water","sees Ada","greets Anna"),
 "CONJ":("and then","while","and the"),
}

def letters(s): return re.sub(r"[^a-z]","",s.casefold())
def audit(s):
 t=letters(s); r=t[::-1]
 return {"letters":len(t),"two_pointer_exact":bool(t) and all(a==b for a,b in zip(t,r)),"sha256_forward":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(r.encode()).hexdigest()}
def consume(a,b):
 n=min(len(a),len(b))
 if a[:n]!=b[:n]: return None
 return a[n:],b[n:]

def recursive_paths(max_depth=3):
 """C -> SUBJ VP | SUBJ VP CONJ C, materialized only to bounded depth."""
 out=[]
 for depth in range(1,max_depth+1):
  p=("SUBJ","VP")
  for _ in range(depth-1): p=p+("CONJ","SUBJ","VP")
  out.append(p)
 return tuple(out)

def run(limit=180000,max_depth=3):
 paths=recursive_paths(max_depth); states=pruned=0; exact=[]; seen=set(); boundary=defaultdict(list)
 for lp in paths:
  for rp_rendered in paths:
   rp=tuple(reversed(rp_rendered))
   for lw in BANK[lp[0]]:
    for rw in BANK[rp[0]]:
     boundary[(letters(lw)[0],letters(rw)[-1])].append((lp,rp,lw,rw))
 seeds=[x for key,items in boundary.items() if key[0]==key[1] for x in items]
 controls=[]
 for lp,rp,lw,rw in seeds[:8]:
  left=" ".join([lw]+[BANK[x][0] for x in lp[1:]])
  right=" ".join([BANK[x][0] for x in reversed(rp[1:])]+[rw])
  rendered = left+"; "+right+"."
  controls.append({"rendered":rendered,"audit":audit(rendered),"complete_clause_series":True})
 for lp,rp,lw,rw in seeds:
  got=consume(letters(lw),letters(rw)[::-1])
  if got is None: continue
  stack=[(1,1,lw,rw,got[0],got[1],(("L",lp[0],lw),("R",rp[0],rw)))]
  while stack and states<limit:
   li,ri,left,right,lbuf,rbuf,prov=stack.pop(); states+=1
   if li==len(lp) and ri==len(rp):
    if lbuf or rbuf: pruned+=1; continue
    # Punctuation is editorial only: it is excluded by ``letters`` and does
    # not participate in the character equations, but keeps the two complete
    # clause series readable when shown to a person.
    text=left+"; "+right+"."; a=audit(text)
    if a["two_pointer_exact"] and a["letters"]>38 and text not in seen:
     seen.add(text); exact.append({"rendered":text,"audit":a,"provenance":{"left_roles":lp,"right_roles":rp,"recursive_depth":(len(lp)+1)//3,"boundary_class_seed":True,"phrase_bank":"fresh-authored-complete-clause","finished_tape_reversal":False,"posthoc_repair":False,"mirrored_token_units":False,"catalogue_replay":False,"word_path":prov}})
    continue
   if li<len(lp):
    for w in reversed(BANK[lp[li]]):
     got=consume(lbuf+letters(w),rbuf)
     if got is None: pruned+=1; continue
     stack.append((li+1,ri,left+" "+w,right,got[0],got[1],prov+(("L",lp[li],w),)))
   if ri<len(rp):
    for w in reversed(BANK[rp[ri]]):
     got=consume(lbuf,rbuf+letters(w)[::-1])
     if got is None: pruned+=1; continue
     stack.append((li,ri+1,left,w+" "+right,got[0],got[1],prov+(("R",rp[ri],w),)))
  if states>=limit: break
 return {"method":"recursive-clause-series-grammar-20260920","max_depth":max_depth,"paths":len(paths),"boundary_index_keys":len(boundary),"compatible_seeds":len(seeds),"states":states,"pruned":pruned,"controls":controls,"exact_candidates":exact,"candidate_count":len(exact),"status":"reader gate required" if exact else "construction frontier empty","next_construction":"add a recursive relative clause as a typed nonterminal, preserving complete predicate valency and boundary-class indexing"}

if __name__=="__main__":
 d=run(); (ROOT/"runs/recursive-clause-series-grammar-20260920.json").write_text(json.dumps(d,indent=2)+"\n"); print(json.dumps(d,indent=2))
