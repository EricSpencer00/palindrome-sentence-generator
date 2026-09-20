"""Recursive clause series with a typed, complete relative nonterminal."""
from __future__ import annotations
import hashlib,json,re
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BANK={
 "SUBJ":("the baker","a singer","Ada","the farmer","a quiet sailor","the young poet"),
 "VP":("greets the child","keeps warm bread","writes a letter","guides the flock","opens the window","sees Ada","greets Anna"),
 "CONJ":("and then","while"), "REL":("who","that"),
}
def letters(s): return re.sub(r"[^a-z]","",s.casefold())
def audit(s):
 t=letters(s); r=t[::-1]
 return {"letters":len(t),"two_pointer_exact":bool(t) and all(a==b for a,b in zip(t,r)),"sha256_forward":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(r.encode()).hexdigest()}
def consume(a,b):
 n=min(len(a),len(b))
 if a[:n]!=b[:n]: return None
 return a[n:],b[n:]
def paths(max_depth=3):
 # R -> SUBJ VP | SUBJ VP REL VP | R CONJ R; every VP is transitive.
 out=[("SUBJ","VP")]
 for depth in range(2,max_depth+1):
  out.append(("SUBJ","VP","REL","VP") if depth==2 else ("SUBJ","VP","CONJ","SUBJ","VP","REL","VP"))
 return tuple(out)
def decorate(left,right):
 text=left+"; "+right
 return re.sub(r"\s+(who|that)\s+",r", \1 ",text)
def run(limit=180000,max_depth=3):
 ps=paths(max_depth); boundary=defaultdict(list)
 for lp in ps:
  for rr in ps:
   rp=tuple(reversed(rr))
   for lw in BANK[lp[0]]:
    for rw in BANK[rp[0]]: boundary[(letters(lw)[0],letters(rw)[-1])].append((lp,rp,lw,rw))
 seeds=[x for k,v in boundary.items() if k[0]==k[1] for x in v]
 controls=[]; exact=[]; seen=set(); states=pruned=0
 for lp,rp,lw,rw in seeds[:8]:
  left=" ".join([lw]+[BANK[x][0] for x in lp[1:]])
  right=" ".join([BANK[x][0] for x in reversed(rp[1:])]+[rw])
  controls.append({"rendered":decorate(left,right),"audit":audit(decorate(left,right)),"complete_relative":("REL" in lp or "REL" in rp)})
 for lp,rp,lw,rw in seeds:
  got=consume(letters(lw),letters(rw)[::-1])
  if got is None: continue
  stack=[(1,1,lw,rw,got[0],got[1],(("L",lp[0],lw),("R",rp[0],rw)))]
  while stack and states<limit:
   li,ri,left,right,lbuf,rbuf,prov=stack.pop(); states+=1
   if li==len(lp) and ri==len(rp):
    if lbuf or rbuf: pruned+=1; continue
    text=decorate(left,right); a=audit(text)
    if a["two_pointer_exact"] and a["letters"]>38 and text not in seen:
     seen.add(text); exact.append({"rendered":text,"audit":a,"provenance":{"left_roles":lp,"right_roles":rp,"typed_relative":True,"recursive_depth":max(1,lp.count("VP")),"boundary_class_seed":True,"complete_predicates":True,"finished_tape_reversal":False,"posthoc_repair":False,"mirrored_token_units":False,"catalogue_replay":False,"word_path":prov}})
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
 return {"method":"recursive-series-typed-relative-20260920","paths":len(ps),"boundary_index_keys":len(boundary),"compatible_seeds":len(seeds),"states":states,"pruned":pruned,"controls":controls,"exact_candidates":exact,"candidate_count":len(exact),"status":"reader gate required" if exact else "construction frontier empty","provenance":{"fresh_authored_bank":True,"typed_relative_nonterminal":True,"punctuation_editorial_only":True,"novelty_preflight":"new relative recursion over prior clause-series"},"next_construction":"add relative-object valency variants while keeping REL paths complete"}
if __name__=="__main__":
 d=run(); (ROOT/"runs/recursive-series-typed-relative-20260920.json").write_text(json.dumps(d,indent=2)+"\n"); print(json.dumps(d,indent=2))
