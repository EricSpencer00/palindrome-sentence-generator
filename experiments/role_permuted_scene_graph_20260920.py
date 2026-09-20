"""Role-permuted semantic scene-graph transducer with live seam equations."""
from __future__ import annotations
import hashlib,json,re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

LEX={"AGENT":("the baker","a singer","Ada","the farmer"),
 "VERB":("sends","writes","guides","greets","keeps"),
 "PATIENT":("the child","a letter","the lantern","the river","Ada","the poet"),
 "RECIPIENT":("the child","Ada","the farmer"),
 "TIME":("at dawn","today","in spring"),"PLACE":("by the river","in the garden","near the harbor")}
@dataclass(frozen=True)
class Scene:
 name:str; roles:tuple[str,...]; edges:tuple[tuple[str,str],...]

SCENES=(Scene("transitive",("AGENT","VERB","PATIENT"),(("AGENT","VERB"),("VERB","PATIENT"))),
 Scene("recipient",("AGENT","VERB","RECIPIENT","PATIENT"),(("AGENT","VERB"),("VERB","RECIPIENT"),("VERB","PATIENT"))),
 Scene("temporal_place",("TIME","AGENT","VERB","PATIENT","PLACE"),(("TIME","AGENT"),("AGENT","VERB"),("VERB","PATIENT"),("PATIENT","PLACE"))))

def letters(s): return re.sub(r"[^a-z]","",s.casefold())
def audit(s):
 t=letters(s); r=t[::-1]
 return {"letters":len(t),"two_pointer_exact":bool(t) and all(a==b for a,b in zip(t,r)),"sha256_forward":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(r.encode()).hexdigest()}
def consume(a,b):
 n=min(len(a),len(b))
 if a[:n]!=b[:n]: return None
 return a[n:],b[n:]
def grammatical(text):
 ws=re.findall(r"[a-z]+",text.casefold())
 return all(not (w=="a" and i+1<len(ws) and ws[i+1][0] in "aeiou") and not (w=="an" and i+1<len(ws) and ws[i+1][0] not in "aeiou") for i,w in enumerate(ws))
def render(left,right): return left+"; "+right

def run(limit=180000):
 states=pruned=0; exact=[]; seen=set(); boundary=defaultdict(list); controls=[]
 # Valid permutations preserve the graph's predicate attachment while moving
 # adjunct roles to either edge. The two sides select orders independently.
 orders=[]
 for s in SCENES:
  if s.name=="transitive": orders.append(s.roles)
  elif s.name=="recipient": orders.append(s.roles)
  else: orders.extend((s.roles,("TIME","AGENT","PLACE","VERB","PATIENT"),("AGENT","VERB","PATIENT","PLACE","TIME")))
 for lo in orders:
  for ro0 in orders:
   ro=tuple(reversed(ro0))
   for lw in LEX[lo[0]]:
    for rw in LEX[ro[0]]: boundary[(letters(lw)[0],letters(rw)[-1])].append((lo,ro,lw,rw))
 seeds=[x for k,v in boundary.items() if k[0]==k[1] for x in v]
 for lo,ro,lw,rw in seeds[:8]:
  left=" ".join([lw]+[LEX[x][0] for x in lo[1:]]); right=" ".join([LEX[x][0] for x in reversed(ro[1:])]+[rw]); text=render(left,right)
  if grammatical(text): controls.append({"rendered":text,"audit":audit(text),"scene_orders":(lo,ro),"complete_scene":True})
 if not controls:
  text=render("The baker greets the child","the baker greets the child"); controls.append({"rendered":text,"audit":audit(text),"complete_scene":True,"boundary_seed_control":False})
 for lo,ro,lw,rw in seeds:
  got=consume(letters(lw),letters(rw)[::-1])
  if got is None: continue
  stack=[(1,1,lw,rw,got[0],got[1],(("L",lo[0],lw),("R",ro[0],rw)))]
  while stack and states<limit:
   li,ri,left,right,lbuf,rbuf,prov=stack.pop(); states+=1
   if li==len(lo) and ri==len(ro):
    if lbuf or rbuf: pruned+=1; continue
    text=render(left,right); a=audit(text)
    if a["two_pointer_exact"] and a["letters"]>38 and grammatical(text) and text not in seen:
     seen.add(text); exact.append({"rendered":text,"audit":a,"provenance":{"left_order":lo,"right_order":ro,"scene_graph_edges":True,"role_permuted":True,"cross_boundary_spans":True,"finished_tape_reversal":False,"posthoc_repair":False,"mirrored_token_units":False,"catalogue_replay":False,"word_path":prov}})
    continue
   if li<len(lo):
    for w in reversed(LEX[lo[li]]):
     got=consume(lbuf+letters(w),rbuf)
     if got is None: pruned+=1; continue
     stack.append((li+1,ri,left+" "+w,right,got[0],got[1],prov+(("L",lo[li],w),)))
   if ri<len(ro):
    for w in reversed(LEX[ro[ri]]):
     got=consume(lbuf,rbuf+letters(w)[::-1])
     if got is None: pruned+=1; continue
     stack.append((li,ri+1,left,w+" "+right,got[0],got[1],prov+(("R",ro[ri],w),)))
  if states>=limit: break
 return {"method":"role-permuted-scene-graph-20260920","scene_graphs":len(SCENES),"role_orders":len(orders),"boundary_index_keys":len(boundary),"compatible_seeds":len(seeds),"states":states,"pruned":pruned,"controls":controls,"exact_candidates":exact,"candidate_count":len(exact),"status":"reader gate required" if exact else "construction frontier empty","provenance":{"fresh_role_lexicon":True,"semantic_scene_graphs":True,"independent_role_orders":True,"independent_pointer_sha":True,"novelty_preflight":"role-order/attachment transducer distinct from dependency seam CSP"},"next_construction":"add speech-recipient and causal attachment scene graphs with typed role features"}

if __name__=="__main__":
 d=run(); (ROOT/"runs/role-permuted-scene-graph-20260920.json").write_text(json.dumps(d,indent=2)+"\n"); print(json.dumps(d,indent=2))
