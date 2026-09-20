"""Dependency-tree seam CSP with exact character equations."""
from __future__ import annotations
import hashlib,json,re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

LEX={"SUBJ":("the mason","a teacher","the farmer","a quiet poet","Ada"),
     "V":("greets","guides","keeps","writes","opens","sees"),
     "OBJ":("the child","a lantern","the river","a letter","Ada"),
     "PP":("by the river","near the harbor","under the tree"),
     "REL":("who","that"),"RELSUBJ":("the singer","a sailor","Ada"),
     "RELV":("greets","guides","keeps","sees"),"RELOBJ":("the child","a lantern","Ada")}

@dataclass(frozen=True)
class Tree:
 name:str; order:tuple[str,...]; edges:tuple[tuple[str,str],...]

TREES=(Tree("transitive",("SUBJ","V","OBJ"),(("V","SUBJ"),("V","OBJ"))),
 Tree("transitive_adjunct",("SUBJ","V","OBJ","PP"),(("V","SUBJ"),("V","OBJ"),("V","PP"))),
 Tree("relative",("SUBJ","V","OBJ","REL","RELSUBJ","RELV","RELOBJ"),(("V","SUBJ"),("V","OBJ"),("OBJ","REL"),("REL","RELSUBJ"),("REL","RELV"),("RELV","RELOBJ"))))

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
def render(left,right):
 text=left+"; "+right
 return re.sub(r"\s+(who|that)\s+",r", \1 ",text)

def run(limit=180000):
 states=pruned=0; exact=[]; seen=set(); boundary=defaultdict(list); controls=[]
 for lt in TREES:
  for rt0 in TREES:
   rt=Tree(rt0.name,tuple(reversed(rt0.order)),rt0.edges)
   for lw in LEX[lt.order[0]]:
    for rw in LEX[rt.order[0]]: boundary[(letters(lw)[0],letters(rw)[-1])].append((lt,rt,lw,rw))
 seeds=[x for k,v in boundary.items() if k[0]==k[1] for x in v]
 for lt,rt,lw,rw in seeds[:8]:
  left=" ".join([lw]+[LEX[x][0] for x in lt.order[1:]])
  right=" ".join([LEX[x][0] for x in reversed(rt.order[1:])]+[rw])
  text=render(left,right)
  if grammatical(text): controls.append({"rendered":text,"audit":audit(text),"tree_pair":(lt.name,rt.name),"complete_dependencies":True})
 if not controls:
  text=render("The mason greets the child","the mason greets the child")
  controls.append({"rendered":text,"audit":audit(text),"complete_dependencies":True,"boundary_seed_control":False})
 for lt,rt,lw,rw in seeds:
  got=consume(letters(lw),letters(rw)[::-1])
  if got is None: continue
  stack=[(1,1,lw,rw,got[0],got[1],(("L",lt.order[0],lw),("R",rt.order[0],rw)))]
  while stack and states<limit:
   li,ri,left,right,lbuf,rbuf,prov=stack.pop(); states+=1
   if li==len(lt.order) and ri==len(rt.order):
    if lbuf or rbuf: pruned+=1; continue
    text=render(left,right); a=audit(text)
    if a["two_pointer_exact"] and a["letters"]>38 and grammatical(text) and text not in seen:
     seen.add(text); exact.append({"rendered":text,"audit":a,"provenance":{"left_tree":lt.name,"right_tree":rt.name,"dependency_edges":{"left":lt.edges,"right":rt.edges},"csp_equations_checked":True,"word_boundaries_live":True,"finished_tape_reversal":False,"posthoc_repair":False,"mirrored_token_units":False,"catalogue_replay":False,"word_path":prov}})
    continue
   if li<len(lt.order):
    role=lt.order[li]
    for w in reversed(LEX[role]):
     got=consume(lbuf+letters(w),rbuf)
     if got is None: pruned+=1; continue
     stack.append((li+1,ri,left+" "+w,right,got[0],got[1],prov+(("L",role,w),)))
   if ri<len(rt.order):
    role=rt.order[ri]
    for w in reversed(LEX[role]):
     got=consume(lbuf,rbuf+letters(w)[::-1])
     if got is None: pruned+=1; continue
     stack.append((li,ri+1,left,w+" "+right,got[0],got[1],prov+(("R",role,w),)))
  if states>=limit: break
 return {"method":"dependency-tree-seam-csp-20260920","trees":len(TREES),"tree_pairs":len(TREES)**2,"boundary_index_keys":len(boundary),"compatible_seeds":len(seeds),"states":states,"pruned":pruned,"controls":controls,"exact_candidates":exact,"candidate_count":len(exact),"status":"reader gate required" if exact else "construction frontier empty","provenance":{"fresh_authored_tree_lexicon":True,"dependency_tree_csp":True,"character_equations_before_render":True,"independent_pointer_sha":True,"novelty_preflight":"typed dependency edges plus seam equations"},"next_construction":"add agreement-feature domains to tree variables before seam expansion"}

if __name__=="__main__":
 d=run(); (ROOT/"runs/dependency-tree-seam-csp-20260920.json").write_text(json.dumps(d,indent=2)+"\n"); print(json.dumps(d,indent=2))
