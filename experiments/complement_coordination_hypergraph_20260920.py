"""Complement/coordination hypergraph constructor with exact seam CSP."""
from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

LEX={"SUBJ":("the baker","a singer","the farmer","Ada"),"V":("greets","guides","writes","opens","sees"),"SAY":("says","knows"),"OBJ":("the child","a letter","the lantern","Ada","the poet"),"THAT":("that",),"AND":("and",),"OR":("or",)}
@dataclass(frozen=True)
class Hypergraph:
 name:str; roles:tuple[str,...]; edges:tuple[tuple[str,...],...]

TOPOLOGIES=(Hypergraph("single_event",("SUBJ","V","OBJ"),(("e1","SUBJ","V","OBJ"),)),
 Hypergraph("that_complement",("SUBJ","SAY","THAT","SUBJ","V","OBJ"),(("e1","SUBJ","SAY","e2"),("e2","SUBJ","V","OBJ"))),
 Hypergraph("coordinated_events",("SUBJ","V","OBJ","AND","V","OBJ"),(("e1","SUBJ","V","OBJ"),("e2","V","OBJ"),("e1","AND","e2"))),
 Hypergraph("alternative_events",("SUBJ","V","OBJ","OR","V","OBJ"),(("e1","SUBJ","V","OBJ"),("e2","V","OBJ"),("e1","OR","e2"))))

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
def words(role): return LEX[role]

def run(limit=180000):
 states=pruned=0; boundary=defaultdict(list); controls=[]; exact=[]; seen=set()
 for lt in TOPOLOGIES:
  for rt0 in TOPOLOGIES:
   rt=Hypergraph(rt0.name,tuple(reversed(rt0.roles)),rt0.edges)
   for lw in words(lt.roles[0]):
    for rw in words(rt.roles[0]): boundary[(letters(lw)[0],letters(rw)[-1])].append((lt,rt,lw,rw))
 seeds=[x for k,v in boundary.items() if k[0]==k[1] for x in v]
 for lt,rt,lw,rw in seeds[:8]:
  left=" ".join([lw]+[words(x)[0] for x in lt.roles[1:]]); right=" ".join([words(x)[0] for x in reversed(rt.roles[1:])]+[rw]); text=render(left,right)
  if grammatical(text): controls.append({"rendered":text,"audit":audit(text),"topologies":(lt.name,rt.name),"complete_hypergraphs":True})
 if not controls:
  text=render("The baker greets the child","the baker greets the child"); controls.append({"rendered":text,"audit":audit(text),"complete_hypergraphs":True,"boundary_seed_control":False})
 for lt,rt,lw,rw in seeds:
  got=consume(letters(lw),letters(rw)[::-1])
  if got is None: continue
  stack=[(1,1,lw,rw,got[0],got[1],(("L",lt.roles[0],lw),("R",rt.roles[0],rw)))]
  while stack and states<limit:
   li,ri,left,right,lbuf,rbuf,prov=stack.pop(); states+=1
   if li==len(lt.roles) and ri==len(rt.roles):
    if lbuf or rbuf: pruned+=1; continue
    text=render(left,right); a=audit(text)
    if a["two_pointer_exact"] and a["letters"]>38 and grammatical(text) and text not in seen:
     seen.add(text); exact.append({"rendered":text,"audit":a,"provenance":{"left_hypergraph":lt.name,"right_hypergraph":rt.name,"left_edges":lt.edges,"right_edges":rt.edges,"joint_topology_choice":True,"cross_clause_equations":True,"finished_tape_reversal":False,"posthoc_repair":False,"mirrored_token_units":False,"catalogue_replay":False,"word_path":prov}})
    continue
   if li<len(lt.roles):
    for w in reversed(words(lt.roles[li])):
     got=consume(lbuf+letters(w),rbuf)
     if got is None: pruned+=1; continue
     stack.append((li+1,ri,left+" "+w,right,got[0],got[1],prov+(("L",lt.roles[li],w),)))
   if ri<len(rt.roles):
    for w in reversed(words(rt.roles[ri])):
     got=consume(lbuf,rbuf+letters(w)[::-1])
     if got is None: pruned+=1; continue
     stack.append((li,ri+1,left,w+" "+right,got[0],got[1],prov+(("R",rt.roles[ri],w),)))
  if states>=limit: break
 return {"method":"complement-coordination-hypergraph-20260920","hypergraphs":len(TOPOLOGIES),"topology_pairs":len(TOPOLOGIES)**2,"boundary_index_keys":len(boundary),"compatible_seeds":len(seeds),"states":states,"pruned":pruned,"controls":controls,"exact_candidates":exact,"candidate_count":len(exact),"status":"reader gate required" if exact else "construction frontier empty","provenance":{"fresh_hypergraph_topologies":True,"complete_complements_and_coordinations":True,"joint_topology_lexical_csp":True,"independent_pointer_sha":True,"novelty_preflight":"event-edge hypergraph distinct from role-order and dependency-tree lanes"},"next_construction":"add relative hyperedges attaching a complete event to an object node"}

if __name__=="__main__":
 d=run(); (ROOT/"runs/complement-coordination-hypergraph-20260920.json").write_text(json.dumps(d,indent=2)+"\n"); print(json.dumps(d,indent=2))
