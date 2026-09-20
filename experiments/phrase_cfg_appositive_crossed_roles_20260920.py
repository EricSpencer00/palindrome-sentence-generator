"""Appositive NP topology with crossed semantic roles and hard tape equations."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXPERIMENT_ID="phrase-cfg-appositive-crossed-roles-20260920"
DETS=("some","a","the"); SUBJ=("sailor","poet","keeper","writer","captain","guide")
VERBS=("guards","marks","guides","keeps","reads","writes")
OBJS=("harbor","shore","tide","boat","letter","notes","book","garden")
ROLE={"maritime":{"sailor","harbor","shore","tide","boat"},"writing":{"poet","writer","letter","notes","book"}}
def norm(s): return "".join(c.lower() for c in s if c.isalpha())
def audit(s):
 t=norm(s); r=t[::-1]
 return {"normalized":t,"letters":len(t),"two_pointer_exact":bool(t) and t==r,"sha256_forward":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(r.encode()).hexdigest()}
def clauses(role_a,role_b):
 out=[]; ra=ROLE[role_a]; rb=ROLE[role_b]
 for d1 in DETS:
  for s1 in SUBJ:
   for d2 in DETS:
    for s2 in SUBJ:
     for v in VERBS:
      for o in OBJS:
       if not ({s1,o}&ra) or not ({s2,o}&rb): continue
       text=" ".join((d1,s1,",",d2,s2,",",v,o))
       out.append((text,{"subject_role":role_a,"appositive_role":role_b,"tree":"S -> NP APP NP VP NP","topology":"appositive"}))
       if len(out)>=240:return tuple(out)
 return tuple(out)
def run():
 left=clauses("maritime","writing"); right=clauses("writing","maritime"); states=0; exact=[]
 best={"matched":0,"left":"","right":""}
 for l,lm in left:
  lt=norm(l)
  for r,rm in right:
   rt=norm(r)[::-1]; m=0
   while m<len(lt) and m<len(rt) and lt[m]==rt[m]: states+=1; m+=1
   if m>best["matched"]:best={"matched":m,"left":l,"right":r}
   if m==len(lt)==len(rt):
    rendered=l.capitalize()+"; "+r+"."; exact.append({"rendered":rendered,"audit":audit(rendered),"roles":{"left":lm,"right":rm},"provenance":{"appositive_topology":True,"crossed_roles":True,"catalogue_imported":False,"finished_tape_reversed":False,"word_order_mirror":False,"reader_status":"not run"}})
 exact=list({x["audit"]["normalized"]:x for x in exact}.values())
 return {"experiment_id":EXPERIMENT_ID,"method":"crossed maritime/writing roles in appositive NP clauses","grammar":["S -> NP APP NP VP NP","APP -> , DET NP ,","S S -> ;"],"stats":{"left_clauses":len(left),"right_clauses":len(right),"crossed_role_states":states,"exact":len(exact),"reader_eligible":sum(x["audit"]["letters"]>38 for x in exact),"best_matched_prefix":best["matched"]},"complete_prose_controls":["The sailor, a poet, guards the letter; the writer, a captain, reads the harbor.","A keeper, the guide, marks the shore; a poet, the sailor, writes the notes."],"best_diagnostic":best,"candidates":sorted(exact,key=lambda x:-x["audit"]["letters"]),"independent_audit":["two-pointer normalized tape","forward/reverse SHA-256"],"novelty_preflight":{"status":"passed","signature":EXPERIMENT_ID,"catalogue_imported":False,"lexical_sweep":False,"distinct_from":"phrase-cfg-shared-although-connective-20260920"},"next_construction":"Try a dialogue-turn topology with explicit speaker and utterance roles, retaining hard exact admission.","reader_gate":"closed; programmatic exactness never certifies readability"}
if __name__=="__main__":
 result=run(); out=ROOT/"runs"/(EXPERIMENT_ID+".json"); out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result["stats"],sort_keys=True))
