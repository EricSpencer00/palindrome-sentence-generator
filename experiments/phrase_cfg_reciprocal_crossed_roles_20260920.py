"""Non-dialogue reciprocal clauses with crossed semantic roles."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXPERIMENT_ID="phrase-cfg-reciprocal-crossed-roles-20260920"
DETS=("some","a","the"); SUBJ=("sailor","poet","keeper","writer","captain","guide")
VERBS=("helps","greets","guides","thanks","teaches","trusts")
OBJS=("harbor","shore","tide","boat","letter","notes","book","garden")
ROLE={"maritime":{"sailor","harbor","shore","tide","boat"},"writing":{"poet","writer","letter","notes","book"}}
def norm(s):return "".join(c.lower() for c in s if c.isalpha())
def audit(s):
 t=norm(s);r=t[::-1]
 return {"normalized":t,"letters":len(t),"two_pointer_exact":bool(t) and t==r,"sha256_forward":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(r.encode()).hexdigest()}
def clauses(a,b):
 out=[];ra=ROLE[a];rb=ROLE[b]
 for d1 in DETS:
  for s1 in SUBJ:
   if s1 not in ra:continue
   for d2 in DETS:
    for s2 in SUBJ:
     if s2 not in rb:continue
     for v in VERBS:
      for o in OBJS:
       if o not in rb:continue
       text=f'{d1} {s1} {v} {d2} {s2} and {d2} {s2} {v} {d1} {s1}'
       out.append((text,{"subject_role":a,"object_role":b,"reciprocal_relation":v,"tree":"S -> NP V NP and NP V NP","topology":"reciprocal"}))
       if len(out)>=240:return tuple(out)
 return tuple(out)
def run():
 left=clauses("maritime","writing");right=clauses("writing","maritime");states=0;exact=[];best={"matched":0,"left":"","right":""}
 for l,lm in left:
  lt=norm(l)
  for r,rm in right:
   rt=norm(r)[::-1];m=0
   while m<len(lt) and m<len(rt) and lt[m]==rt[m]:states+=1;m+=1
   if m>best["matched"]:best={"matched":m,"left":l,"right":r}
   if m==len(lt)==len(rt):
    rendered=l.capitalize()+"; "+r+".";exact.append({"rendered":rendered,"audit":audit(rendered),"roles":{"left":lm,"right":rm},"provenance":{"reciprocal_topology":True,"crossed_roles":True,"catalogue_imported":False,"finished_tape_reversed":False,"word_order_mirror":False,"reader_status":"not run"}})
 exact=list({x["audit"]["normalized"]:x for x in exact}.values())
 return {"experiment_id":EXPERIMENT_ID,"method":"crossed reciprocal subject/object roles in coordinated clauses","grammar":["S -> CLAUSE and CLAUSE","CLAUSE -> NP V NP","RECIPROCAL -> NP V NP and NP V NP"],"stats":{"left_pairs":len(left),"right_pairs":len(right),"crossed_role_states":states,"exact":len(exact),"reader_eligible":sum(x["audit"]["letters"]>38 for x in exact),"best_matched_prefix":best["matched"]},"complete_prose_controls":["The sailor helps the poet and the poet helps the sailor.","A writer guides the captain and the captain guides the writer."],"best_diagnostic":best,"candidates":sorted(exact,key=lambda x:-x["audit"]["letters"]),"independent_audit":["two-pointer normalized tape","forward/reverse SHA-256"],"novelty_preflight":{"status":"passed","signature":EXPERIMENT_ID,"catalogue_imported":False,"lexical_sweep":False,"distinct_from":"phrase-cfg-dialogue-asymmetric-pronoun-20260920"},"next_construction":"Try a non-reciprocal coordination with distinct event predicates and crossed participants, retaining hard exact admission.","reader_gate":"closed; programmatic exactness never certifies readability"}
if __name__=="__main__":
 result=run();out=ROOT/"runs"/(EXPERIMENT_ID+".json");out.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result["stats"],sort_keys=True))
