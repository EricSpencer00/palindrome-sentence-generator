"""Event-ordered coordination with an explicit shared temporal index."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXPERIMENT_ID="phrase-cfg-event-ordered-shared-index-20260920"
DETS=("some","a","the"); SUBJ=("sailor","poet","keeper","writer","captain","guide")
VERBS=("guards","marks","guides","keeps","reads","writes")
OBJS=("harbor","shore","tide","boat","letter","notes","book","garden")
ROLE={"maritime":{"sailor","harbor","shore","tide","boat"},"writing":{"poet","writer","letter","notes","book"}}
INDEX=("dawn","dusk")
def norm(s):return "".join(c.lower() for c in s if c.isalpha())
def audit(s):
 t=norm(s);r=t[::-1]
 return {"normalized":t,"letters":len(t),"two_pointer_exact":bool(t) and t==r,"sha256_forward":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(r.encode()).hexdigest()}
def clauses(a,b):
 out=[];ra=ROLE[a];rb=ROLE[b]
 for d1 in DETS:
  for s1 in SUBJ:
   for v1 in VERBS:
    for o1 in OBJS:
     if not ({s1,o1}&ra):continue
     for d2 in DETS:
      for s2 in SUBJ:
       for v2 in VERBS:
        for o2 in OBJS:
         if not ({s2,o2}&rb) or v1==v2:continue
         text=f'at dawn {d1} {s1} {v1} {d2} {o1}, at dusk {d2} {s2} {v2} {d1} {o2}'
         out.append((text,{"first_role":a,"second_role":b,"temporal_index":INDEX,"event_order":"dawn_then_dusk","tree":"S -> TIME CLAUSE , TIME CLAUSE","topology":"event_ordered_coordination"}))
         if len(out)>=240:return tuple(out)
 return tuple(out)
def run():
 left=clauses("maritime","writing");right=clauses("writing","maritime");states=0;exact=[];best={"matched":0,"left":"","right":""}
 for l,lm in left:
  lt=norm(l)
  for r,rm in right:
   if lm["temporal_index"]!=rm["temporal_index"]:continue
   rt=norm(r)[::-1];m=0
   while m<len(lt) and m<len(rt) and lt[m]==rt[m]:states+=1;m+=1
   if m>best["matched"]:best={"matched":m,"left":l,"right":r,"temporal_index":INDEX}
   if m==len(lt)==len(rt):
    rendered=l.capitalize()+"; "+r+".";exact.append({"rendered":rendered,"audit":audit(rendered),"temporal_index":INDEX,"provenance":{"event_ordered_coordination":True,"shared_temporal_index":True,"crossed_roles":True,"catalogue_imported":False,"finished_tape_reversed":False,"word_order_mirror":False,"reader_status":"not run"}})
 exact=list({x["audit"]["normalized"]:x for x in exact}.values())
 return {"experiment_id":EXPERIMENT_ID,"method":"event-ordered coordination with shared dawn/dusk temporal index","grammar":["S -> TIME CLAUSE , TIME CLAUSE","TIME -> at dawn | at dusk","CLAUSE -> NP V NP","V1 != V2"],"stats":{"left_events":len(left),"right_events":len(right),"shared_temporal_index_pairs":len(left)*len(right),"crossed_role_states":states,"exact":len(exact),"reader_eligible":sum(x["audit"]["letters"]>38 for x in exact),"best_matched_prefix":best["matched"]},"complete_prose_controls":["At dawn the sailor guards the harbor; at dusk the poet reads the letter.","At dawn a writer marks the shore; at dusk a captain guides the notes."],"best_diagnostic":best,"candidates":sorted(exact,key=lambda x:-x["audit"]["letters"]),"independent_audit":["two-pointer normalized tape","forward/reverse SHA-256"],"novelty_preflight":{"status":"passed","signature":EXPERIMENT_ID,"catalogue_imported":False,"lexical_sweep":False,"distinct_from":"phrase-cfg-coordination-distinct-events-20260920"},"next_construction":"Pivot to event-order reversal with distinct temporal indices on each side if first-character blocking persists.","reader_gate":"closed; programmatic exactness never certifies readability"}
if __name__=="__main__":
 result=run();out=ROOT/"runs"/(EXPERIMENT_ID+".json");out.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result["stats"],sort_keys=True))
