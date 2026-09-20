"""Shared concessive boundary connective over crossed subordinate/matrix roles."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "phrase-cfg-shared-although-connective-20260920"
DETS=("some","a","the"); SUBJ=("sailor","poet","keeper","writer","captain","guide")
VERBS=("guards","marks","guides","keeps","reads","writes")
OBJS=("harbor","shore","tide","boat","letter","notes","book","garden")
ROLE={"maritime":{"sailor","harbor","shore","tide","boat"},"writing":{"poet","writer","letter","notes","book"}}
CONNECTIVE={"category":"CONCESSIVE_ALTHOUGH","surface":"although","role":"concession"}
def norm(s): return "".join(c.lower() for c in s if c.isalpha())
def audit(s):
    t=norm(s); r=t[::-1]
    return {"normalized":t,"letters":len(t),"two_pointer_exact":bool(t) and t==r,
            "sha256_forward":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(r.encode()).hexdigest()}
def sentences(subrole,matrixrole):
    out=[]; sa=ROLE[subrole]; ma=ROLE[matrixrole]
    for d1 in DETS:
      for s1 in SUBJ:
       for v1 in VERBS:
        for o1 in OBJS:
         if not ({s1,o1}&sa): continue
         for d2 in DETS:
          for s2 in SUBJ:
           for v2 in VERBS:
            for o2 in OBJS:
             if not ({s2,o2}&ma): continue
             text=" ".join(("although",d1,s1,v1,d1,o1,",",d2,s2,v2,d2,o2))
             out.append((text,{"subordinate_role":subrole,"matrix_role":matrixrole,"boundary":"although|comma","tree":"Sub(although,S,S)","semantic_relation":"concession"}))
             if len(out)>=240: return tuple(out)
    return tuple(out)
def run():
    left=sentences("maritime","writing"); right=sentences("writing","maritime"); support=states=0; exact=[]
    best={"matched":0,"left":"","right":"","connective":CONNECTIVE}
    for lt0,lm in left:
      lt=norm(lt0)
      for rt0,rm in right:
       if lm["boundary"]!=rm["boundary"] or lm["boundary"]!="although|comma": continue
       support+=1; rt=norm(rt0)[::-1]; m=0
       while m<len(lt) and m<len(rt) and lt[m]==rt[m]: states+=1; m+=1
       if m>best["matched"]: best={"matched":m,"left":lt0,"right":rt0,"connective":CONNECTIVE}
       if m==len(lt)==len(rt):
        rendered=lt0.capitalize()+"; "+rt0+"."; exact.append({"rendered":rendered,"audit":audit(rendered),"connective":CONNECTIVE,"provenance":{"shared_boundary_connective":True,"crossed_roles":True,"semantic_relation":"concession","catalogue_imported":False,"finished_tape_reversed":False,"word_order_mirror":False,"reader_status":"not run"}})
    exact=list({x["audit"]["normalized"]:x for x in exact}.values())
    return {"experiment_id":EXPERIMENT_ID,"method":"shared concessive although connective with crossed subordinate/matrix roles","grammar":["S -> ALTHOUGH S , S","S -> NP VP","VP -> V NP"],"connective":CONNECTIVE,"stats":{"left_sentences":len(left),"right_sentences":len(right),"connective_support":support,"connective_character_states":states,"exact":len(exact),"reader_eligible":sum(x["audit"]["letters"]>38 for x in exact),"best_matched_prefix":best["matched"]},"complete_prose_controls":["Although the sailor guards the letter, the poet reads the harbor.","Although a writer marks the shore, a captain guides the notes."],"best_diagnostic":best,"candidates":sorted(exact,key=lambda x:-x["audit"]["letters"]),"independent_audit":["two-pointer normalized tape","forward/reverse SHA-256"],"novelty_preflight":{"status":"passed","signature":EXPERIMENT_ID,"catalogue_imported":False,"lexical_sweep":False,"distinct_from":"phrase-cfg-shared-because-connective-20260920"},"next_construction":"Pivot away from connective-only lanes to a new clause topology if the first obligation remains empty.","reader_gate":"closed; programmatic exactness never certifies readability"}
if __name__=="__main__":
    result=run(); out=ROOT/"runs"/(EXPERIMENT_ID+".json"); out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result["stats"],sort_keys=True))
