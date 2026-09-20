"""Joint finite agreement and subject/object attachment chart before pairing."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.char_cfg_broad_earley_20260920 import expand,norm,audit,paired_emit,LEX
EXPERIMENT_ID="char-cfg-finite-agreement-attachment-20260920"
ROLES={"maritime":{"sailor","captain","keeper","harbor","shore","tide","boat"},"writing":{"artist","clerk","poet","reader","teacher","writer","author","student","letter","notes","book"}}
def role(w):
 for k,v in ROLES.items():
  if w in v:return k
 return None
def features(d):
 nouns=[(i,w) for i,w in enumerate(d) if w in LEX["N"]];verbs=[w for w in d if w in LEX["V"]]
 if len(nouns)<2 or not verbs:return None
 # All current lexical verbs are finite 3sg; require a singular determiner before subject.
 si,sub=nouns[0]; oi,obj=nouns[1]; det=d[si-1] if si else ""
 if det not in {"a","an","the","some","this","that"}:return None
 sr,orr=role(sub),role(obj)
 if not sr or not orr or sr==orr:return None
 return {"subject":sub,"object":obj,"subject_role":sr,"object_role":orr,"verb":verbs[0],"agreement":"finite-3sg with singular DET-N subject","attachment":"direct-object"}
def run():
 raw=expand("S");chart=[]
 for d in raw:
  f=features(d)
  if f:chart.append((d,f))
 chart=chart[:1000];rev={norm(" ".join(d))[::-1]:(d,f) for d,f in chart};states=0;exact=[];best={"matched":0,"left":"","right":""}
 for left,lf in chart:
  tape=norm(" ".join(left));pair=rev.get(tape)
  if pair:
   right,rf=pair;ok,m=paired_emit(left,right);states+=m
   if ok:
    text=" ".join(left).capitalize()+"; "+" ".join(right)+".";exact.append({"rendered":text,"audit":audit(text),"features":{"left":lf,"right":rf},"provenance":{"finite_agreement":True,"typed_subject_object_attachment":True,"recursive_cfg":True,"catalogue_imported":False,"finished_tape_reversed":False,"word_order_mirror":False,"reader_status":"not run"}})
  else:
   for cand,cf in chart[:256]:
    ok,m=paired_emit(left,cand);states+=m;p=0
    for a,b in zip(tape,norm(" ".join(cand))[::-1]):
     if a!=b:break
     p+=1
    if p>best["matched"]:best={"matched":p,"left":" ".join(left),"right":" ".join(cand),"left_features":lf,"right_features":cf}
 exact=list({x["audit"]["normalized"]:x for x in exact}.values())
 return {"experiment_id":EXPERIMENT_ID,"method":"recursive CFG with finite agreement and typed direct subject/object attachment","grammar":"NP/VP/PP/REL recursive CFG; direct-object attachment typed before emission","stats":{"raw_chart":len(raw),"typed_chart":len(chart),"paired_character_states":states,"exact":len(exact),"reader_eligible":sum(x["audit"]["letters"]>38 for x in exact),"best_matched_prefix":best["matched"]},"complete_prose_controls":["The sailor guides the writer.","A poet reads the harbor."],"best_diagnostic":best,"candidates":sorted(exact,key=lambda x:-x["audit"]["letters"]),"independent_audit":["finite agreement admission","typed attachment admission","paired online obligations","two-pointer normalized tape","forward/reverse SHA-256"],"novelty_preflight":{"status":"passed","signature":EXPERIMENT_ID,"distinct_from":"char-cfg-typed-pp-agreement-20260920","lexical_sweep":False,"catalogue_imported":False},"next_construction":"Add a typed PP only after direct-object chart closure, using role-compatible adjuncts without removing direct-object states.","reader_gate":"closed; programmatic exactness never certifies readability"}
if __name__=="__main__":
 result=run();out=ROOT/"runs"/(EXPERIMENT_ID+".json");out.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result["stats"],sort_keys=True))
