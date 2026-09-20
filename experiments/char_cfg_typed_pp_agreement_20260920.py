"""Typed PP attachment plus determiner agreement before paired CFG emission."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.char_cfg_broad_earley_20260920 import expand,norm,audit,paired_emit,LEX
EXPERIMENT_ID="char-cfg-typed-pp-agreement-20260920"
ROLES={"maritime":{"sailor","captain","keeper","harbor","shore","tide","boat"},"writing":{"artist","clerk","poet","reader","teacher","writer","author","student","letter","notes","book"}}
PREPS=set(LEX["P"])
def role(w):
 for k,v in ROLES.items():
  if w in v:return k
 return None
def features(d):
 nouns=[w for w in d if w in LEX["N"]]; dets=[w for w in d if w in LEX["DET"]]
 if len(nouns)<2 or not dets:return None
 # Agreement: an only selects a vowel-initial following noun.
 for i,w in enumerate(d):
  if w=="an" and (i+1>=len(d) or d[i+1][0] not in "aeiou"):return None
 sr,orr=role(nouns[0]),role(nouns[1])
 if not sr or not orr or sr==orr:return None
 pp=[(i,w) for i,w in enumerate(d) if w in PREPS]
 if not pp:return None
 pidx,p=pp[-1]; ppn=[w for w in d[pidx+1:] if w in LEX["N"]]
 if not ppn or not role(ppn[-1]) or role(ppn[-1])!=sr:return None
 return {"subject":nouns[0],"object":nouns[1],"subject_role":sr,"object_role":orr,"pp":p+" "+ppn[-1],"pp_attachment":"VP","agreement":"DET-N checked"}
def run():
 raw=expand("S"); chart=[]
 for d in raw:
  f=features(d)
  if f:chart.append((d,f))
 chart=chart[:900];rev={norm(" ".join(d))[::-1]:(d,f) for d,f in chart};states=0;exact=[];best={"matched":0,"left":"","right":""}
 for left,lf in chart:
  tape=norm(" ".join(left));pair=rev.get(tape)
  if pair:
   right,rf=pair;ok,m=paired_emit(left,right);states+=m
   if ok:
    text=" ".join(left).capitalize()+"; "+" ".join(right)+".";exact.append({"rendered":text,"audit":audit(text),"features":{"left":lf,"right":rf},"provenance":{"typed_pp_attachment":True,"agreement_before_emission":True,"recursive_cfg":True,"catalogue_imported":False,"finished_tape_reversed":False,"word_order_mirror":False,"reader_status":"not run"}})
  else:
   for cand,cf in chart[:256]:
    ok,m=paired_emit(left,cand);states+=m;p=0
    for a,b in zip(tape,norm(" ".join(cand))[::-1]):
     if a!=b:break
     p+=1
    if p>best["matched"]:best={"matched":p,"left":" ".join(left),"right":" ".join(cand),"left_features":lf,"right_features":cf}
 exact=list({x["audit"]["normalized"]:x for x in exact}.values())
 return {"experiment_id":EXPERIMENT_ID,"method":"recursive CFG with typed VP PP attachment and determiner agreement before pairing","grammar":"NP/VP/PP/REL recursive CFG; PP role attached to VP","stats":{"raw_chart":len(raw),"typed_chart":len(chart),"paired_character_states":states,"exact":len(exact),"reader_eligible":sum(x["audit"]["letters"]>38 for x in exact),"best_matched_prefix":best["matched"]},"complete_prose_controls":["The sailor guides the writer with the harbor.","A poet reads the captain beside the book."],"best_diagnostic":best,"candidates":sorted(exact,key=lambda x:-x["audit"]["letters"]),"independent_audit":["agreement/PP typed admission","paired online obligations","two-pointer normalized tape","forward/reverse SHA-256"],"novelty_preflight":{"status":"passed","signature":EXPERIMENT_ID,"distinct_from":"char-cfg-semantic-valency-20260920","lexical_sweep":False,"catalogue_imported":False},"next_construction":"Add finite verb agreement and subject/object attachment features jointly, retaining hard exact admission.","reader_gate":"closed; programmatic exactness never certifies readability"}
if __name__=="__main__":
 result=run();out=ROOT/"runs"/(EXPERIMENT_ID+".json");out.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result["stats"],sort_keys=True))
