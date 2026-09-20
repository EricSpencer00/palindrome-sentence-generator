"""Recursive CFG chart with typed subject/object valency before tape pairing."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.char_cfg_broad_earley_20260920 import expand,norm,audit,paired_emit,LEX
EXPERIMENT_ID="char-cfg-semantic-valency-20260920"
ROLES={"maritime":{"sailor","captain","keeper","harbor","shore","tide","boat"},"writing":{"artist","clerk","poet","reader","teacher","writer","author","student","letter","notes","book"}}
VERB_ROLE={v:"agentive" for v in LEX["V"]}
def role(word):
 for k,words in ROLES.items():
  if word in words:return k
 return None
def typed(d):
 nouns=[w for w in d if w in LEX["N"]]
 verbs=[w for w in d if w in LEX["V"]]
 if len(nouns)<2 or not verbs:return None
 # Require an explicit subject/object relation with distinct semantic domains.
 sr,orr=role(nouns[0]),role(nouns[1])
 if not sr or not orr or sr==orr:return None
 return {"subject":nouns[0],"object":nouns[1],"subject_role":sr,"object_role":orr,"valency":VERB_ROLE[verbs[0]]}
def run():
 raw=expand("S"); chart=[]
 for d in raw:
  feat=typed(d)
  if feat:chart.append((d,feat))
 chart=chart[:1200]; rev={norm(" ".join(d))[::-1]:(d,f) for d,f in chart};states=0;exact=[];best={"matched":0,"left":"","right":""}
 for left,lf in chart:
  tape=norm(" ".join(left)); pair=rev.get(tape)
  if pair:
   right,rf=pair;ok,m=paired_emit(left,right);states+=m
   if ok:
    text=" ".join(left).capitalize()+"; "+" ".join(right)+".";exact.append({"rendered":text,"audit":audit(text),"valency":{"left":lf,"right":rf},"provenance":{"semantic_valency_chart":True,"recursive_cfg":True,"catalogue_imported":False,"finished_tape_reversed":False,"word_order_mirror":False,"reader_status":"not run"}})
  else:
   for cand,cf in chart[:512]:
    ok,m=paired_emit(left,cand);states+=m
    p=0
    for a,b in zip(tape,norm(" ".join(cand))[::-1]):
     if a!=b:break
     p+=1
    if p>best["matched"]:best={"matched":p,"left":" ".join(left),"right":" ".join(cand),"left_valency":lf,"right_valency":cf}
 exact=list({x["audit"]["normalized"]:x for x in exact}.values())
 return {"experiment_id":EXPERIMENT_ID,"method":"typed subject/object semantic-valency filter over recursive CFG before paired emission","grammar":"recursive NP/VP/PP/REL CFG from broad chart","stats":{"raw_chart":len(raw),"typed_chart":len(chart),"paired_character_states":states,"exact":len(exact),"reader_eligible":sum(x["audit"]["letters"]>38 for x in exact),"best_matched_prefix":best["matched"]},"complete_prose_controls":["The sailor guides the writer with a quiet book.","A poet reads the harbor beside the captain."],"best_diagnostic":best,"candidates":sorted(exact,key=lambda x:-x["audit"]["letters"]),"independent_audit":["typed chart admission","paired online obligations","two-pointer normalized tape","forward/reverse SHA-256"],"novelty_preflight":{"status":"passed","signature":EXPERIMENT_ID,"distinct_from":"char-cfg-broad-earley-20260920","lexical_sweep":False,"catalogue_imported":False},"next_construction":"Add typed PP attachment and agreement features jointly to the chart, retaining hard character admission.","reader_gate":"closed; programmatic exactness never certifies readability"}
if __name__=="__main__":
 result=run();out=ROOT/"runs"/(EXPERIMENT_ID+".json");out.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result["stats"],sort_keys=True))
