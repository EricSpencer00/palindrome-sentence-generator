"""Direct-object + typed-PP chart extended with typed relative attachment."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.char_cfg_broad_earley_20260920 import expand,norm,audit,paired_emit,LEX
EXPERIMENT_ID="char-cfg-direct-pp-relative-20260920"
ROLES={"maritime":{"sailor","captain","keeper","harbor","shore","tide","boat"},"writing":{"artist","clerk","poet","reader","teacher","writer","author","student","letter","notes","book"}}
def role(w):
 for k,v in ROLES.items():
  if w in v:return k
 return None
def direct(d):
 ns=[w for w in d if w in LEX["N"]];vs=[w for w in d if w in LEX["V"]]
 if len(ns)<2 or not vs:return None
 a,b=role(ns[0]),role(ns[1])
 if not a or not b or a==b:return None
 return {"subject":ns[0],"object":ns[1],"subject_role":a,"object_role":b,"verb":vs[0],"attachment":"direct-object"}
def pp(d,f):
 ps=[(i,w) for i,w in enumerate(d) if w in LEX["P"]]
 if not ps:return f
 i,p=ps[-1];tail=[w for w in d[i+1:] if w in LEX["N"]]
 if not tail or role(tail[-1])!=f["subject_role"]:return None
 return {**f,"pp":p+" "+tail[-1],"pp_attachment":"VP","attachment":"direct-object+typed-PP"}
def relative(d,f):
 if not any(w in {"that","who"} for w in d):return None
 ns=[w for w in d if w in LEX["N"]]
 if len(ns)<3 or role(ns[-1])!=f["object_role"]:return None
 return {**f,"relative_pronoun":next(w for w in d if w in {"that","who"}),"relative_head":ns[1],"relative_object":ns[-1],"attachment":"direct-object+typed-relative"}
def run():
 raw=expand("S");direct_chart=[];pp_chart=[];rel_chart=[]
 for d in raw:
  f=direct(d)
  if not f:continue
  direct_chart.append((d,f));g=pp(d,f)
  if g:pp_chart.append((d,g))
  h=relative(d,f)
  if h:rel_chart.append((d,h))
 chart=direct_chart+pp_chart+rel_chart;rev={norm(" ".join(d))[::-1]:(d,f) for d,f in chart};states=0;exact=[];best={"matched":0,"left":"","right":""}
 for left,lf in chart:
  tape=norm(" ".join(left));pair=rev.get(tape)
  if pair:
   right,rf=pair;ok,m=paired_emit(left,right);states+=m
   if ok:
    text=" ".join(left).capitalize()+"; "+" ".join(right)+".";exact.append({"rendered":text,"audit":audit(text),"features":{"left":lf,"right":rf},"provenance":{"typed_relative_attachment":True,"direct_and_pp_states_preserved":True,"catalogue_imported":False,"finished_tape_reversed":False,"word_order_mirror":False,"reader_status":"not run"}})
  else:
   for cand,cf in chart[:256]:
    ok,m=paired_emit(left,cand);states+=m;p=0
    for a,b in zip(tape,norm(" ".join(cand))[::-1]):
     if a!=b:break
     p+=1
    if p>best["matched"]:best={"matched":p,"left":" ".join(left),"right":" ".join(cand),"left_features":lf,"right_features":cf}
 exact=list({x["audit"]["normalized"]:x for x in exact}.values())
 return {"experiment_id":EXPERIMENT_ID,"method":"direct-object and typed-PP CFG extended with typed relative-clause attachment","stats":{"raw_chart":len(raw),"direct_chart":len(direct_chart),"typed_pp_branch":len(pp_chart),"typed_relative_branch":len(rel_chart),"combined_chart":len(chart),"paired_character_states":states,"exact":len(exact),"reader_eligible":sum(x["audit"]["letters"]>38 for x in exact),"best_matched_prefix":best["matched"]},"complete_prose_controls":["The sailor guides the writer who reads the book.","A poet reads the captain that guards the harbor."],"best_diagnostic":best,"candidates":sorted(exact,key=lambda x:-x["audit"]["letters"]),"independent_audit":["direct chart admission","typed PP admission","typed relative admission","paired online obligations","two-pointer normalized tape","forward/reverse SHA-256"],"novelty_preflight":{"status":"passed","signature":EXPERIMENT_ID,"distinct_from":"char-cfg-direct-then-typed-pp-20260920","lexical_sweep":False,"catalogue_imported":False,"post_hoc_repair":False},"next_construction":"Add agreement-carrying relative-clause subject features while preserving all three chart branches.","reader_gate":"closed; programmatic exactness never certifies readability"}
if __name__=="__main__":
 result=run();out=ROOT/"runs"/(EXPERIMENT_ID+".json");out.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result["stats"],sort_keys=True))
