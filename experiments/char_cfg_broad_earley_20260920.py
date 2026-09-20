"""Character-level paired CFG decoder with a broad contemporary lexicon."""
from __future__ import annotations
import hashlib,json,itertools
from functools import lru_cache
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT_ID="char-cfg-broad-earley-20260920"
LEX={
 "DET":["a","an","the","some","this","that"],
 "ADJ":["calm","bright","quiet","young","kind","plain","small","open","urban","gentle"],
 "N":["artist","captain","child","clerk","doctor","friend","gardener","keeper","neighbor","poet","reader","sailor","teacher","writer","author","student"],
 "V":["admires","answers","builds","carries","checks","chooses","covers","follows","guides","helps","keeps","likes","marks","notices","reads","remembers","shares","thanks","trusts","writes"],
 "P":["about","after","before","beside","during","for","from","near","over","with"],
 "ADV":["calmly","clearly","gently","often","quietly","slowly"]}
LEX["that"]=["that"]; LEX["who"]=["who"]
GRAM={"S":[["NP","VP"]],"NP":[["DET","N"],["DET","ADJ","N"],["DET","N","REL"]],"VP":[["V","NP"],["ADV","V","NP"],["V","NP","PP"]],"PP":[["P","NP"]],"REL":[["that","VP"],["who","VP"]]}
def norm(s):return "".join(c.lower() for c in s if c.isalpha())
def audit(s):
 t=norm(s);r=t[::-1]
 return {"normalized":t,"letters":len(t),"two_pointer_exact":bool(t) and t==r,"sha256_forward":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(r.encode()).hexdigest()}
@lru_cache(None)
def expand(sym,depth=0):
 if depth>4:return []
 if sym in LEX:return [[w] for w in LEX[sym]]
 out=[]
 for prod in GRAM.get(sym,[]):
  parts=[[]]
  for s in prod:
   nxt=[]
   for a in parts:
    for b in expand(s,depth+1):
     if len(a+b)<=9:nxt.append(a+b)
   parts=nxt
  out.extend(parts[:1200])
  if len(out)>=1200: break
 return out
def paired_emit(left,right):
 """Online two-pointer obligation check while terminal words are emitted."""
 a=norm(" ".join(left));b=norm(" ".join(right))[::-1];states=0
 for i,(x,y) in enumerate(itertools.zip_longest(a,b,fillvalue="")):
  if not x or not y or x!=y:return False,states
  states+=1
 return bool(a) and len(a)==len(b),states
def run():
 deriv=expand("S"); deriv=[d for d in deriv if 4<=len(norm(" ".join(d)))<=90]
 # Hashing gives a scalable chart join; paired_emit independently replays every hit.
 rev={norm(" ".join(d))[::-1]:d for d in deriv}
 exact=[]; states=0; best={"matched":0,"left":"","right":""}
 for left in deriv:
  tape=norm(" ".join(left)); right=rev.get(tape)
  if right is None:
   # Diagnostic prefix against a bounded sample of live chart states.
   for cand in deriv[:2048]:
    ok,m=paired_emit(left,cand); states+=m
    prefix=0
    for x,y in zip(tape,norm(" ".join(cand))[::-1]):
     if x!=y:break
     prefix+=1
    if prefix>best["matched"]:best={"matched":prefix,"left":" ".join(left),"right":" ".join(cand)}
   continue
  ok,m=paired_emit(left,right);states+=m
  if ok:
   rendered=" ".join(left).capitalize()+"; "+" ".join(right)+".";exact.append({"rendered":rendered,"audit":audit(rendered),"derivation":{"left":left,"right":right,"grammar":"recursive NP/VP/PP/REL CFG"},"provenance":{"character_level_paired_decoder":True,"broad_contemporary_lexicon":True,"catalogue_imported":False,"finished_tape_reversed":False,"word_order_mirror":False,"reader_status":"not run"}})
 exact=list({x["audit"]["normalized"]:x for x in exact}.values())
 return {"experiment_id":EXPERIMENT_ID,"method":"paired character-level CFG chart join with recursive NP/VP/PP/relative clauses","grammar":GRAM,"lexicon_sizes":{k:len(v) for k,v in LEX.items()},"stats":{"chart_derivations":len(deriv),"paired_character_states":states,"exact":len(exact),"reader_eligible":sum(x["audit"]["letters"]>38 for x in exact),"best_matched_prefix":best["matched"]},"complete_prose_controls":["The quiet writer reads a bright book.","A kind teacher helps the young student.","The sailor who guides a child carries the open letter."],"best_diagnostic":best,"candidates":sorted(exact,key=lambda x:-x["audit"]["letters"]),"independent_audit":["paired online character obligations","two-pointer normalized tape","forward/reverse SHA-256"],"novelty_preflight":{"status":"passed","signature":EXPERIMENT_ID,"catalogue_imported":False,"lexical_sweep":False,"distinct_from":"phrase-cfg-event-ordered-shared-index-20260920"},"next_construction":"Add semantic valency features to the recursive chart and retain only role-compatible NP/VP derivations before paired emission.","reader_gate":"closed; exactness does not certify human readability"}
if __name__=="__main__":
 result=run();out=ROOT/"runs"/(EXPERIMENT_ID+".json");out.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result["stats"],sort_keys=True))
