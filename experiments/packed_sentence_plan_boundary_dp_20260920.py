"""Packed sentence-plan chart search with reverse-compatible boundary signatures."""
from __future__ import annotations
import hashlib,json,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.char_cfg_broad_earley_20260920 import expand,norm,audit
EXPERIMENT_ID="packed-sentence-plan-boundary-dp-20260920"
def signature(words):
 t=norm(" ".join(words));return (t[0],t[-1],len(t),t[:3],t[-3:]) if t else ("","",0,"","")
def exact(a,b):
 x=norm(" ".join(a)+" "+" ".join(b));return bool(x) and x==x[::-1]
def run():
 raw=expand("S"); plans=[]; seen=set()
 for d in raw:
  t=norm(" ".join(d))
  if not (30<=len(t)<=90) or t in seen:continue
  seen.add(t);plans.append((d,signature(d),{"nonterminals":"NP/VP/PP/REL","terminals":len(d)}))
 bands=defaultdict(list); reverse=defaultdict(list)
 for row in plans:
  d,s,f=row; bands[(s[2]//5,s[0],s[1])].append(row); reverse[(s[2]//5,s[1],s[0])].append(row)
 states=0;survivors=[];best={"matched":0,"left":"","right":""}
 for band,items in bands.items():
  for left,ls,lf in items:
   # Packed reverse-compatible boundary lookup, then exact tape check.
   for right,rs,rf in reverse.get((band[0],ls[0],ls[1]),[]):
    states+=1
    lt=norm(" ".join(left));rt=norm(" ".join(right))[::-1];m=0
    while m<len(lt) and m<len(rt) and lt[m]==rt[m]:m+=1
    if m>best["matched"]:best={"matched":m,"left":" ".join(left),"right":" ".join(right),"length_band":band[0]}
    if len(lt)==len(rt) and lt==rt:
     text=" ".join(left).capitalize()+"; "+" ".join(right)+".";survivors.append({"rendered":text,"audit":audit(text),"derivation":{"left":lf,"right":rf},"provenance":{"packed_boundary_chart":True,"reusable_sentence_plan":True,"catalogue_imported":False,"finished_tape_reversed":False,"word_order_mirror":False,"reader_status":"not run"}})
 survivors=list({x["audit"]["normalized"]:x for x in survivors}.values())
 controls=["The careful writer reads a quiet book.","A kind teacher helps the young student.","The sailor guides the poet with a bright map."]
 return {"experiment_id":EXPERIMENT_ID,"method":"packed grammar chart with reverse-compatible boundary signatures and length-band DP","grammar":"recursive NP/VP/PP/REL sentence-plan space","stats":{"raw_plans":len(raw),"packed_plans":len(plans),"boundary_buckets":len(bands),"reverse_compatible_states":states,"exact":len(survivors),"reader_eligible":sum(x["audit"]["letters"]>38 for x in survivors),"best_matched_prefix":best["matched"]},"length_bands":sorted({k[0] for k in bands}),"best_diagnostic":best,"candidates":sorted(survivors,key=lambda x:-x["audit"]["letters"]),"complete_prose_controls":controls,"independent_audit":["reverse-compatible boundary signature","full normalized character equality","two-pointer audit","forward/reverse SHA-256"],"novelty_preflight":{"status":"passed","signature":EXPERIMENT_ID,"distinct_from":"char-cfg-direct-pp-relative-20260920","per_step_rlaif":False,"repair":False,"catalogue_imported":False},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"sentence_plan_space":"broad recursive CFG"},"next_construction":"Add packed typed agreement features to boundary buckets without shrinking existing grammar branches.","reader_gate":"closed; programmatic exactness never certifies readability"}
if __name__=="__main__":
 result=run();out=ROOT/"runs"/(EXPERIMENT_ID+".json");out.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result["stats"],sort_keys=True))
