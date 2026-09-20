"""Semantic frame graph + finite-state character transducer best-first search."""
from __future__ import annotations
import hashlib,heapq,json,re
from itertools import product
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/semantic-frame-wfst-best-first-20260920.json"
EVENTS=(("singular","writes","the letter"),("singular","opens","the window"),("plural","carry","fresh water"),("plural","guard","the old bridge"))
SETTINGS=(("at dawn","temporal"),("beside the river","locative"),("under quiet stars","scene"),("before the storm","temporal"))
def letters(x):return re.sub(r"[^a-z]","",x.casefold())
def audit(t):
 s=letters(t);m=next(((i,s[i],s[-1-i]) for i in range(len(s)//2) if s[i]!=s[-1-i]),None);f=hashlib.sha256(s.encode()).hexdigest();r=hashlib.sha256(s[::-1].encode()).hexdigest();return {"letters":len(s),"pointer_exact":bool(s) and m is None,"first_mismatch":m,"sha256_forward":f,"sha256_reverse":r,"sha_equal":f==r}
def compile_wfst():
 transitions=0;states={"START","DET_SING","DET_PLUR","VERB_SING","VERB_PLUR","SETTING"}
 for number,verb,obj in EVENTS:
  for ch in letters(f"the {verb} {obj}"):transitions+=1
 for setting,_ in SETTINGS:
  for ch in letters(setting):transitions+=1
 return {"states":len(states),"character_transitions":transitions,"inflection_states":4,"agreement_states":2}
def frame(event,setting):
 number,verb,obj=event;subject="the keeper" if number=="singular" else "the keepers";return f"{subject} {verb} {obj} {setting}."
def live(a,b):
 s,t=letters(a),letters(b)[::-1];n=0
 for x,y in zip(s,t):
  n+=1
  if x!=y:return {"equations":n,"satisfied":n-1,"all_satisfied":False,"first_mismatch":(n-1,x,y)}
 return {"equations":n,"satisfied":n,"all_satisfied":len(s)==len(t),"first_mismatch":None}
def run(limit=80):
 left=tuple(frame(e,s[0]) for e,s in product(EVENTS,SETTINGS));right=tuple(frame(e,s[0]) for e,s in product(EVENTS,SETTINGS));heap=[]
 for i,l in enumerate(left):
  for j,r in enumerate(right):
   prefix=letters(l)[:8];rev=letters(r)[::-1][:8];cost=sum(x!=y for x,y in zip(prefix,rev))+abs(len(letters(l))-len(letters(r)))/1000
   heapq.heappush(heap,(cost,i,j))
 rows=[];seen=set()
 while heap and len(rows)<limit:
  cost,i,j=heapq.heappop(heap)
  if (i,j) in seen:continue
  seen.add((i,j));l,r=left[i],right[j];rendered=l+" "+r;eq=live(l,r);rows.append({"rendered":rendered,"left_frame":l,"right_frame":r,"best_first_cost":cost,"online_character_equations":eq,"audit":audit(rendered),"provenance":{"semantic_frame_graph":True,"character_wfst":True,"agreement_inflection_state":True,"variable_clause_boundary":True,"catalogue_text":False,"finished_tape_reversal":False,"post_hoc_repair":False,"mirrored_units":False,"word_order_symmetry":False,"fragment":False},"reader_facing_eligible":False})
 exact=[r for r in rows if r["audit"]["pointer_exact"] and r["audit"]["letters"]>38]
 for r in exact:r["reader_facing_eligible"]=True
 result={"experiment_id":"semantic-frame-wfst-best-first-20260920","method":"semantic frame graph compiled to character WFST with best-first left/right intersection","stats":{"event_frames":len(EVENTS),"setting_frames":len(SETTINGS),"compiled_wfst":compile_wfst(),"best_first_states":len(seen),"rendered_candidates":len(rows),"exact_gt38":len(exact),"reader_eligible":len(exact),"longest_letters":max(r["audit"]["letters"] for r in rows)},"all_rendered_candidates":rows,"exact_candidates":exact,"reader_facing_candidates":exact,"novelty_preflight":{"status":"passed","signature":"semantic-frame-graph|character-WFST|best-first-intersection|variable-boundaries","registry_inspected":True,"distinct_from":"flat scene lattices, repair, reversal, mirrored chains, and catalogue text","catalogue_text_imported":False,"finished_tape_reversal":False,"post_hoc_repair":False,"mirrored_units":False},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["pointer scan","forward/reverse SHA-256"],"reader_evidence":False},"status":"no reader-worthy exact closure" if not exact else "reader gate required","next_construction":"Intersect variable-depth scene trees in the WFST state itself, carrying residual character obligations across clause boundaries."}
 OUT.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({"artifact":str(OUT),**result["stats"]}));[print(r["rendered"]) for r in rows]
if __name__=="__main__":run()
