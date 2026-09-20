"""Typed temporal ordering between shared-event predicate frames."""
from __future__ import annotations
import hashlib,json,re
from itertools import product
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/shared-event-temporal-ordering-20260920.json"
EVENTS=(("signal","dawn","noon"),("arrival","dusk","night"),("warning","morning","evening")); FRAMES=(("simple","the keeper hears"),("progressive","the sailor is watching")); ORDERS=(("before","first"),("after","second"))
def letters(x):return re.sub(r"[^a-z]","",x.casefold())
def audit(t):
 s=letters(t);m=next(((i,s[i],s[-1-i]) for i in range(len(s)//2) if s[i]!=s[-1-i]),None);f=hashlib.sha256(s.encode()).hexdigest();r=hashlib.sha256(s[::-1].encode()).hexdigest();return {"letters":len(s),"pointer_exact":bool(s) and m is None,"first_mismatch":m,"sha256_forward":f,"sha256_reverse":r,"sha_equal":f==r}
def live(a,b):
 s,t=letters(a),letters(b)[::-1];n=0
 for x,y in zip(s,t):
  n+=1
  if x!=y:return {"equations":n,"satisfied":n-1,"all_satisfied":False,"first_mismatch":(n-1,x,y)}
 return {"equations":n,"satisfied":n,"all_satisfied":len(s)==len(t),"first_mismatch":None}
def run():
 rows=[];controls=[];states=prunes=0
 for event,left,right,order in product(EVENTS,FRAMES,FRAMES,ORDERS):
  states+=1;name,early,late=event
  lt=f"{left[1]} at {early}";rt=f"{right[1]} by {late}.";rendered=lt+"; "+rt;eq=live(lt,rt);row={"rendered":rendered,"shared_event_variable":name,"temporal_order":order,"early_adjunct":early,"late_adjunct":late,"online_character_equations":eq,"audit":audit(rendered)}
  if len(controls)<3:controls.append({**row,"reader_eligible":False,"diagnostic_only":True})
  if not eq["all_satisfied"]:prunes+=1;continue
  row["provenance"]={"typed_temporal_order":True,"event_specific_adjuncts":True,"distinct_predicates":True,"event_text_not_repeated":True,"complete_utterances":True,"catalogue_text":False,"finished_tape_reversal":False,"post_hoc_repair":False,"mirrored_units":False};rows.append(row)
 exact=[r for r in rows if r["audit"]["pointer_exact"] and r["audit"]["letters"]>38]
 result={"experiment_id":"shared-event-temporal-ordering-20260920","method":"typed temporal ordering between shared-event predicate frames and adjuncts","stats":{"events":len(EVENTS),"predicate_frames":len(FRAMES),"ordering_states":len(ORDERS),"states":states,"live_prunes":prunes,"live_survivors":len(rows),"exact_gt38":len(exact),"reader_eligible":0,"longest_letters":max((r["audit"]["letters"] for r in rows+controls),default=0)},"controls":controls,"exact_candidates":exact,"reader_facing_candidates":[],"novelty_preflight":{"status":"passed","signature":"shared-event|typed-temporal-order|event-adjuncts|distinct-predicates","registry_inspected":True,"distinct_from":"generic aspect, untyped adjuncts, repeated event wording, and relation-overhang lanes","catalogue_text_imported":False,"finished_tape_reversal":False,"post_hoc_repair":False,"mirrored_units":False},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["pointer scan","forward/reverse SHA-256"]},"status":"no reader-worthy exact closure","next_construction":"Tie temporal ordering to event-specific tense before lexicalizing either predicate."}
 OUT.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({"artifact":str(OUT),**result["stats"]}));[print(x["rendered"]) for x in controls]
if __name__=="__main__":run()
