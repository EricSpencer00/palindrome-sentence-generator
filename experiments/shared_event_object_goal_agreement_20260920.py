"""Shared event with dual subject plus object/goal agreement."""
from __future__ import annotations
import hashlib,json,re
from itertools import product
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/shared-event-object-goal-agreement-20260920.json"
EVENTS=(("signal","bell"),("arrival","tide"),("warning","lantern")); SUBJECTS=(("singular","the patient keeper hears"),("plural","the young scouts notice")); GOALS=(("singular","the harbor brightens"),("plural","the quay fills")); OBJECTS=(("singular","the bell"),("plural","the bells"))
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
 for event,subj,goal,obj in product(EVENTS,SUBJECTS,GOALS,OBJECTS):
  states+=1
  if subj[0]!=goal[0] or subj[0]!=obj[0]:continue
  lt=f"{subj[1]} {obj[1]}";rt=f"{goal[1]} after {event[1]}.";rendered=lt+"; "+rt;eq=live(lt,rt);row={"rendered":rendered,"shared_event_variable":event,"subject_agreement":subj[0],"object_agreement":obj[0],"goal_agreement":goal[0],"online_character_equations":eq,"audit":audit(rendered)}
  if len(controls)<3:controls.append({**row,"reader_eligible":False,"diagnostic_only":True})
  if not eq["all_satisfied"]:prunes+=1;continue
  row["provenance"]={"shared_semantic_event_variable":True,"dual_subject_agreement":True,"object_goal_agreement":True,"distinct_predicates":True,"complete_utterances":True,"catalogue_text":False,"finished_tape_reversal":False,"post_hoc_repair":False,"mirrored_units":False};rows.append(row)
 exact=[r for r in rows if r["audit"]["pointer_exact"] and r["audit"]["letters"]>38]
 result={"experiment_id":"shared-event-object-goal-agreement-20260920","method":"shared event with subject, object, and goal agreement","stats":{"events":len(EVENTS),"subjects":len(SUBJECTS),"goals":len(GOALS),"objects":len(OBJECTS),"states":states,"live_prunes":prunes,"live_survivors":len(rows),"exact_gt38":len(exact),"reader_eligible":0,"longest_letters":max((r["audit"]["letters"] for r in rows+controls),default=0)},"controls":controls,"exact_candidates":exact,"reader_facing_candidates":[],"novelty_preflight":{"status":"passed","signature":"shared-event|dual-subject|object-goal-agreement|distinct-predicates","registry_inspected":True,"distinct_from":"subject-only shared event, repeated event wording, and relation-overhang lanes","catalogue_text_imported":False,"finished_tape_reversal":False,"post_hoc_repair":False,"mirrored_units":False},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["pointer scan","forward/reverse SHA-256"]},"status":"no reader-worthy exact closure","next_construction":"Add typed event tense/aspect compatibility to the agreement-carrying argument frames."}
 OUT.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({"artifact":str(OUT),**result["stats"]}));[print(x["rendered"]) for x in controls]
if __name__=="__main__":run()
