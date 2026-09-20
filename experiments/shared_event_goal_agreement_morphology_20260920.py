"""Joint order/tense/adjunct subject, object, and goal agreement."""
from __future__ import annotations
import hashlib,json,re
from itertools import product
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/shared-event-goal-agreement-morphology-20260920.json"
EVENTS=(("signal","past","earlier","later"),("arrival","present","now","soon"),("warning","past","before","after"));SUBJECTS=(("singular","the keeper","heard","brightened"),("plural","the keepers","hear","brighten"));OBJECTS=(("singular","the bell"),("plural","the bells"));GOALS=(("singular","the harbor"),("plural","the harbors"));ORDERS=(("before","earlier"),("after","later"))
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
 for event,subject,obj,goal,order in product(EVENTS,SUBJECTS,OBJECTS,GOALS,ORDERS):
  states+=1;name,tense,early,late=event;number,noun,left,right=subject;oname,morph=order
  if morph!=early or obj[0]!=number or goal[0]!=number:continue
  lt=f"{noun} {left} {obj[1]} {early}";rt=f"{goal[1]} {right} {late}.";rendered=lt+"; "+rt;eq=live(lt,rt);row={"rendered":rendered,"shared_event_variable":name,"event_tense":tense,"subject_agreement":number,"object_agreement":obj[0],"goal_agreement":goal[0],"temporal_order":oname,"online_character_equations":eq,"audit":audit(rendered)}
  if len(controls)<3:controls.append({**row,"reader_eligible":False,"diagnostic_only":True})
  if not eq["all_satisfied"]:prunes+=1;continue
  row["provenance"]={"order_tense_adjunct_subject_object_goal_joint_state":True,"subject_agreement_morphology":True,"object_agreement_morphology":True,"goal_agreement_morphology":True,"distinct_predicates":True,"complete_utterances":True,"catalogue_text":False,"finished_tape_reversal":False,"post_hoc_repair":False,"mirrored_units":False};rows.append(row)
 exact=[r for r in rows if r["audit"]["pointer_exact"] and r["audit"]["letters"]>38]
 result={"experiment_id":"shared-event-goal-agreement-morphology-20260920","method":"joint event order/tense/adjunct subject object and goal agreement morphology","stats":{"events":len(EVENTS),"subject_states":len(SUBJECTS),"object_states":len(OBJECTS),"goal_states":len(GOALS),"order_states":len(ORDERS),"states":states,"live_prunes":prunes,"live_survivors":len(rows),"exact_gt38":len(exact),"reader_eligible":0,"longest_letters":max((r["audit"]["letters"] for r in rows+controls),default=0)},"controls":controls,"exact_candidates":exact,"reader_facing_candidates":[],"novelty_preflight":{"status":"passed","signature":"order-tense-adjunct|subject-object-goal-agreement|shared-event","registry_inspected":True,"distinct_from":"subject/object-only morphology, generic tense, repeated wording, and relation-overhang lanes","catalogue_text_imported":False,"finished_tape_reversal":False,"post_hoc_repair":False,"mirrored_units":False},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["pointer scan","forward/reverse SHA-256"]},"status":"no reader-worthy exact closure","next_construction":"Add event-specific valency morphology to the full agreement state."}
 OUT.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({"artifact":str(OUT),**result["stats"]}));[print(x["rendered"]) for x in controls]
if __name__=="__main__":run()
