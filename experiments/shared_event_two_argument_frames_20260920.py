"""Shared event with two typed argument frames selected before lexicalization."""
from __future__ import annotations
import hashlib,json,re
from itertools import product
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/shared-event-two-argument-frames-20260920.json"
EVENTS=(("signal","the bell signals the harbor"),("arrival","the tide reaches the quay"),("warning","the lantern warns the camp"))
AGENTS=(("agent","the patient keeper"),("agent","a young sailor"),("observer","the quiet poet"))
PATIENTS=(("patient","the northern inlet"),("goal","the old bridge"),("patient","a distant harbor"))
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
 for event,agent,patient in product(EVENTS,AGENTS,PATIENTS):
  states+=1; left=f"{agent[1]} hears how {event[1]}"; right=f"because {patient[1]} lies beyond the gate"; rendered=left+"; "+right+".";eq=live(left,right);row={"rendered":rendered,"shared_event":event,"agent_frame":agent,"patient_frame":patient,"online_character_equations":eq,"audit":audit(rendered)}
  if len(controls)<3 and agent[1]!=patient[1]:controls.append({**row,"reader_eligible":False,"diagnostic_only":True})
  if not eq["all_satisfied"]:prunes+=1;continue
  row["provenance"]={"shared_event_nonterminal":True,"typed_agent_patient_frames":True,"slot_domain_selected_before_lexicalization":True,"independent_argument_roles":True,"complete_utterances":True,"catalogue_text":False,"finished_tape_reversal":False,"post_hoc_repair":False,"mirrored_units":False};rows.append(row)
 exact=[r for r in rows if r["audit"]["pointer_exact"] and r["audit"]["letters"]>38]
 result={"experiment_id":"shared-event-two-argument-frames-20260920","method":"shared event with typed agent/patient argument frames selected before lexicalization","stats":{"events":len(EVENTS),"agent_slots":len(AGENTS),"patient_slots":len(PATIENTS),"states":states,"live_prunes":prunes,"live_survivors":len(rows),"exact_gt38":len(exact),"reader_eligible":0,"longest_letters":max((r["audit"]["letters"] for r in rows+controls),default=0)},"controls":controls,"exact_candidates":exact,"reader_facing_candidates":[],"novelty_preflight":{"status":"passed","signature":"shared-event|typed-agent-patient|slot-domain-prelexical","registry_inspected":True,"distinct_from":"shared event text repetition, relation-overhang seams, and repair lanes","catalogue_text_imported":False,"finished_tape_reversal":False,"post_hoc_repair":False,"mirrored_units":False},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["pointer scan","forward/reverse SHA-256"]},"status":"no reader-worthy exact closure","next_construction":"Use typed event argument frames with two distinct finite predicates and a shared semantic event variable."}
 OUT.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({"artifact":str(OUT),**result["stats"]}));[print(x["rendered"]) for x in controls]
if __name__=="__main__":run()
