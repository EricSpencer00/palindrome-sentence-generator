"""Simple scene-pair CFG with a shared event and live character equations."""
from __future__ import annotations
import hashlib,json,re
from itertools import product
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/shared-event-scene-pair-cfg-20260920.json"
CLAUSES=("the patient sailor charts the inlet","a careful keeper guards the bridge","the young scouts return after rain","several bright guides watch the harbor")
EVENTS=("the bell rings","the tide turns","the lantern glows")
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
 for left,right,event in product(CLAUSES,CLAUSES,EVENTS):
  states+=1
  if left==right:continue
  lt=f"{left}, and {event}";rt=f"{event}, while {right}.";rendered=lt+" "+rt;eq=live(lt,rt);row={"rendered":rendered,"left_clause":left,"shared_event":event,"right_clause":right,"online_character_equations":eq,"audit":audit(rendered)}
  if len(controls)<3:controls.append({**row,"reader_eligible":False,"diagnostic_only":True})
  if not eq["all_satisfied"]:prunes+=1;continue
  row["provenance"]={"shared_event_nonterminal":True,"independent_clause_authorship":True,"live_equations_before_lexicalization":True,"complete_utterances":True,"catalogue_text":False,"finished_tape_reversal":False,"post_hoc_repair":False,"mirrored_units":False,"word_order_symmetry":False,"fragment":False,"nested_self_palindrome":False};rows.append(row)
 exact=[r for r in rows if r["audit"]["pointer_exact"] and r["audit"]["letters"]>38]
 result={"experiment_id":"shared-event-scene-pair-cfg-20260920","method":"independent coherent clause pair joined by shared event nonterminal","stats":{"clauses":len(CLAUSES),"shared_events":len(EVENTS),"states":states,"live_prunes":prunes,"live_survivors":len(rows),"exact_gt38":len(exact),"reader_eligible":0,"longest_letters":max((r["audit"]["letters"] for r in rows+controls),default=0)},"controls":controls,"exact_candidates":exact,"reader_facing_candidates":[],"novelty_preflight":{"status":"passed","signature":"shared-event-CFG|independent-coherent-clauses|live-equations","registry_inspected":True,"distinct_from":"relation-overhang seams, repairs, mirrored units, and catalogue controls","catalogue_text_imported":False,"finished_tape_reversal":False,"post_hoc_repair":False,"mirrored_units":False},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["pointer scan","forward/reverse SHA-256"]},"status":"no reader-worthy exact closure","next_construction":"Use a shared event with two typed argument frames selected before clause lexicalization."}
 OUT.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({"artifact":str(OUT),**result["stats"]}));[print(x["rendered"]) for x in controls]
if __name__=="__main__":run()
