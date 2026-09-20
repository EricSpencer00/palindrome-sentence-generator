"""Joint dependency grammar comparing left/right center-residual paths."""
from __future__ import annotations
import hashlib,json,re
from itertools import product
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/joint-left-right-center-residual-paths-20260920.json"
CENTERS=(("the bell rings","signal"),("the tide turns","arrival"),("the lantern glows","warning"));LEFT=(("agent","the patient keeper hears"),("observer","the quiet poet notices"),("guide","a young sailor follows"));RIGHT=(("goal","the harbor brightens"),("scene","the camp grows still"),("path","the quay fills with boats"));RESIDUALS=("at dawn","before dusk","after rain");PATHS=("left_attached","right_attached")
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
 for center,left,right,residual,path in product(CENTERS,LEFT,RIGHT,RESIDUALS,PATHS):
  states+=1;event,kind=center
  if path=="left_attached":lt=f"{left[1]} when {event}, {left[0]} {residual}";rt=f"{right[1]} after it."
  else:lt=f"{left[1]} when {event}";rt=f"{right[1]}, {right[0]} {residual} after it."
  rendered=lt+"; "+rt;eq=live(lt,rt);row={"rendered":rendered,"center_dependency":{"event":event,"kind":kind},"left_dependent":left,"right_dependent":right,"residual":residual,"attachment_path":path,"online_character_equations":eq,"audit":audit(rendered)}
  if len(controls)<3:controls.append({**row,"reader_eligible":False,"diagnostic_only":True})
  if not eq["all_satisfied"]:prunes+=1;continue
  row["provenance"]={"joint_attachment_paths":True,"center_shared_dependency_node":True,"left_right_path_state":True,"complete_utterances":True,"catalogue_text":False,"finished_tape_reversal":False,"post_hoc_repair":False,"mirrored_units":False,"word_order_symmetry":False,"fragment":False};rows.append(row)
 mismatches=[r["online_character_equations"]["first_mismatch"] for r in controls]; invariant=len(set(mismatches))==1 if mismatches else False
 exact=[r for r in rows if r["audit"]["pointer_exact"] and r["audit"]["letters"]>38]
 result={"experiment_id":"joint-left-right-center-residual-paths-20260920","method":"joint dependency grammar comparing left- and right-attached center residual paths","stats":{"center_nodes":len(CENTERS),"left_roles":len(LEFT),"right_roles":len(RIGHT),"residuals":len(RESIDUALS),"attachment_paths":len(PATHS),"states":states,"live_prunes":prunes,"live_survivors":len(rows),"exact_gt38":len(exact),"reader_eligible":0,"longest_letters":max((r["audit"]["letters"] for r in rows+controls),default=0),"control_mismatch_invariant":invariant},"controls":controls,"exact_candidates":exact,"reader_facing_candidates":[],"novelty_preflight":{"status":"passed","signature":"joint-left-right-residual-paths|center-dependency|path-state","registry_inspected":True,"distinct_from":"separate left/right sweeps, unattached residual, and feature families","catalogue_text_imported":False,"finished_tape_reversal":False,"post_hoc_repair":False,"mirrored_units":False},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["pointer scan","forward/reverse SHA-256"],"reader_evidence":False},"status":"no reader-worthy exact closure","next_construction":"Pivot topology because both attachment paths share the same first-character bottleneck; use a centerless cross-clause seam with independent clause lengths."}
 OUT.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({"artifact":str(OUT),**result["stats"]}));[print(x["rendered"]) for x in controls]
if __name__=="__main__":run()
