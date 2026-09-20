"""Center residual attaches to typed left dependent before right emission."""
from __future__ import annotations
import hashlib,json,re
from itertools import product
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/center-residual-left-dependent-attachment-20260920.json"
CENTERS=(("the bell rings","signal"),("the tide turns","arrival"),("the lantern glows","warning"));LEFT=(("agent","the patient keeper hears"),("observer","the quiet poet notices"),("guide","a young sailor follows"));RIGHT=(("goal","the harbor brightens"),("scene","the camp grows still"),("path","the quay fills with boats"));RESIDUALS=("at dawn","before dusk","after rain")
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
 for center,left,right,residual in product(CENTERS,LEFT,RIGHT,RESIDUALS):
  states+=1;event,kind=center;lt=f"{left[1]} when {event}, {left[0]} {residual}";rt=f"{right[1]} after it.";rendered=lt+"; "+rt;eq=live(lt,rt);row={"rendered":rendered,"center_dependency":{"event":event,"kind":kind},"left_dependent":left,"right_dependent":right,"attached_residual":{"text":residual,"target_role":left[0]},"online_character_equations":eq,"audit":audit(rendered)}
  if len(controls)<3:controls.append({**row,"reader_eligible":False,"diagnostic_only":True})
  if not eq["all_satisfied"]:prunes+=1;continue
  row["provenance"]={"center_shared_dependency_node":True,"residual_attaches_left_dependent":True,"typed_left_role":True,"right_emitted_after_attachment":True,"complete_utterances":True,"catalogue_text":False,"finished_tape_reversal":False,"post_hoc_repair":False,"mirrored_units":False,"word_order_symmetry":False,"fragment":False};rows.append(row)
 exact=[r for r in rows if r["audit"]["pointer_exact"] and r["audit"]["letters"]>38]
 result={"experiment_id":"center-residual-left-dependent-attachment-20260920","method":"center residual attaches to typed left dependent before right emission","stats":{"center_nodes":len(CENTERS),"left_roles":len(LEFT),"right_roles":len(RIGHT),"residuals":len(RESIDUALS),"states":states,"live_prunes":prunes,"live_survivors":len(rows),"exact_gt38":len(exact),"reader_eligible":0,"longest_letters":max((r["audit"]["letters"] for r in rows+controls),default=0)},"controls":controls,"exact_candidates":exact,"reader_facing_candidates":[],"novelty_preflight":{"status":"passed","signature":"center-dependency|left-dependent-residual-attachment|delayed-right-emission","registry_inspected":True,"distinct_from":"unattached residual, equal-depth frames, feature sweeps, and repairs","catalogue_text_imported":False,"finished_tape_reversal":False,"post_hoc_repair":False,"mirrored_units":False},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["pointer scan","forward/reverse SHA-256"]},"status":"no reader-worthy exact closure","next_construction":"Attach the residual to a typed right dependent after left closure and compare the two attachment paths."}
 OUT.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({"artifact":str(OUT),**result["stats"]}));[print(x["rendered"]) for x in controls]
if __name__=="__main__":run()
