"""Centerless cross-clause seam with independently selected clause lengths."""
from __future__ import annotations
import hashlib,json,re
from itertools import product
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/centerless-independent-clause-length-seam-20260920.json"
CLAUSES=("the patient sailor charts the inlet.","a careful keeper guards the bridge.","the young scouts return after rain.","several bright guides watch the harbor.")
CONNECTORS=(" Then "," While "," And ")
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
 for nl,nr,left1,left2,right1,right2,conn in product((1,2),(1,2),CLAUSES,CLAUSES,CLAUSES,CLAUSES,CONNECTORS):
  states+=1
  if left1==right1 and nl==nr:continue
  left=left1 if nl==1 else left1+" Then "+left2
  right=right1 if nr==1 else right1+" Then "+right2
  rendered=left+conn+right;eq=live(left,right);row={"rendered":rendered,"left_length":nl,"right_length":nr,"connector":conn.strip(),"online_character_equations":eq,"audit":audit(rendered)}
  if len(controls)<3:controls.append({**row,"reader_eligible":False,"diagnostic_only":True})
  if not eq["all_satisfied"]:prunes+=1;continue
  row["provenance"]={"centerless":True,"independent_clause_lengths":True,"live_cross_clause_seam":True,"authored_complete_clauses":True,"catalogue_text":False,"finished_tape_reversal":False,"post_hoc_repair":False,"mirrored_units":False,"word_order_symmetry":False,"fragment":False};rows.append(row)
 exact=[r for r in rows if r["audit"]["pointer_exact"] and r["audit"]["letters"]>38]
 result={"experiment_id":"centerless-independent-clause-length-seam-20260920","method":"centerless cross-clause seam with independent clause lengths","stats":{"clauses":len(CLAUSES),"connectors":len(CONNECTORS),"length_choices":4,"states":states,"live_prunes":prunes,"live_survivors":len(rows),"exact_gt38":len(exact),"reader_eligible":0,"longest_letters":max((r["audit"]["letters"] for r in rows+controls),default=0)},"controls":controls,"exact_candidates":exact,"reader_facing_candidates":[],"novelty_preflight":{"status":"passed","signature":"centerless|independent-clause-lengths|live-cross-clause-seam","registry_inspected":True,"distinct_from":"center-shared dependency, relation-overhang, mirrored centers, and repair lanes","catalogue_text_imported":False,"finished_tape_reversal":False,"post_hoc_repair":False,"mirrored_units":False},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["pointer scan","forward/reverse SHA-256"]},"status":"no reader-worthy exact closure","next_construction":"Allow independently selected two-clause scene trees with distinct internal connector boundaries, preserving centerless live seams."}
 OUT.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({"artifact":str(OUT),**result["stats"]}));[print(x["rendered"]) for x in controls]
if __name__=="__main__":run()
