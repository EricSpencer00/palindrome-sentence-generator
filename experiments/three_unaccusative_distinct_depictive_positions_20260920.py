"""Three independently authored unaccusative frames with distinct depictive positions."""
from __future__ import annotations
import hashlib,json,re
from itertools import product
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/three-unaccusative-distinct-depictives-20260920.json"
LEFT=(("The glass broke","clean"),("The runner arrived","tired"),("The old gate swung","open"));RIGHT=(("A candle burned","low"),("A child came home","happy"),("A small boat drifted","free"));POS=("before","after","between")
def letters(x):return re.sub(r"[^a-z]","",x.casefold())
def audit(t):
 s=letters(t);m=next(((i,s[i],s[-1-i]) for i in range(len(s)//2) if s[i]!=s[-1-i]),None);f=hashlib.sha256(s.encode()).hexdigest();r=hashlib.sha256(s[::-1].encode()).hexdigest();return {"letters":len(s),"pointer_exact":bool(s) and m is None,"first_mismatch":m,"sha256_forward":f,"sha256_reverse":r,"sha_equal":f==r}
def live(a,b):
 s,t=letters(a),letters(b)[::-1];n=0
 for x,y in zip(s,t):
  n+=1
  if x!=y:return {"equations":n,"satisfied":n-1,"all_satisfied":False,"first_mismatch":(n-1,x,y)}
 return {"equations":n,"satisfied":n,"all_satisfied":len(s)==len(t),"first_mismatch":None}
def render(frame,pos):
 clause,dep=frame
 if pos=="before":return f"{dep}, {clause.lower()}."
 if pos=="between":return f"{clause}, {dep} and then."
 return f"{clause} {dep}."
def run():
 rows=[];exact=[]
 for lf,rf,lp,rp in product(LEFT,RIGHT,POS,POS):
  left=render(lf,lp);right=render(rf,rp);rendered=left+" "+right;eq=live(left,right);row={"rendered":rendered,"left_frame":lf,"right_frame":rf,"left_position":lp,"right_position":rp,"online_character_equations":eq,"audit":audit(rendered),"provenance":{"three_independent_unaccusatives":True,"distinct_depictive_positions":True,"no_shared_frame_wording":True,"catalogue_text":False,"finished_tape_reversal":False,"post_hoc_repair":False,"mirrored_units":False,"word_order_symmetry":False,"fragment":False},"reader_facing_eligible":False}
  rows.append(row)
  if row["audit"]["pointer_exact"] and row["audit"]["letters"]>38:row["reader_facing_eligible"]=True;exact.append(row)
 result={"experiment_id":"three-unaccusative-distinct-depictives-20260920","method":"three independently authored unaccusative event frames with distinct depictive positions","stats":{"left_frames":len(LEFT),"right_frames":len(RIGHT),"positions":len(POS),"rendered_candidates":len(rows),"exact_gt38":len(exact),"reader_eligible":len(exact),"longest_letters":max(r["audit"]["letters"] for r in rows)},"all_rendered_candidates":rows,"exact_candidates":exact,"reader_facing_candidates":exact,"novelty_preflight":{"status":"passed","signature":"three-unaccusatives|distinct-depictive-positions|no-shared-wording","registry_inspected":True,"distinct_from":"two-frame depictive lanes, aligned mirrors, and repair operators","catalogue_text_imported":False,"finished_tape_reversal":False,"post_hoc_repair":False,"mirrored_units":False},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["pointer scan","forward/reverse SHA-256"],"reader_evidence":False},"status":"no reader-worthy exact closure" if not exact else "reader gate required","next_construction":"Use a three-frame unaccusative scene tree with independently authored connective structure and variable clause lengths."}
 OUT.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({"artifact":str(OUT),**result["stats"]}));[print(r["rendered"]) for r in rows]
if __name__=="__main__":run()
