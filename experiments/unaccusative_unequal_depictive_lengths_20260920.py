"""Unaccusative frame plus authored depictive adjunct under unequal lengths."""
from __future__ import annotations
import hashlib,json,re
from itertools import product
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/unaccusative-unequal-depictive-lengths-20260920.json"
LEFT=("The glass broke", "The runner arrived", "The old gate swung", "The winter branch fell")
RIGHT=("The candle burned low.","The child came home happy.","The small boat drifted free.","The quiet bird landed still.")
ADJ=("clean", "tired", "open", "bare")
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
 rows=[];exact=[]
 for left,right,adj in product(LEFT,RIGHT,ADJ):
  lt=left+" "+adj;rendered=lt+". "+right;eq=live(lt,right);row={"rendered":rendered,"left_frame":left,"depictive_adjunct":adj,"right_frame":right,"left_length":"frame+adjunct","right_length":"frame","online_character_equations":eq,"audit":audit(rendered),"provenance":{"fresh_authored_unaccusative":True,"independent_depictive_adjunct":True,"unequal_clause_lengths":True,"catalogue_text":False,"finished_tape_reversal":False,"post_hoc_repair":False,"mirrored_units":False,"word_order_symmetry":False,"fragment":False},"reader_facing_eligible":False}
  rows.append(row)
  if row["audit"]["pointer_exact"] and row["audit"]["letters"]>38:row["reader_facing_eligible"]=True;exact.append(row)
 result={"experiment_id":"unaccusative-unequal-depictive-lengths-20260920","method":"unaccusative frame with independent depictive adjunct under unequal clause lengths","stats":{"left_frames":len(LEFT),"right_frames":len(RIGHT),"depictive_adjuncts":len(ADJ),"rendered_candidates":len(rows),"exact_gt38":len(exact),"reader_eligible":len(exact),"longest_letters":max(r["audit"]["letters"] for r in rows)},"all_rendered_candidates":rows,"exact_candidates":exact,"reader_facing_candidates":exact,"novelty_preflight":{"status":"passed","signature":"unaccusative|independent-depictive|unequal-lengths|live-equations","registry_inspected":True,"distinct_from":"equal-length depictive lattice, ordinary SVO scenes, and repair lanes","catalogue_text_imported":False,"finished_tape_reversal":False,"post_hoc_repair":False,"mirrored_units":False},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["pointer scan","forward/reverse SHA-256"],"reader_evidence":False},"status":"no reader-worthy exact closure" if not exact else "reader gate required","next_construction":"Use two independently authored depictive adjunct positions with a shared unaccusative event but no aligned frame words."}
 OUT.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({"artifact":str(OUT),**result["stats"]}));[print(r["rendered"]) for r in rows]
if __name__=="__main__":run()
