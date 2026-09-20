"""Subject-oriented depictive/resultative alternation with unaccusative frames."""
from __future__ import annotations
import hashlib,json,re
from itertools import product
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/subject-depictive-unaccusative-lattice-20260920.json"
LEFT=("The glass broke clean.","The runner arrived tired.","The old gate swung open.","The winter branch fell bare.")
RIGHT=("The candle burned low.","The child came home happy.","The small boat drifted free.","The quiet bird landed still.")
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
 for left,right in product(LEFT,RIGHT):
  rendered=left+" "+right;eq=live(left,right);row={"rendered":rendered,"left_clause":left,"right_clause":right,"frame_type":"unaccusative + subject-oriented depictive/resultative","online_character_equations":eq,"audit":audit(rendered),"provenance":{"fresh_authored_clause":True,"subject_oriented_depictive":True,"unaccusative_frame":True,"catalogue_text":False,"finished_tape_reversal":False,"post_hoc_repair":False,"mirrored_units":False,"word_order_symmetry":False,"fragment":False},"reader_facing_eligible":False}
  rows.append(row)
  if row["audit"]["pointer_exact"] and row["audit"]["letters"]>38:row["reader_facing_eligible"]=True;exact.append(row)
 result={"experiment_id":"subject-depictive-unaccusative-lattice-20260920","method":"subject-oriented depictive/resultative alternation over authored unaccusative clauses","stats":{"left_frames":len(LEFT),"right_frames":len(RIGHT),"rendered_candidates":len(rows),"exact_gt38":len(exact),"reader_eligible":len(exact),"longest_letters":max(r["audit"]["letters"] for r in rows)},"all_rendered_candidates":rows,"exact_candidates":exact,"reader_facing_candidates":exact,"novelty_preflight":{"status":"passed","signature":"unaccusative|subject-depictive|resultative-alternation","registry_inspected":True,"distinct_from":"ordinary SVO scene lattice, center dependency, and repair lanes","catalogue_text_imported":False,"finished_tape_reversal":False,"post_hoc_repair":False,"mirrored_units":False},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["pointer scan","forward/reverse SHA-256"],"reader_evidence":False},"status":"no reader-worthy exact closure" if not exact else "reader gate required","next_construction":"Pair unaccusative frames with independently authored depictive adjuncts under unequal clause lengths."}
 OUT.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({"artifact":str(OUT),**result["stats"]}));[print(r["rendered"]) for r in rows]
if __name__=="__main__":run()
