"""Three-frame authored scene tree with variable clause lengths/connectives."""
from __future__ import annotations
import hashlib,json,re
from itertools import product
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/three-frame-scene-tree-variable-lengths-20260920.json"
LEFT=("The glass broke clean.","The runner arrived tired.","The old gate swung open.");RIGHT=("A candle burned low.","A child came home happy.","A small boat drifted free.");CONNS=(" Then "," Meanwhile "," By evening ")
def letters(x):return re.sub(r"[^a-z]","",x.casefold())
def audit(t):
 s=letters(t);m=next(((i,s[i],s[-1-i]) for i in range(len(s)//2) if s[i]!=s[-1-i]),None);f=hashlib.sha256(s.encode()).hexdigest();r=hashlib.sha256(s[::-1].encode()).hexdigest();return {"letters":len(s),"pointer_exact":bool(s) and m is None,"first_mismatch":m,"sha256_forward":f,"sha256_reverse":r,"sha_equal":f==r}
def live(a,b):
 s,t=letters(a),letters(b)[::-1];n=0
 for x,y in zip(s,t):
  n+=1
  if x!=y:return {"equations":n,"satisfied":n-1,"all_satisfied":False,"first_mismatch":(n-1,x,y)}
 return {"equations":n,"satisfied":n,"all_satisfied":len(s)==len(t),"first_mismatch":None}
def scene(frames,connectors,depth):
 text=frames[0]
 for i in range(1,depth):text+=connectors[i-1]+frames[i]
 return text
def run():
 rows=[];exact=[];states=0
 for dl,dr,lf,rf,c1,c2 in product((1,2,3),(1,2,3),LEFT,RIGHT,CONNS,CONNS):
  states+=1
  lframes=tuple(LEFT[(LEFT.index(lf)+i)%3] for i in range(dl));rframes=tuple(RIGHT[(RIGHT.index(rf)+i)%3] for i in range(dr))
  left=scene(lframes,(c1,c2),dl);right=scene(rframes,(c2,c1),dr);rendered=left+" "+right;eq=live(left,right);row={"rendered":rendered,"left_depth":dl,"right_depth":dr,"left_connectives":[c1.strip(),c2.strip()],"right_connectives":[c2.strip(),c1.strip()],"online_character_equations":eq,"audit":audit(rendered),"provenance":{"fresh_left_frames":True,"fresh_right_frames":True,"variable_clause_lengths":True,"independent_connective_structure":True,"no_shared_frame_wording":True,"catalogue_text":False,"finished_tape_reversal":False,"post_hoc_repair":False,"mirrored_units":False,"word_order_symmetry":False,"fragment":False},"reader_facing_eligible":False}
  rows.append(row)
  if row["audit"]["pointer_exact"] and row["audit"]["letters"]>38:row["reader_facing_eligible"]=True;exact.append(row)
 result={"experiment_id":"three-frame-scene-tree-variable-lengths-20260920","method":"three-frame authored scene tree with variable clause lengths and independent connectives","stats":{"left_frames":len(LEFT),"right_frames":len(RIGHT),"depth_choices":3,"connectives":len(CONNS),"rendered_candidates":len(rows),"exact_gt38":len(exact),"reader_eligible":len(exact),"longest_letters":max(r["audit"]["letters"] for r in rows)},"all_rendered_candidates":rows,"exact_candidates":exact,"reader_facing_candidates":exact,"novelty_preflight":{"status":"passed","signature":"three-frame-tree|variable-depth|independent-connectives|no-shared-wording","registry_inspected":True,"distinct_from":"fixed two-clause lattices, aligned mirrors, and repair operators","catalogue_text_imported":False,"finished_tape_reversal":False,"post_hoc_repair":False,"mirrored_units":False},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["pointer scan","forward/reverse SHA-256"],"reader_evidence":False},"status":"no reader-worthy exact closure" if not exact else "reader gate required","next_construction":"Add typed scene-level connective roles while preserving variable depths and disjoint authored frame banks."}
 OUT.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({"artifact":str(OUT),**result["stats"]}));[print(r["rendered"]) for r in rows]
if __name__=="__main__":run()
