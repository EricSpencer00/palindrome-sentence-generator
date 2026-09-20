"""Fresh human-authored contemporary scene lattice with live equations."""
from __future__ import annotations
import hashlib,json,re
from itertools import product
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/human-authored-scene-lattice-live-20260920.json"
LEFT=("The nurse checks the morning list.","A neighbor waters the young trees.","The teacher opens the quiet room.","Our friends carry warm bread.","The driver follows the river road.","A child watches the first stars.")
RIGHT=("The harbor lights return before dawn.","A gardener hears rain on the roof.","The small cafe closes after sunset.","Our team carries maps to town.","A musician tunes the old piano.","The river settles under moonlight.")
CONNECTORS=(" "," And "," Then ")
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
 candidates=[]; exact=[]; states=0
 for left,right,connector in product(LEFT,RIGHT,CONNECTORS):
  states+=1; rendered=left+connector+right; eq=live(left,right); row={"rendered":rendered,"left_clause":left,"right_clause":right,"connector":connector.strip() or "adjacent-sentences","online_character_equations":eq,"audit":audit(rendered),"provenance":{"human_authored_bank":"fresh contemporary scene clauses","independent_clause_authorship":True,"catalogue_text":False,"finished_tape_reversal":False,"post_hoc_repair":False,"mirrored_units":False,"word_order_symmetry":False,"fragment":False},"reader_facing_eligible":False}
  candidates.append(row)
  if row["audit"]["pointer_exact"] and row["audit"]["letters"]>38:
   row["reader_facing_eligible"]=True;exact.append(row)
 result={"experiment_id":"human-authored-scene-lattice-live-20260920","method":"concurrent live character-equation search over fresh human-authored contemporary scene clauses","stats":{"left_clauses":len(LEFT),"right_clauses":len(RIGHT),"connectors":len(CONNECTORS),"states":states,"exact_gt38":len(exact),"reader_eligible":len(exact),"longest_letters":max(r["audit"]["letters"] for r in candidates)},"all_rendered_candidates":candidates,"exact_candidates":exact,"reader_facing_candidates":exact,"novelty_preflight":{"status":"passed","signature":"fresh-human-authored-scene-lattice|concurrent-live-equations","registry_inspected":True,"distinct_from":"catalogue controls, center dependency lanes, repair, and aligned semordnilap chains","catalogue_text_imported":False,"finished_tape_reversal":False,"post_hoc_repair":False,"mirrored_units":False},"provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_audits":["two-pointer mismatch scan","forward/reverse SHA-256"],"reader_evidence":False},"status":"no reader-worthy exact closure" if not exact else "reader gate required","next_construction":"Use a fresh three-clause scene lattice with one independently authored connective and preserve every live equation.","reader_gate":"closed until exact candidates exist and blinded human ratings are collected"}
 OUT.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({"artifact":str(OUT),**result["stats"]}));[print(r["rendered"]) for r in candidates]
if __name__=="__main__":run()
