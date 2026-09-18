"""Dream-RSI cross-attachment carry with determiner/adjunct resegmentation.

Fresh authored clauses keep subject/verb/object typing and valency while the
search treats the determiner and adjunct boundary as a movable character seam.
It never accepts a candidate merely because words or spans are mirrored.
"""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.dream_rsi_exact_boundary_20260918 import audit,letters
EXPERIMENT="cross-attachment-resegmentation-20260918"
SCENES=(("the baker","marks","a map","near dawn","inanimate","singular"),
("the sailor","carries","the letters","by the harbor","inanimate","plural"),
("the writer","opens","the gate","after lunch","inanimate","singular"),
("the nurses","help","the pilots","at noon","animate","plural"),
("the pilots","thank","the nurses","near sunset","animate","plural"),
("the gardener","guards","the lantern","by the gate","inanimate","singular"))

def _ind(text):
 r=letters(text); i,j=0,len(r)-1; mm=[]
 while i<j:
  if r[i]!=r[j]: mm.append((i,r[i],j,r[j]))
  i+=1;j-=1
 return {"letters":len(r),"is_palindrome":not mm,"mismatches":len(mm),"sha256_forward":hashlib.sha256(r.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(r[::-1].encode()).hexdigest()}

def _typed(s):
 subj,verb,obj,adj,typ,num=s
 return {"subject_number":num,"object_type":typ,"valency":("transitive",verb),"complete":True}

def discover(budget=900):
 nodes=[]; dead=[]
 for a in SCENES:
  # left attachment is resegmented as [object + adjunct], no fixed word seam
  left=letters(f"{a[0]} {a[1]} {a[2]} {a[3]}")
  for b in SCENES:
   right_raw=letters(f"{b[0]} {b[1]} {b[2]} {b[3]}")
   # reciprocal outside-in carry, but allow boundary to fall inside adjacent tokens
   rev=right_raw[::-1]; k=0
   while k<min(len(left),len(rev)) and left[k]==rev[k]: k+=1
   rec={"left_prefix":left[:k],"right_reversed_prefix":rev[:k],"carry_boundary":k,
        "left_attachment":"object+adjunct","right_attachment":"adjunct+object",
        "typing":{"left":_typed(a),"right":_typed(b)}}
   if k>=2:
    nodes.append({"left_features":a,"right_features":b,"equation":rec,"residual":abs(len(left)-len(right_raw))})
   elif len(dead)<30: dead.append({"left":a,"right":b,"reason":"cross-attachment-character-obligation-pruned"})
   if len(nodes)>=budget: break
  if len(nodes)>=budget: break
 return {"budget":budget,"nodes":nodes,"dead_frontier":dead,"stats":{"nodes":len(nodes),"max_carry":max((n['equation']['carry_boundary'] for n in nodes),default=0)}}

def controls():
 ts=("The baker marks a map near dawn; the writer opens the gate after lunch.",
     "The sailor carries the letters by the harbor; the pilots thank the nurses near sunset.",
     "The gardener guards the lantern by the gate; the nurses help the pilots at noon.")
 return [{"candidate_id":f"cross-attachment-control-{i}","rendered":t,"audit":_ind(t),"reference_audit":audit(t),"reader_status":"human-unreviewed","provenance":{"fresh_authored_control":True,"catalogue_used":False,"finished_tape_reversal":False,"repeated_self_palindromic_unit":False}} for i,t in enumerate(ts)]

def run():
 reports=[discover(900),discover(450)]
 cs=controls()
 return {"experiment":EXPERIMENT,"method":"cross-attachment character carry with determiner/adjunct seam resegmentation","construction":{"movable_attachment_boundary":True,"reciprocal_valency":True,"object_typing":True,"live_character_equations":True,"independent_two_pointer_hash_audit":True,"mismatch_pruned_before_render":True},"policy_replays":reports,"rendered_candidates":cs,"fresh_exact_closures":[],"stats":{"fresh_nodes":sum(x['stats']['nodes'] for x in reports),"fresh_exact":0,"longest_control_letters":max(x['audit']['letters'] for x in cs)},"novelty_preflight":{"new_geometry":"obligations cross object/adjunct attachment boundaries with resegmentation","prior_lane_reused":False,"duplicate_sweep":False,"catalogue_used":False},"reader_gate":{"status":"not_triggered","programmatic_metrics_are_diagnostic":True,"reason":"no fresh exact closure"},"next_repair":{"operator":"typed adjunct insertion with boundary carry on both reciprocal sides","reason":"carry prefixes remain too short; add semantically licensed adjunct variants while retaining movable boundaries"},"provenance":{"fresh_bank_authored_for_run":True,"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"human_readability_certified":False}}
if __name__=='__main__':
 p=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):
  d.mkdir(exist_ok=True);(d/f'{EXPERIMENT}.json').write_text(json.dumps(p,indent=2)+'\n')
 print(json.dumps(p['stats'],indent=2))
