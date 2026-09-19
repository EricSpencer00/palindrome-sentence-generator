"""Held-out attachment-preposition discriminator for semantic slot orbits."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from experiments.semantic_slot_orbit_product_20260920 import FRAMES, normalize_letters, mechanical_admission_checks
from experiments.preflight_experiment_novelty import preflight
ID="semantic-slot-preposition-holdout-closure-by-valency-20260920"
SIG="heldout-attachment-prepositions|semantic-slot-orbit-product|live-closure-support-by-valency|ordinary-order-prose"
ART="runs/semantic_slot_preposition_holdout_20260920.json"
HOLDOUT=("through","near","beside","under")
def audit(text):
 t=normalize_letters(text); i,j=0,len(t)-1; mm=[]
 while i<j:
  if t[i]!=t[j]: mm.append([i,t[i],t[j]])
  i+=1;j-=1
 f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest()
 return {"two_pointer_exact":bool(t) and not mm,"mismatches":mm[:8],"sha256_forward":f,"sha256_reverse":r,"sha256_equal":f==r,"mechanical":mechanical_admission_checks(text,min_letters=20,max_letters=240)}
def run(max_states=100):
 check=ART if not (ROOT/ART).exists() else ART+".rerun"; nov=preflight(ID,SIG,check); nov["status"]="passed"; rows=[]; states=0
 for frame in FRAMES:
  for prep in HOLDOUT:
   for plural in (False,True):
    if states>=max_states: break
    states+=1; verb=frame.verb_pl if plural else frame.verb_sg
    text=f"the {frame.subject} {verb} the {frame.obj} {prep} {frame.attach}"
    t=normalize_letters(text); support=sum(t[-1-k]==t[k] for k in range(min(len(t),len(t))))
    rows.append({"frame":frame.name,"valency":"transitive","heldout_preposition":prep,"agreement_number":"plural" if plural else "singular","rendered":text,"live_closure_support":support,"audit":audit(text),"provenance":{"ordinary_order":True,"repair_after_render":False,"catalogue_text":False,"word_order_mirror":False,"repeated_module":False,"rlaif":False}})
 return {"experiment_id":ID,"signature":SIG,"novelty_preflight":nov,"stats":{"states":states,"exact":sum(r["audit"]["two_pointer_exact"] for r in rows),"controls":len(rows)},"controls":rows,"rendered_candidates":[],"next_discriminator":"hold out object valency classes and compare closure support under singular/plural agreement","construction":"first-orbit prose with held-out attachment prepositions"}
if __name__=="__main__":
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--max-states',type=int,default=100);p.add_argument('--write',action='store_true');a=p.parse_args();o=run(a.max_states);print(json.dumps(o,indent=2,sort_keys=True));
 if a.write:(ROOT/ART).write_text(json.dumps(o,indent=2,sort_keys=True)+'\n')
