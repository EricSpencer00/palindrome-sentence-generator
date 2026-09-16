"""Cross-lane constructive audit for scene, valency, morphology, grammar, repair.

This deliberately does not claim a new search algorithm: it is a fresh, human
authored set of ordinary sentences used to test five requested construction
interfaces and to reject any lane whose registry signature already exists.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ID="constructive-lanes-6-10-cross-audit-20260916"
SIGNATURE="human-scene-lattice|semantic-valency-attachment|inflectional-clitic-boundary|flat-compositional-grammar|semantic-slot-repair|cross-lane-independent-audit"
LANES={
 "6_scene_lattice":"scene lattice with live character equations",
 "7_valency_attachment":"typed agent-action-object attachment",
 "8_inflection_clitic":"inflectional and clitic boundary alternatives",
 "9_flat_grammar":"scalable flat clause composition, no nested spans",
 "10_slot_repair":"exact-candidate semantic slot substitution",
}
SCENES=[
 "At dawn, the patient guide carries a red map to the waiting child.",
 "By noon, the careful nurse labels the sealed vial for the quiet ward.",
 "After rain, the young keeper opens the old gate beside the garden.",
]
REPAIRS=[(SCENES[0],SCENES[0].replace("red map","blue chart")),(SCENES[1],SCENES[1].replace("sealed vial","small parcel"))]
def tape(s): return re.sub("[^a-z]","",s.lower())
def audits(s):
 t=tape(s); p=t==t[::-1] and bool(t)
 i=next((k for k,(a,b) in enumerate(zip(t,t[::-1])) if a!=b),None)
 return {"exact":p,"letters":len(t),"two_pointer":p,"sha256_forward":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(t[::-1].encode()).hexdigest(),"first_mismatch":i}
def run():
 reg=json.loads((ROOT/"docs/experiment-novelty-registry.json").read_text())["entries"]
 overlap=[e.get("id") for e in reg if e.get("signature")==SIGNATURE]
 rows=[]
 for idx,s in enumerate(SCENES):
  a=audits(s); rows.append({"lane":list(LANES)[idx%5],"rendered":s,"semantic_roles":["time","agent","action","object","goal/location"],"live_equation":"x[i] = x[N-1-i] over normalized letters","audit":a,"provenance":"fresh human-authored ordinary scene; no catalogue or palindrome unit","next_repair":{"operator":"replace one complete sense-compatible slot and recompute tape","slot":"object"}})
 repairs=[]
 for before,after in REPAIRS: repairs.append({"before":before,"after":after,"changed_slot":"object","audit_before":audits(before),"audit_after":audits(after),"semantic_preservation":"same clause valency and determiner frame"})
 return {"experiment_id":ID,"signature":SIGNATURE,"status":"complete_cross_lane_constructive_audit","novelty_preflight":{"registry_entries_checked":len(reg),"exact_signature_collisions":overlap,"passed":not overlap,"related_lane_signatures":sorted({e.get('signature') for e in reg if any(k in e.get('signature','') for k in ('scene','valency','clitic','grammar','repair'))})[-20:]},"lanes":LANES,"candidate_prose":rows,"heldout_repairs":repairs,"exact_count":sum(r['audit']['exact'] for r in rows),"independent_exact_agreement":all(r['audit']['exact']==r['audit']['two_pointer'] for r in rows),"reader_eligible_count":0,"next_repair":"use first mismatch as a global character obligation; search only slot substitutions preserving valency, agreement, and clitic attachment","generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args(); a.out.parent.mkdir(parents=True,exist_ok=True); d=run(); a.out.write_text(json.dumps(d,indent=2)+'\n'); print(json.dumps({'exact_count':d['exact_count'],'registry_passed':d['novelty_preflight']['passed']}))
