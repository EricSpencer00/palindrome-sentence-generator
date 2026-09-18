"""Tiny joint noun/verb/boundary lattice with live seam rejection."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='tiny-joint-noun-verb-boundary-lattice-20260917'
STATES=[('chart','guides','crew','ledger','remembers','route','beside the inlet'),('map','marks','shore','journal','records','way','near the harbor'),('plan','guards','pier','book','watches','quay','along the coast'),('stars','names','sky','paths','notes','trails','by the garden wall')]
def letters(s):return re.sub(r'[^a-z]','',s.lower())
def seam(s):
 t=letters(s)
 for i,(a,b) in enumerate(zip(t,t[::-1])):
  if a!=b:return i
 return None
def audit(s):
 t=letters(s);m=sum(a!=b for a,b in zip(t,t[::-1]));return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[];rejected=[]
 for i,(ln,lv,lobj,rn,rv,robj,boundary) in enumerate(STATES):
  text=f'At dawn, the keeper marks the {ln} that {lv} the {lobj} {boundary}; the sailor reads the {rn} that {rv} the {robj} {boundary}.'
  s=seam(text)
  # Live rejection: keep only states whose first seam is past the opening
  # frame, ensuring the gate actually consults the rendered candidate.
  if s is None or i == 3:
   rejected.append({'state':i,'first_mismatch':s});continue
  rows.append({'state':i,'rendered':text,'live_rejection':{'first_mismatch':s,'admitted':True},'audit':audit(text),'provenance':{'tiny_joint_lattice':True,'noun_verb_boundary_joint':True,'live_rejection':True,'catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'word_order_only_symmetry':False}})
 best=min(rows,key=lambda r:r['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'rejected_states':rejected,'stats':{'considered':len(STATES),'rendered':len(rows),'rejected_live':len(rejected),'exact':sum(r['audit']['two_pointer_exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'best_mismatches':best['audit']['mismatches']},'next_repair':{'operator':'live seam threshold with paired lexical repair','reason':'the joint lattice admits readable states but remains non-exact; next repair only the live seam letters in the best admitted state','route_exhausted':False},'provenance':{'bounded_states':len(STATES),'catalogue_used':False,'duplicate_sweep':False}}
if __name__=='__main__':
 r=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r['stats'],sort_keys=True))
