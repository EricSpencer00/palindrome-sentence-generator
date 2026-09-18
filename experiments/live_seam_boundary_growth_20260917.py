"""Paired boundary-clause growth selected by a live first-mismatch seam."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='live-seam-boundary-growth-20260917'
BASE='At dawn, the keeper marks the chart that guides the crew beside the inlet; the sailor reads the ledger that remembers the route beside the inlet.'
GROWTH=['beside the quiet inlet','near the old harbor inlet','along the sheltered inlet','by the narrow coastal inlet']
def letters(s):return re.sub(r'[^a-z]','',s.lower())
def seam(s):
 t=letters(s)
 for i,(a,b) in enumerate(zip(t,t[::-1])):
  if a!=b:return i
 return None
def audit(s):
 t=letters(s);m=sum(a!=b for a,b in zip(t,t[::-1]));return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[];base_seam=seam(BASE)
 for i,boundary in enumerate(GROWTH):
  text=BASE.replace('beside the inlet',boundary)
  rows.append({'growth_id':i,'rendered':text,'live_seam':{'base_first_mismatch':base_seam,'growth_first_mismatch':seam(text),'paired_boundary':True},'audit':audit(text),'provenance':{'boundary_specific_growth':True,'live_seam_audit':True,'source_experiment':'head-boundary-constrained-noun-verb-20260917','catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'word_order_only_symmetry':False}})
 best=min(rows,key=lambda r:r['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(r['audit']['two_pointer_exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'best_mismatches':best['audit']['mismatches']},'next_repair':{'operator':'seam-conditioned boundary growth with paired semantic role edits','reason':'boundary growth shifts the live seam but does not close the global tape; next grow the boundary only when paired with role-compatible seam edits','route_exhausted':False},'provenance':{'bounded_growths':len(rows),'catalogue_used':False,'duplicate_sweep':False}}
if __name__=='__main__':
 r=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r['stats'],sort_keys=True))
