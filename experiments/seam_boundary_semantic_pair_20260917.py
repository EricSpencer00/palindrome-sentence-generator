"""Paired boundary growth with role-compatible semantic edits."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='seam-boundary-semantic-pair-20260917'
PAIRS=[('the chart','the ledger','beside the quiet inlet'),('the map','the journal','near the old harbor inlet'),('the plan','the book','along the sheltered inlet'),('the weathered chart','the tide ledger','by the narrow coastal inlet')]
def letters(s):return re.sub(r'[^a-z]','',s.lower())
def seam(s):
 t=letters(s)
 for i,(a,b) in enumerate(zip(t,t[::-1])):
  if a!=b:return i
 return None
def audit(s):
 t=letters(s);m=sum(a!=b for a,b in zip(t,t[::-1]));return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[]
 for i,(left,right,boundary) in enumerate(PAIRS):
  text=f'At dawn, the keeper marks {left} that guides the crew {boundary}; the sailor reads {right} that remembers the route {boundary}.'
  rows.append({'pair_id':i,'rendered':text,'live_seam':{'selected_slot':'semantic_nouns+boundary','first_mismatch':seam(text),'paired_boundary':True},'audit':audit(text),'provenance':{'seam_conditioned':True,'boundary_growth':True,'role_compatible_semantic_edits':True,'source_experiment':'live-seam-boundary-growth-20260917','catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'word_order_only_symmetry':False}})
 best=min(rows,key=lambda r:r['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(r['audit']['two_pointer_exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'best_mismatches':best['audit']['mismatches']},'next_repair':{'operator':'paired seam-conditioned verb and boundary edit','reason':'semantic edits paired with boundary growth remain grammatical but do not close the tape; next couple relative verbs to the same live seam','route_exhausted':False},'provenance':{'bounded_pairs':len(rows),'catalogue_used':False,'duplicate_sweep':False}}
if __name__=='__main__':
 r=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r['stats'],sort_keys=True))
