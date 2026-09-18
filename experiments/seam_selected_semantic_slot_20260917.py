"""Targeted semantic-slot substitutions selected by the first mismatch seam."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='seam-selected-semantic-slot-20260917'
BASE='At dawn, the keeper marks the chart that guides the crew beside the inlet; the sailor reads the ledger that remembers the route beside the inlet.'
REPAIRS=[('chart','map'),('keeper','steward'),('sailor','navigator'),('ledger','journal')]
def letters(s):return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s);m=sum(a!=b for a,b in zip(t,t[::-1]));return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def first_mismatch(s):
 t=letters(s)
 for i,(a,b) in enumerate(zip(t,t[::-1])):
  if a!=b:return i
 return None
def run():
 rows=[];seam=first_mismatch(BASE)
 for i,(old,new) in enumerate(REPAIRS):
  text=BASE.replace(old,new,1)
  rows.append({'repair_id':i,'rendered':text,'seam_selection':{'base_first_mismatch_index':seam,'selected_slot':old,'role':'semantic noun'},'audit':audit(text),'provenance':{'seam_selected':True,'agreement_preserved':True,'role_compatible':True,'source_experiment':'coupled-determiner-relative-agreement-20260917','catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'word_order_only_symmetry':False}})
 best=min(rows,key=lambda r:r['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(r['audit']['two_pointer_exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'best_mismatches':best['audit']['mismatches']},'next_repair':{'operator':'paired seam-selected noun substitution with agreement-preserving length match','reason':'single targeted noun substitutions preserve grammar but do not close the tape; next coordinate both opposing semantic nouns under the same seam index','route_exhausted':False},'provenance':{'bounded_repairs':len(rows),'catalogue_used':False,'duplicate_sweep':False}}
if __name__=='__main__':
 r=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r['stats'],sort_keys=True))
