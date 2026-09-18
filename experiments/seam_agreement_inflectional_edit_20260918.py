"""Seam-selected agreement-aware inflectional edits across clause/boundary."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='seam-agreement-inflectional-edit-20260918'
ROWS=[('guides the crew','marks the route','beside the inlet','singular'),('guide the crews','mark the routes','beside the inlets','plural'),('charts the shore','records the way','near the harbor','singular'),('chart the shores','record the ways','near the harbors','plural')]
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
 for i,(lv,rv,boundary,num) in enumerate(ROWS):
  text=f'At dawn, the keeper marks the chart that {lv} {boundary}; the sailor reads the ledger that {rv} {boundary}.'
  rows.append({'pair_id':i,'rendered':text,'inflection':{'number':num,'relative_and_boundary_coupled':True,'first_mismatch':seam(text)},'audit':audit(text),'provenance':{'seam_selected':True,'agreement_aware_inflection':True,'source_experiment':'seam-paired-relative-boundary-edit-20260918','catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'word_order_only_symmetry':False}})
 best=min(rows,key=lambda r:r['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(r['audit']['two_pointer_exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'best_mismatches':best['audit']['mismatches']},'next_repair':{'operator':'inflectional seam edit with paired determiner change','reason':'agreement-aware inflections shift the seam but do not close the tape; next couple the endings with paired determiners','route_exhausted':False},'provenance':{'bounded_rows':len(rows),'catalogue_used':False}}
if __name__=='__main__':
 r=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r['stats'],sort_keys=True))
