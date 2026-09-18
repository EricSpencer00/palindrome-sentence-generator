"""Paired boundary-inflection repair preserving semantic attachment and keys."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='paired-boundary-inflection-20260917'
ROWS=[
 ('beside the inlet','beside the inlet'),
 ('near the inlet','near the inlet'),
 ('along the inlet','along the inlet'),
 ('by the old inlet','by the old inlet'),
]
def letters(s):return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s);m=sum(a!=b for a,b in zip(t,t[::-1]));return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[]
 for i,(left_boundary,right_boundary) in enumerate(ROWS):
  text=f'At dawn, the keeper marks the chart that guides the crew {left_boundary}; the sailor reads the ledger that remembers the route {right_boundary}.'
  rows.append({'pair_id':i,'rendered':text,'boundary_inflection':{'left':left_boundary,'right':right_boundary,'paired':left_boundary==right_boundary,'attachment':'location'},'audit':audit(text),'provenance':{'semantic_role_keys_preserved':True,'boundary_inflection_only':True,'source_experiment':'key-preserving-slot-length-20260917','catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'word_order_only_symmetry':False}})
 best=min(rows,key=lambda r:r['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(r['audit']['two_pointer_exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'best_mismatches':best['audit']['mismatches']},'next_repair':{'operator':'paired attachment-preserving determiner/inflection alternatives','reason':'paired boundary inflections preserve role and attachment but do not close the tape; next change determiners and inflections jointly while retaining the same semantic frame','route_exhausted':False},'provenance':{'bounded_pairs':len(rows),'catalogue_used':False,'duplicate_sweep':False}}
if __name__=='__main__':
 r=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r['stats'],sort_keys=True))
