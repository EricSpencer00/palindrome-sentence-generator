"""Attachment-aware paired relative-clause length balancing."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='attachment-clause-length-balance-20260918'
ROWS=[('that guides the crew','that marks the route','beside the inlet'),('that charts the shore','that records the way','near the harbor'),('that guards the pier','that watches the quay','along the coast'),('that names the stars','that notes the paths','by the garden wall')]
def letters(s):return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s);m=sum(a!=b for a,b in zip(t,t[::-1]));return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[];rej=0
 for i,(left,right,boundary) in enumerate(ROWS):
  dl=len(letters(left));dr=len(letters(right))
  if dl!=dr:rej+=1;continue
  text=f'At dawn, the keeper marks the chart {left} {boundary}; the sailor reads the ledger {right} {boundary}.'
  rows.append({'pair_id':i,'rendered':text,'balance':{'left_letters':dl,'right_letters':dr,'equal':True,'attachment_role':'location'},'audit':audit(text),'provenance':{'attachment_aware_length_balance':True,'source_experiment':'attachment-head-verb-pair-20260918','catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'word_order_only_symmetry':False}})
 best=min(rows,key=lambda r:r['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'considered':len(ROWS),'rendered':len(rows),'rejected_pre_render':rej,'exact':sum(r['audit']['two_pointer_exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'best_mismatches':best['audit']['mismatches']},'next_repair':{'operator':'attachment-aware balanced clause with paired determiner edit','reason':'equal relative-clause lengths preserve attachment but do not close the tape; next pair the balanced clauses with determiner changes','route_exhausted':False},'provenance':{'bounded_rows':len(ROWS),'catalogue_used':False}}
if __name__=='__main__':
 r=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r['stats'],sort_keys=True))
