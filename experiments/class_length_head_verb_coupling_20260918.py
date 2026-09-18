"""Semantic-head/verb coupling under paired class-length gate."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='class-length-head-verb-coupling-20260918'
ROWS=[('chart','maps','crew','ledger','notes','route','roads'),('map','marks','shore','journal','records','way','road'),('plan','guards','pier','book','watches','quay','coast'),('stars','names','sky','paths','notes','trails','walls')]
def letters(s):return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s);m=sum(a!=b for a,b in zip(t,t[::-1]));return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[];rej=0
 for i,(ln,lv,lo,rn,rv,ro,boundary) in enumerate(ROWS):
  dl=len(letters(ln))+len(letters(lv));dr=len(letters(rn))+len(letters(rv))
  if dl!=dr:rej+=1;continue
  text=f'At dawn, the keeper {lv} the {ln} that {lv} the {lo} {boundary}; the sailor {rv} the {rn} that {rv} the {ro} {boundary}.'
  rows.append({'pair_id':i,'rendered':text,'constraint':{'left_head_verb_letters':dl,'right_head_verb_letters':dr,'class_length_matched':True},'audit':audit(text),'provenance':{'head_verb_coupled':True,'class_length_gate':True,'catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'word_order_only_symmetry':False}})
 best=min(rows,key=lambda r:r['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'considered':len(ROWS),'rendered':len(rows),'rejected_pre_render':rej,'exact':sum(r['audit']['two_pointer_exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'best_mismatches':best['audit']['mismatches']},'next_repair':{'operator':'class-length head-verb-boundary repair with attachment check','reason':'head/verb length coupling admits readable candidates but does not close the tape; next include boundary attachment under the same gate','route_exhausted':False},'provenance':{'bounded_rows':len(ROWS),'catalogue_used':False}}
if __name__=='__main__':
 r=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r['stats'],sort_keys=True))
