"""Head/verb class-length states with explicit paired boundary attachment."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='head-verb-boundary-attachment-20260918'
ROWS=[('chart','maps','crew','ledger','notes','route','beside the inlet','location'),('map','marks','shore','journal','records','way','near the harbor','location'),('plan','guards','pier','book','watches','quay','along the coast','location'),('stars','names','sky','paths','notes','trails','by the garden wall','location')]
def letters(s):return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s);m=sum(a!=b for a,b in zip(t,t[::-1]));return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[];rej=0
 for i,(ln,lv,lo,rn,rv,ro,boundary,role) in enumerate(ROWS):
  dl=len(letters(ln))+len(letters(lv));dr=len(letters(rn))+len(letters(rv))
  if dl!=dr or role!='location':rej+=1;continue
  text=f'At dawn, the keeper {lv} the {ln} that guides the {lo} {boundary}; the sailor {rv} the {rn} that remembers the {ro} {boundary}.'
  rows.append({'pair_id':i,'rendered':text,'constraints':{'head_verb_delta_left':dl,'head_verb_delta_right':dr,'attachment_role_left':role,'attachment_role_right':role,'all_satisfied':True},'audit':audit(text),'provenance':{'boundary_attachment_checked':True,'head_verb_class_gate':True,'catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'word_order_only_symmetry':False}})
 best=min(rows,key=lambda r:r['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'considered':len(ROWS),'rendered':len(rows),'rejected_pre_render':rej,'exact':sum(r['audit']['two_pointer_exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'best_mismatches':best['audit']['mismatches']},'next_repair':{'operator':'attachment-aware paired relative-clause lexical repair','reason':'same-role boundaries preserve attachment but do not close the tape; next alter only relative-clause lexical heads under the same gate','route_exhausted':False},'provenance':{'bounded_rows':len(ROWS),'catalogue_used':False}}
if __name__=='__main__':
 r=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r['stats'],sort_keys=True))
