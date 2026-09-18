"""Agreement-aware relative lexical edits with paired seam-length balancing."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='agreement-relative-seam-length-20260918'
ROWS=[('the keeper','marks','chart','guides the crew','the sailor','reads','ledger','marks the route','beside the inlet'),('a keeper','marks','map','charts the shore','a sailor','reads','journal','charts the shore','near the harbor'),('the keepers','mark','charts','guide the crews','the sailors','read','ledgers','guide the crews','along the coast'),('keepers','mark','stars','name the sky','sailors','read','paths','name the sky','by the wall')]
def letters(s):return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s);m=sum(a!=b for a,b in zip(t,t[::-1]));return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[];rej=0
 for i,(ls,lv,lo,lrv,rs,rv,ro,rrv,boundary) in enumerate(ROWS):
  dl=len(letters(lrv));dr=len(letters(rrv))
  if dl!=dr:rej+=1;continue
  text=f'At dawn, {ls} {lv} the {lo} that {lrv} {boundary}; {rs} {rv} the {ro} that {rrv} {boundary}.'
  rows.append({'pair_id':i,'rendered':text,'balance':{'left_relative_letters':dl,'right_relative_letters':dr,'equal':True,'boundary_role':'location'},'audit':audit(text),'provenance':{'agreement_aware':True,'relative_lexical_edit':True,'seam_length_balanced':True,'catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'word_order_only_symmetry':False}})
 best=min(rows,key=lambda r:r['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'considered':len(ROWS),'rendered':len(rows),'rejected_pre_render':rej,'exact':sum(r['audit']['two_pointer_exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'best_mismatches':best['audit']['mismatches']},'next_repair':{'operator':'balanced seam relative edit with paired boundary length','reason':'equal relative lexical lengths preserve agreement but do not close the tape; next balance the paired boundaries at the same seam','route_exhausted':False},'provenance':{'bounded_rows':len(ROWS),'catalogue_used':False}}
if __name__=='__main__':
 r=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r['stats'],sort_keys=True))
