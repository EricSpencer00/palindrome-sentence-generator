"""Agreement-aware determiner/verb structure with paired boundary inflection."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='agreement-boundary-inflection-pair-20260918'
ROWS=[('the keeper','marks','chart','guides the crew','the sailor','reads','ledger','marks the route','beside the inlet'),('a keeper','marks','map','charts the shore','a sailor','reads','journal','charts the shore','near the harbor'),('the keepers','mark','charts','guide the crews','the sailors','read','ledgers','guide the crews','along the coast'),('keepers','mark','stars','name the sky','sailors','read','paths','name the sky','by the garden wall')]
def letters(s):return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s);m=sum(a!=b for a,b in zip(t,t[::-1]));return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[]
 for i,(ls,lv,lo,lrv,rs,rv,ro,rrv,boundary) in enumerate(ROWS):
  text=f'At dawn, {ls} {lv} the {lo} that {lrv} {boundary}; {rs} {rv} the {ro} that {rrv} {boundary}.'
  rows.append({'pair_id':i,'rendered':text,'boundary':{'left':boundary,'right':boundary,'paired':True,'role':'location'},'audit':audit(text),'provenance':{'agreement_aware':True,'boundary_inflection':True,'catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'word_order_only_symmetry':False}})
 best=min(rows,key=lambda r:r['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(r['audit']['two_pointer_exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'best_mismatches':best['audit']['mismatches']},'next_repair':{'operator':'agreement-aware boundary plus relative-clause lexical substitution','reason':'boundary inflection preserves agreement and attachment but does not close the tape; next pair it with a role-compatible relative-clause lexical edit','route_exhausted':False},'provenance':{'bounded_rows':len(rows),'catalogue_used':False}}
if __name__=='__main__':
 r=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r['stats'],sort_keys=True))
