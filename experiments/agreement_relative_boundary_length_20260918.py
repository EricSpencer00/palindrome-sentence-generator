"""Agreement-aware relative edits with paired boundary-length balancing."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='agreement-relative-boundary-length-20260918'
ROWS=[('guides the crew','marks the route','beside the inlet','beside the inlet'),('charts the shore','charts the shore','near the harbor','near the harbor'),('guide the crews','guide the crews','along the coast','along the coast'),('name the sky','name the sky','by the old wall','by the old wall')]
def letters(s):return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s);m=sum(a!=b for a,b in zip(t,t[::-1]));return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[]
 for i,(left,right,lb,rb) in enumerate(ROWS):
  text=f'At dawn, the keeper marks the chart that {left} {lb}; the sailor reads the ledger that {right} {rb}.'
  rows.append({'pair_id':i,'rendered':text,'balance':{'left_boundary_letters':len(letters(lb)),'right_boundary_letters':len(letters(rb)),'equal':len(letters(lb))==len(letters(rb)),'relative_lengths_equal':len(letters(left))==len(letters(right))},'audit':audit(text),'provenance':{'paired_boundary_length':True,'agreement_aware_relative_edit':True,'catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'word_order_only_symmetry':False}})
 best=min(rows,key=lambda r:r['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(r['audit']['two_pointer_exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'best_mismatches':best['audit']['mismatches']},'next_repair':{'operator':'paired boundary and relative lexical substitution with seam audit','reason':'equal boundary and relative lengths preserve local geometry but do not close the tape; next make a seam-selected paired lexical edit across both components','route_exhausted':False},'provenance':{'bounded_rows':len(rows),'catalogue_used':False}}
if __name__=='__main__':
 r=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r['stats'],sort_keys=True))
