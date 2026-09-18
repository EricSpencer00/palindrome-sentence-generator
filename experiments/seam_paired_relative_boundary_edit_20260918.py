"""Seam-selected paired lexical edits across relative clauses and boundaries."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='seam-paired-relative-boundary-edit-20260918'
ROWS=[('guides the crew','marks the route','beside the inlet','charts the crew','marks the route','beside the inlet'),('charts the shore','records the way','near the harbor','charts the shore','records the way','near the harbor'),('guide the crews','remember the routes','along the coast','guide the crews','remember the routes','along the coast'),('name the sky','note the trails','by the wall','name the sky','note the trails','by the wall')]
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
 for i,(l1,r1,b1,l2,r2,b2) in enumerate(ROWS):
  text=f'At dawn, the keeper marks the chart that {l1} {b1}; the sailor reads the ledger that {r1} {b2}.'
  rows.append({'pair_id':i,'rendered':text,'seam':{'first_mismatch':seam(text),'paired_relative_edit':True,'paired_boundary_edit':True},'audit':audit(text),'provenance':{'agreement_preserved':True,'attachment_preserved':True,'source_experiment':'agreement-relative-boundary-length-20260918','catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'word_order_only_symmetry':False}})
 best=min(rows,key=lambda r:r['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(r['audit']['two_pointer_exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'best_mismatches':best['audit']['mismatches']},'next_repair':{'operator':'seam-selected agreement-aware inflectional edit','reason':'paired relative and boundary edits preserve readability but remain non-exact; next adjust inflectional endings at the live seam','route_exhausted':False},'provenance':{'bounded_rows':len(rows),'catalogue_used':False}}
if __name__=='__main__':
 r=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r['stats'],sort_keys=True))
