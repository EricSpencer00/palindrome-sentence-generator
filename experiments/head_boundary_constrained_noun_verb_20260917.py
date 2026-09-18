"""Coupled noun/verb route with relative-head and boundary-length constraints."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='head-boundary-constrained-noun-verb-20260917'
ROWS=[
 ('the chart','guides the crew','the ledger','remembers the route','that','beside the inlet','singular'),
 ('the map','marks the shore','the journal','records the way','which','near the inlet','singular'),
 ('the charts','guide the crews','the ledgers','remember the routes','that','beside the inlets','plural'),
 ('maps','mark shores','journals','record ways','which','near inlets','plural'),
]
def letters(s):return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s);m=sum(a!=b for a,b in zip(t,t[::-1]));return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[]
 for i,(lo,lv,ro,rv,head,boundary,num) in enumerate(ROWS):
  text=f'At dawn, the keeper marks {lo} {head} {lv} {boundary}; the sailor reads {ro} {head} {rv} {boundary}.'
  rows.append({'pair_id':i,'rendered':text,'constraints':{'relative_head':head,'shared_boundary':boundary,'boundary_letters':len(letters(boundary)),'noun_verb_number':num,'all_satisfied':True},'audit':audit(text),'provenance':{'head_and_boundary_constrained':True,'noun_and_relative_verb_coupled':True,'source_experiment':'paired-noun-relative-verb-seam-20260917','catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'word_order_only_symmetry':False}})
 best=min(rows,key=lambda r:r['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(r['audit']['two_pointer_exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'best_mismatches':best['audit']['mismatches']},'next_repair':{'operator':'boundary-specific paired clause growth with live seam audit','reason':'head and boundary constraints preserve paired readability but do not close the global tape; next grow only the boundary clause selected by the live seam','route_exhausted':False},'provenance':{'bounded_rows':len(rows),'catalogue_used':False,'duplicate_sweep':False}}
if __name__=='__main__':
 r=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r['stats'],sort_keys=True))
