"""Seam-selected inflectional endings with coordinated determiners."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='seam-inflection-determiner-pair-20260918'
ROWS=[('the','the','guides the crew','marks the route','beside the inlet'),('a','a','charts the shore','records the way','near the harbor'),('the','the','guide the crews','remember the routes','beside the inlets'),('','', 'chart shores','record ways','by the wall')]
def letters(s):return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s);m=sum(a!=b for a,b in zip(t,t[::-1]));return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[]
 for i,(dl,dr,left,right,boundary) in enumerate(ROWS):
  text=f'At dawn, {dl+" " if dl else ""}keeper marks {dl+" " if dl else ""}chart that {left} {boundary}; {dr+" " if dr else ""}sailor reads {dr+" " if dr else ""}ledger that {right} {boundary}.'
  rows.append({'pair_id':i,'rendered':text,'coordination':{'left_determiner':dl or 'bare','right_determiner':dr or 'bare','inflectional_endings_coupled':True,'boundary_role':'location'},'audit':audit(text),'provenance':{'seam_selected':True,'determiner_inflection_pair':True,'catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'word_order_only_symmetry':False}})
 best=min(rows,key=lambda r:r['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(r['audit']['two_pointer_exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'best_mismatches':best['audit']['mismatches']},'next_repair':{'operator':'determiner-inflection pair with agreement-aware relative verb','reason':'coordinated determiners and endings preserve role attachment but do not close the tape; next couple the relative verb inflection to the same edit','route_exhausted':False},'provenance':{'bounded_rows':len(rows),'catalogue_used':False}}
if __name__=='__main__':
 r=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r['stats'],sort_keys=True))
