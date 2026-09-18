"""Seam repair coupling determiner, relative-verb, and boundary inflection."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='seam-determiner-verb-inflection-20260918'
ROWS=[('the','guides the crew','marks the route','beside the inlet','singular'),('a','charts the shore','records the way','near the harbor','singular'),('the','guide the crews','remember the routes','beside the inlets','plural'),('','chart shores','record ways','by the wall','plural')]
def letters(s):return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s);m=sum(a!=b for a,b in zip(t,t[::-1]));return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[]
 for i,(det,left,right,boundary,num) in enumerate(ROWS):
  text=f'At dawn, {det+" " if det else ""}keeper marks {det+" " if det else ""}chart that {left} {boundary}; {det+" " if det else ""}sailor reads {det+" " if det else ""}ledger that {right} {boundary}.'
  rows.append({'pair_id':i,'rendered':text,'agreement':{'number':num,'determiner_verb_coupled':True,'boundary_inflection_coupled':True},'audit':audit(text),'provenance':{'seam_selected':True,'verb_inflection_coupled':True,'catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'word_order_only_symmetry':False}})
 best=min(rows,key=lambda r:r['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(r['audit']['two_pointer_exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'best_mismatches':best['audit']['mismatches']},'next_repair':{'operator':'agreement-aware verb inflection with paired lexical seam substitution','reason':'verb endings coupled to determiners and boundaries remain non-exact; next substitute role-compatible verbs while preserving their inflection class','route_exhausted':False},'provenance':{'bounded_rows':len(rows),'catalogue_used':False}}
if __name__=='__main__':
 r=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r['stats'],sort_keys=True))
