"""Role-compatible verb substitutions within determiner/inflection class."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='role-compatible-verb-class-20260918'
ROWS=[('the','guides the crew','marks the route','beside the inlet'),('a','charts the shore','records the way','near the harbor'),('the','guards the pier','watches the quay','along the coast'),('','names the sky','notes the trails','by the wall')]
def letters(s):return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s);m=sum(a!=b for a,b in zip(t,t[::-1]));return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[]
 for i,(det,left,right,boundary) in enumerate(ROWS):
  text=f'At dawn, {det+" " if det else ""}keeper marks the chart that {left} {boundary}; {det+" " if det else ""}sailor reads the ledger that {right} {boundary}.'
  rows.append({'pair_id':i,'rendered':text,'verb_class':{'left':left,'right':right,'role':'transitive semantic relation','inflection_preserved':True},'audit':audit(text),'provenance':{'role_compatible_verb_substitution':True,'source_experiment':'seam-determiner-verb-inflection-20260918','catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'word_order_only_symmetry':False}})
 best=min(rows,key=lambda r:r['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(r['audit']['two_pointer_exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'best_mismatches':best['audit']['mismatches']},'next_repair':{'operator':'paired role-compatible verb substitution with length balance','reason':'verb-class substitutions preserve agreement and role but do not close the tape; next match paired verb lengths at the seam','route_exhausted':False},'provenance':{'bounded_rows':len(rows),'catalogue_used':False}}
if __name__=='__main__':
 r=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r['stats'],sort_keys=True))
