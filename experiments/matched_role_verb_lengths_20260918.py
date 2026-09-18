"""Role-compatible paired verbs with matched letter lengths."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='matched-role-verb-lengths-20260918'
ROWS=[('guides','marks','crew','route','beside the inlet'),('charts','marks','shore','way','near the harbor'),('guards','watches','pier','quay','along the coast'),('names','notes','sky','trails','by the wall')]
def letters(s):return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s);m=sum(a!=b for a,b in zip(t,t[::-1]));return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[];rej=0
 for i,(lv,rv,lo,ro,boundary) in enumerate(ROWS):
  if len(letters(lv))!=len(letters(rv)):rej+=1;continue
  text=f'At dawn, the keeper marks the chart that {lv} the {lo} {boundary}; the sailor reads the ledger that {rv} the {ro} {boundary}.'
  rows.append({'pair_id':i,'rendered':text,'verb_balance':{'left_letters':len(letters(lv)),'right_letters':len(letters(rv)),'equal':True,'inflection_class':'present'},'audit':audit(text),'provenance':{'role_compatible':True,'matched_verb_lengths':True,'source_experiment':'role-compatible-verb-class-20260918','catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'word_order_only_symmetry':False}})
 best=min(rows,key=lambda r:r['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'considered':len(ROWS),'rendered':len(rows),'rejected_pre_render':rej,'exact':sum(r['audit']['two_pointer_exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'best_mismatches':best['audit']['mismatches']},'next_repair':{'operator':'matched verb lengths with paired relative-object length','reason':'verb-length matching preserves local seam geometry but does not close the tape; next match the paired semantic-object lengths too','route_exhausted':False},'provenance':{'bounded_rows':len(ROWS),'catalogue_used':False}}
if __name__=='__main__':
 r=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r['stats'],sort_keys=True))
