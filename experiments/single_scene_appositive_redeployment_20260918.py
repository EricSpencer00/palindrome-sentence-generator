"""Orthogonal Dream-RSI redeployment: single-scene appositive prose."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='single-scene-appositive-redeployment-20260918'
SCENES=[('At dawn, the keeper, a patient cartographer, maps the inlet while the sailor checks the tide.'),('After rain, a quiet teacher, an attentive guide, records the garden while children watch.'),('Before noon, the harbor pilot, an experienced navigator, charts the channel as gulls circle.')]
def letters(s):return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s);m=sum(a!=b for a,b in zip(t,t[::-1]));return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[]
 for i,text in enumerate(SCENES):
  rows.append({'scene_id':i,'rendered':text,'policy':{'replay_choice':'single-scene-appositive','seam_conditioned':True,'two_region_authoring':False},'audit':audit(text),'provenance':{'authored_scene':True,'catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'word_order_only_symmetry':False,'novelty_preflight':'orthogonal to paired-clause lanes'}})
 best=min(rows,key=lambda r:r['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(r['audit']['two_pointer_exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'best_mismatches':best['audit']['mismatches']},'next_repair':{'operator':'single-scene appositive seam repair','reason':'the orthogonal appositive construction yields intact readable scenes but no exact closure; next edit the appositive head and adjacent verb as a single seam unit','route_exhausted':False},'provenance':{'bounded_scenes':len(rows),'catalogue_used':False,'duplicate_sweep':False}}
if __name__=='__main__':
 r=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r['stats'],sort_keys=True))
