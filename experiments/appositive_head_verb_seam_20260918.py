"""New single-scene appositive seeds with head+verb seam edits."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='appositive-head-verb-seam-20260918'
SCENES=[('At sunrise, the archivist, a careful keeper, studies the old map while the bell rings.'),('At evening, a gardener, a patient cultivator, tends the roses as swallows settle.'),('After rain, the pilot, a seasoned navigator, charts the inlet while the lantern glows.')]
def letters(s):return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s);m=sum(a!=b for a,b in zip(t,t[::-1]));return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[]
 for i,text in enumerate(SCENES):
  rows.append({'scene_id':i,'rendered':text,'seam_operator':{'mutable_unit':'appositive_head+adjacent_verb','new_seed':True,'single_scene':True},'audit':audit(text),'provenance':{'catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'word_order_only_symmetry':False,'novelty_preflight':'new appositive seeds distinct from prior redeployment'}})
 best=min(rows,key=lambda r:r['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(r['audit']['two_pointer_exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'best_mismatches':best['audit']['mismatches']},'next_repair':{'operator':'appositive seam consonant-vowel class repair','reason':'joint head+verb edits preserve intact single-scene prose but do not close the tape; next constrain the seam character class','route_exhausted':False},'provenance':{'bounded_scenes':len(rows),'catalogue_used':False}}
if __name__=='__main__':
 r=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r['stats'],sort_keys=True))
