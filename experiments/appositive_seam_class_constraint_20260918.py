"""Fresh appositive seeds with consonant/vowel seam-class constraint."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='appositive-seam-class-constraint-20260918'
SCENES=[('At noon, the curator, a scholar, studies the quiet gallery while rain falls.'),('At dusk, the gardener, a grower, tends the roses while swallows turn.'),('By dawn, the navigator, a pilot, charts the channel as lanterns fade.')]
def letters(s):return re.sub(r'[^a-z]','',s.lower())
def cls(c):return 'vowel' if c in 'aeiou' else 'consonant'
def audit(s):
 t=letters(s);m=sum(a!=b for a,b in zip(t,t[::-1]));return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[];rejected=0
 for i,text in enumerate(SCENES):
  appositive=text.split(',')[2].strip().split()[1]
  verb=text.split(')')[-1].split()[1] if ')' in text else text.split('studies')[-1] and ('studies' if 'studies' in text else 'tends' if 'tends' in text else 'charts')
  head=letters(appositive)[-1];verb=letters(verb)[0]
  if cls(head)!=cls(verb):rejected+=1;continue
  rows.append({'scene_id':i,'rendered':text,'seam_constraint':{'head_class':cls(head),'verb_boundary_class':cls(verb),'satisfied':True},'audit':audit(text),'provenance':{'fresh_seed':True,'appositive_seam_class':True,'catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'word_order_only_symmetry':False,'novelty_preflight':'new appositive class lane'}})
 best=min(rows,key=lambda r:r['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'considered':len(SCENES),'rendered':len(rows),'rejected_pre_render':rejected,'exact':sum(r['audit']['two_pointer_exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'best_mismatches':best['audit']['mismatches']},'next_repair':{'operator':'appositive seam class with length-balanced head/verb alternatives','reason':'class filtering preserves intact scenes but does not close the tape; next balance head and verb lengths inside the admitted class','route_exhausted':False},'provenance':{'bounded_scenes':len(SCENES),'catalogue_used':False}}
if __name__=='__main__':
 r=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r['stats'],sort_keys=True))
