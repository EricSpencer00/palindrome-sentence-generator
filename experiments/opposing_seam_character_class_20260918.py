"""Opposing-seam character-class lexical repair."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='opposing-seam-character-class-20260918'
BASE='At dawn, the keeper marks the chart that guides the crew beside the inlet; the sailor reads the ledger that remembers the route beside the inlet.'
PAIRS=[('crew','team','route','path'),('crew','band','route','road'),('chart','card','ledger','diary'),('keeper','warden','sailor','pilot')]
def letters(s):return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s);m=sum(a!=b for a,b in zip(t,t[::-1]));return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[];rejected=0
 for i,(a,b,c,d) in enumerate(PAIRS):
  # Opposing replacements must share their terminal character class.
  if (letters(b)[-1] in 'aeiou') != (letters(d)[-1] in 'aeiou'):
   rejected+=1;continue
  text=BASE.replace(a,b,1).replace(c,d,1)
  rows.append({'repair_id':i,'rendered':text,'class_constraint':{'left_terminal':letters(b)[-1],'right_terminal':letters(d)[-1],'same_vowel_class':True},'audit':audit(text),'provenance':{'opposing_character_class':True,'pre_render_rejection':True,'source_experiment':'live-seam-paired-lexical-repair-20260918','catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'word_order_only_symmetry':False}})
 best=min(rows,key=lambda r:r['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'considered':len(PAIRS),'rendered':len(rows),'rejected_pre_render':rejected,'exact':sum(r['audit']['two_pointer_exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'best_mismatches':best['audit']['mismatches']},'next_repair':{'operator':'paired opposing seam consonant/vowel class with length matching','reason':'character-class coupling admits role-compatible prose but does not close the tape; next add bounded replacement-length matching within each class','route_exhausted':False},'provenance':{'bounded_pairs':len(PAIRS),'catalogue_used':False,'duplicate_sweep':False}}
if __name__=='__main__':
 r=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r['stats'],sort_keys=True))
