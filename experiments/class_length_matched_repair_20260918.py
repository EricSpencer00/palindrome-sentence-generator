"""Character-class repair with matched paired replacement-length deltas."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='class-length-matched-repair-20260918'
BASE='At dawn, the keeper marks the chart that guides the crew beside the inlet; the sailor reads the ledger that remembers the route beside the inlet.'
PAIRS=[('crew','team','route','roads'),('crew','band','route','road'),('chart','cards','ledger','books'),('keeper','warden','sailor','pilots')]
def letters(s):return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s);m=sum(a!=b for a,b in zip(t,t[::-1]));return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[];rejected=0
 for i,(a,b,c,d) in enumerate(PAIRS):
  dl=len(letters(b))-len(letters(a));dr=len(letters(d))-len(letters(c))
  if dl!=dr:rejected+=1;continue
  text=BASE.replace(a,b,1).replace(c,d,1)
  rows.append({'repair_id':i,'rendered':text,'constraint':{'left_delta':dl,'right_delta':dr,'class_length_matched':True},'audit':audit(text),'provenance':{'character_class_preserved':True,'replacement_length_matched':True,'source_experiment':'opposing-seam-character-class-20260918','catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'word_order_only_symmetry':False}})
 best=min(rows,key=lambda r:r['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'considered':len(PAIRS),'rendered':len(rows),'rejected_pre_render':rejected,'exact':sum(r['audit']['two_pointer_exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'best_mismatches':best['audit']['mismatches']},'next_repair':{'operator':'class-length matched semantic-head and verb repair','reason':'equal paired length deltas preserve local geometry but do not close the tape; next couple semantic heads and verbs under the same class-length gate','route_exhausted':False},'provenance':{'bounded_pairs':len(PAIRS),'catalogue_used':False,'duplicate_sweep':False}}
if __name__=='__main__':
 r=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r['stats'],sort_keys=True))
