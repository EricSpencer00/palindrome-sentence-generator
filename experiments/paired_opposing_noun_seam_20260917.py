"""Paired opposing semantic-noun substitutions at one seam index."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='paired-opposing-noun-seam-20260917'
BASE='At dawn, the keeper marks the chart that guides the crew beside the inlet; the sailor reads the ledger that remembers the route beside the inlet.'
PAIRS=[(('chart','map'),('ledger','journal')),(('chart','plan'),('ledger','book')),(('keeper','steward'),('sailor','seaman')),(('keeper','warden'),('sailor','pilot'))]
def letters(s):return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s);m=sum(a!=b for a,b in zip(t,t[::-1]));return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[]
 for i,((a,b),(c,d)) in enumerate(PAIRS):
  text=BASE.replace(a,b,1).replace(c,d,1)
  rows.append({'pair_id':i,'rendered':text,'seam_selection':{'left_slot':a,'right_slot':c,'coordinated_same_seam':True,'length_delta_left':len(letters(b))-len(letters(a)),'length_delta_right':len(letters(d))-len(letters(c))},'audit':audit(text),'provenance':{'two_sided_coordination':True,'agreement_preserved':True,'source_experiment':'seam-selected-semantic-slot-20260917','catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'word_order_only_symmetry':False}})
 best=min(rows,key=lambda r:r['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(r['audit']['two_pointer_exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'best_mismatches':best['audit']['mismatches']},'next_repair':{'operator':'paired noun and relative-verb substitution with agreement','reason':'two-sided noun coordination preserves the scene but does not close the tape; next coordinate nouns with their relative verbs at the same seam','route_exhausted':False},'provenance':{'bounded_pairs':len(rows),'catalogue_used':False,'duplicate_sweep':False}}
if __name__=='__main__':
 r=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r['stats'],sort_keys=True))
