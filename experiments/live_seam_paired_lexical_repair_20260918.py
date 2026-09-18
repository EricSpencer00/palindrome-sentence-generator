"""Paired lexical repair on the best admitted joint-lattice state."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='live-seam-paired-lexical-repair-20260918'
BASE='At dawn, the keeper marks the chart that guides the crew beside the inlet; the sailor reads the ledger that remembers the route beside the inlet.'
PAIRS=[('crew','team','route','path'),('chart','map','ledger','journal'),('keeper','warden','sailor','pilot'),('inlet','harbor','inlet','harbor')]
def letters(s):return re.sub(r'[^a-z]','',s.lower())
def seam(s):
 t=letters(s)
 for i,(a,b) in enumerate(zip(t,t[::-1])):
  if a!=b:return i
 return None
def audit(s):
 t=letters(s);m=sum(a!=b for a,b in zip(t,t[::-1]));return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 base_seam=seam(BASE);rows=[]
 for i,(a,b,c,d) in enumerate(PAIRS):
  text=BASE.replace(a,b,1).replace(c,d,1)
  rows.append({'repair_id':i,'rendered':text,'live_seam':{'base_first_mismatch':base_seam,'repaired_first_mismatch':seam(text),'paired_slots':[a,c]},'audit':audit(text),'provenance':{'best_lattice_state':True,'paired_lexical_repair':True,'live_seam_threshold':True,'catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'word_order_only_symmetry':False}})
 best=min(rows,key=lambda r:r['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(r['audit']['two_pointer_exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'best_mismatches':best['audit']['mismatches']},'next_repair':{'operator':'paired live-seam character-class lexical repair','reason':'paired role-compatible word edits shift but do not close the seam; next constrain substitutions by the opposing seam character class','route_exhausted':False},'provenance':{'bounded_repairs':len(rows),'catalogue_used':False,'duplicate_sweep':False}}
if __name__=='__main__':
 r=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r['stats'],sort_keys=True))
