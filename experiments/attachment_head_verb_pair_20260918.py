"""Paired attachment-aware relative-head and verb substitutions."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='attachment-head-verb-pair-20260918'
ROWS=[('chart','guides the crew','ledger','remembers the route','beside the inlet','that'),('map','marks the shore','journal','records the way','near the harbor','which'),('plan','guards the pier','book','watches the quay','along the coast','that'),('stars','names the sky','paths','notes the trails','by the garden wall','which')]
def letters(s):return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s);m=sum(a!=b for a,b in zip(t,t[::-1]));return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[]
 for i,(lo,lv,ro,rv,boundary,head) in enumerate(ROWS):
  text=f'At dawn, the keeper marks the {lo} {head} {lv} {boundary}; the sailor reads the {ro} {head} {rv} {boundary}.'
  rows.append({'pair_id':i,'rendered':text,'attachment':{'left':'location','right':'location','relative_head':head,'head_verb_coupled':True,'same_role':True},'audit':audit(text),'provenance':{'attachment_aware_head_verb_pair':True,'source_experiment':'attachment-aware-relative-head-repair-20260918','catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'word_order_only_symmetry':False}})
 best=min(rows,key=lambda r:r['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(r['audit']['two_pointer_exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'best_mismatches':best['audit']['mismatches']},'next_repair':{'operator':'attachment-aware paired clause-length balancing','reason':'head and verb coupling preserves role attachment but does not close the tape; next balance the two relative clauses while retaining the paired attachment','route_exhausted':False},'provenance':{'bounded_rows':len(rows),'catalogue_used':False}}
if __name__=='__main__':
 r=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r['stats'],sort_keys=True))
