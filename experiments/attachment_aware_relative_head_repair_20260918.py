"""Attachment-aware relative-head substitutions under paired role constraints."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='attachment-aware-relative-head-repair-20260918'
ROWS=[('chart','guides the crew','ledger','remembers the route','beside the inlet'),('map','marks the shore','journal','records the way','near the harbor'),('plan','guards the pier','book','watches the quay','along the coast'),('stars','names the sky','paths','notes the trails','by the garden wall')]
def letters(s):return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s);m=sum(a!=b for a,b in zip(t,t[::-1]));return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[]
 for i,(lo,lv,ro,rv,boundary) in enumerate(ROWS):
  text=f'At dawn, the keeper marks the {lo} that {lv} {boundary}; the sailor reads the {ro} that {rv} {boundary}.'
  rows.append({'pair_id':i,'rendered':text,'attachment':{'left':'location','right':'location','relative_head':'that','same_role':True},'audit':audit(text),'provenance':{'attachment_aware_relative_head':True,'source_experiment':'head-verb-boundary-attachment-20260918','catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'word_order_only_symmetry':False}})
 best=min(rows,key=lambda r:r['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(r['audit']['two_pointer_exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'best_mismatches':best['audit']['mismatches']},'next_repair':{'operator':'paired attachment-aware relative-head and verb substitution','reason':'head alternatives preserve location attachment but do not close the tape; next coordinate relative heads and verbs while retaining the role gate','route_exhausted':False},'provenance':{'bounded_rows':len(rows),'catalogue_used':False}}
if __name__=='__main__':
 r=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r['stats'],sort_keys=True))
