"""Coupled determiner, number, relative-head, and verb alternatives."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='coupled-determiner-relative-agreement-20260917'
ROWS=[
 ('the keeper','marks','the chart','that guides the crew','the sailor','reads','the ledger','that remembers the route','singular'),
 ('a keeper','marks','a chart','which guides the crew','a sailor','reads','a ledger','which remembers the route','singular'),
 ('the keepers','mark','the charts','that guide the crews','the sailors','read','the ledgers','that remember the routes','plural'),
 ('keepers','mark','charts','which guide crews','sailors','read','ledgers','which remember routes','plural'),
]
def letters(s):return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s);m=sum(a!=b for a,b in zip(t,t[::-1]));return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[]
 for i,(ls,lv,lo,lr,rs,rv,ro,rr,num) in enumerate(ROWS):
  text=f'At dawn, {ls} {lv} {lo} {lr} beside the inlet; {rs} {rv} {ro} {rr} beside the inlet.'
  rows.append({'pair_id':i,'rendered':text,'agreement':{'number':num,'subject_and_relative_agree':True,'left_right_agree':True},'audit':audit(text),'provenance':{'coupled_determiner_inflection_relative':True,'same_semantic_frame':True,'catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'word_order_only_symmetry':False}})
 best=min(rows,key=lambda r:r['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(r['audit']['two_pointer_exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'best_mismatches':best['audit']['mismatches']},'next_repair':{'operator':'agreement-coupled semantic slot substitution at the first mismatch','reason':'full determiner and relative agreement remains grammatical but does not close the global tape; next change only a role-compatible semantic slot selected by the seam','route_exhausted':False},'provenance':{'bounded_rows':len(rows),'catalogue_used':False,'duplicate_sweep':False}}
if __name__=='__main__':
 r=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r['stats'],sort_keys=True))
