"""Paired determiner and inflection alternatives within one semantic frame."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='paired-determiner-inflection-20260917'
PAIRS=[
 ('the keeper','marks','the chart','the sailor','reads','the ledger','singular'),
 ('a keeper','marks','a chart','a sailor','reads','a ledger','singular'),
 ('the keepers','mark','the charts','the sailors','read','the ledgers','plural'),
 ('keepers','mark','charts','sailors','read','ledgers','plural'),
]
def letters(s):return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s);m=sum(a!=b for a,b in zip(t,t[::-1]));return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[]
 for i,(ls,lv,lo,rs,rv,ro,num) in enumerate(PAIRS):
  text=f'At dawn, {ls} {lv} {lo} that guides the crew beside the inlet; {rs} {rv} {ro} that remembers the route beside the inlet.'
  rows.append({'pair_id':i,'rendered':text,'agreement':{'number':num,'left_right_agree':True},'audit':audit(text),'provenance':{'paired_determiner_change':True,'paired_inflection_change':True,'same_semantic_frame':True,'catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'word_order_only_symmetry':False}})
 best=min(rows,key=lambda r:r['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(r['audit']['two_pointer_exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'best_mismatches':best['audit']['mismatches']},'next_repair':{'operator':'paired determiner-inflection plus relative-clause agreement','reason':'determiner and number changes preserve the scene but leave a large residual; next couple these changes to agreeing relative-clause heads and verbs','route_exhausted':False},'provenance':{'bounded_pairs':len(rows),'catalogue_used':False,'duplicate_sweep':False}}
if __name__=='__main__':
 r=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r['stats'],sort_keys=True))
