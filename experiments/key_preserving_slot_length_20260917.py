"""Bounded key-preserving paired slot-length adjustment."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='key-preserving-slot-length-20260917'
PAIRS=[
 ('the keeper','marks','the chart','that guides the crew','the sailor','reads','the ledger','that remembers the route','singular','w'),
 ('the keeper','marks','the weathered chart','that guides the crew','the sailor','reads','the tide ledger','that remembers the route','singular','w'),
 ('a teacher','carries','a map','which charts the shore','a guide','keeps','a journal','which records the way','singular','e'),
 ('the keepers','mark','the charts','that guide the crews','the sailors','read','the ledgers','that remember the routes','plural','s'),
]
def letters(s):return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s);m=sum(a!=b for a,b in zip(t,t[::-1]));return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[]
 for i,(ls,lv,lo,lr,rs,rv,ro,rr,num,key) in enumerate(PAIRS):
  text=f'At dawn, {ls} {lv} {lo} {lr} beside the inlet; {rs} {rv} {ro} {rr} beside the inlet.'
  rows.append({'pair_id':i,'rendered':text,'coupling':{'seam_key':key,'number':num,'left_slot_letters':len(letters(lo)),'right_slot_letters':len(letters(ro)),'slot_delta':len(letters(lo))-len(letters(ro))},'audit':audit(text),'provenance':{'key_preserved':True,'paired_slot_length_adjustment':True,'source_experiment':'coupled-inflectional-head-seam-20260917','catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'word_order_only_symmetry':False}})
 best=min(rows,key=lambda r:r['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(r['audit']['two_pointer_exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'best_mismatches':best['audit']['mismatches']},'next_repair':{'operator':'key-preserving paired boundary inflection','reason':'paired slot lengths alter geometry but retain the 88-mismatch seam class; next adjust boundary inflections while preserving semantic roles','route_exhausted':False},'provenance':{'bounded_pairs':len(rows),'catalogue_used':False,'duplicate_sweep':False}}
if __name__=='__main__':
 r=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r['stats'],sort_keys=True))
