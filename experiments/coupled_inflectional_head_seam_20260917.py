"""Coupled semantic-head choices keyed by opposing seam letters."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='coupled-inflectional-head-seam-20260917'
LEFT=[('the keeper','marks','the chart','that guides the crew','singular','w'),('a teacher','carries','a map','which charts the shore','singular','e'),('the keepers','mark','the charts','that guide the crews','plural','s'),('teachers','carry','maps','which chart the shores','plural','s')]
RIGHT=[('the sailor','reads','the ledger','that remembers the route','singular','w'),('a guide','keeps','a journal','which records the way','singular','e'),('the sailors','read','the ledgers','that remember the routes','plural','s'),('guides','keep','journals','which record the ways','plural','s')]
def letters(s):return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s);m=sum(a!=b for a,b in zip(t,t[::-1]));return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[];rejected=0
 for li,(ls,lv,lo,lr,ln,key) in enumerate(LEFT):
  for ri,(rs,rv,ro,rr,rn,rkey) in enumerate(RIGHT):
   if key!=rkey or ln!=rn:rejected+=1;continue
   text=f'At dawn, {ls} {lv} {lo} {lr} beside the inlet; {rs} {rv} {ro} {rr} beside the inlet.'
   rows.append({'left_slot':li,'right_slot':ri,'rendered':text,'coupling':{'opposing_head_key':key,'left_number':ln,'right_number':rn,'satisfied':True},'audit':audit(text),'provenance':{'coupled_head_choice':True,'pre_render_filter':True,'source_experiment':'inflectional-same-seam-20260917','catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'word_order_only_symmetry':False}})
 best=min(rows,key=lambda r:r['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'rendered':len(rows),'rejected_pre_render':rejected,'exact':sum(r['audit']['two_pointer_exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'best_mismatches':best['audit']['mismatches']},'next_repair':{'operator':'paired semantic-slot substitution with key-preserving length repair','reason':'coupling opposing head choices keeps prose grammatical but does not close the global tape; next preserve the key while adjusting paired slot lengths','route_exhausted':False},'provenance':{'bounded_left_slots':len(LEFT),'bounded_right_slots':len(RIGHT),'rejected_before_render':rejected,'catalogue_used':False}}
if __name__=='__main__':
 r=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r['stats'],sort_keys=True))
