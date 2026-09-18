"""Bounded inflectional alternatives within three-letter seam/agreement classes."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='inflectional-same-seam-20260917'
LEFT=[('the keeper','marks','the chart','that guides the crew','singular'),('a teacher','carries','a map','which charts the shore','singular'),('the keepers','mark','the charts','that guide the crews','plural'),('teachers','carry','maps','which chart the shores','plural')]
RIGHT=[('the sailor','reads','the ledger','that remembers the route','singular'),('a guide','keeps','a journal','which records the way','singular'),('the sailors','read','the ledgers','that remember the routes','plural'),('guides','keep','journals','which record the ways','plural')]
def letters(s):return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s);m=sum(a!=b for a,b in zip(t,t[::-1]));return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[];rejected=0
 for li,(ls,lv,lo,lr,ln) in enumerate(LEFT):
  for ri,(rs,rv,ro,rr,rn) in enumerate(RIGHT):
   if letters(lr)[:3]!=letters(rr)[:3] or ln!=rn:rejected+=1;continue
   text=f'At dawn, {ls} {lv} {lo} {lr} beside the inlet; {rs} {rv} {ro} {rr} beside the inlet.'
   rows.append({'left_slot':li,'right_slot':ri,'rendered':text,'class':{'signature':letters(lr)[:3],'number':ln},'audit':audit(text),'provenance':{'same_seam_class':True,'inflectional_alternatives':True,'pre_render_filter':True,'source_experiment':'three-letter-agreement-csp-20260917','catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'word_order_only_symmetry':False}})
 best=min(rows,key=lambda r:r['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'rendered':len(rows),'rejected_pre_render':rejected,'exact':sum(r['audit']['two_pointer_exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'best_mismatches':best['audit']['mismatches']},'next_repair':{'operator':'slot-level agreement lattice with seam-conditioned lexical heads','reason':'inflectional alternatives within the same seam class preserve readable grammar but do not close the tape; next couple head choice to opposing seam letters','route_exhausted':False},'provenance':{'bounded_left_slots':len(LEFT),'bounded_right_slots':len(RIGHT),'rejected_before_render':rejected,'catalogue_used':False}}
if __name__=='__main__':
 r=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r['stats'],sort_keys=True))
