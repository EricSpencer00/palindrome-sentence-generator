"""Paired semantic nouns and relative verbs changed together at one seam."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='paired-noun-relative-verb-seam-20260917'
ROWS=[
 ('the chart','guides the crew','the ledger','remembers the route','singular'),
 ('the map','marks the shore','the journal','records the way','singular'),
 ('the charts','guide the crews','the ledgers','remember the routes','plural'),
 ('maps','mark shores','journals','record ways','plural'),
]
def letters(s):return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s);m=sum(a!=b for a,b in zip(t,t[::-1]));return {'letters':len(t),'two_pointer_exact':bool(t) and m==0,'mismatches':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[]
 for i,(lo,lv,ro,rv,num) in enumerate(ROWS):
  head='that' if num=='singular' else 'which'
  text=f'At dawn, the keeper marks {lo} {head} {lv} beside the inlet; the sailor reads {ro} {head} {rv} beside the inlet.'
  rows.append({'pair_id':i,'rendered':text,'coupling':{'left_noun':lo,'left_relative_verb':lv,'right_noun':ro,'right_relative_verb':rv,'number':num,'same_seam':True},'audit':audit(text),'provenance':{'noun_and_relative_verb_coupled':True,'agreement_preserved':True,'source_experiment':'paired-opposing-noun-seam-20260917','catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'word_order_only_symmetry':False}})
 best=min(rows,key=lambda r:r['audit']['mismatches'])
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(r['audit']['two_pointer_exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows),'best_mismatches':best['audit']['mismatches']},'next_repair':{'operator':'paired noun-verb-relative-head substitution with boundary balancing','reason':'noun and relative verb coupling remains grammatical but does not close the tape; next include relative-head and boundary length constraints','route_exhausted':False},'provenance':{'bounded_rows':len(rows),'catalogue_used':False,'duplicate_sweep':False}}
if __name__=='__main__':
 r=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r['stats'],sort_keys=True))
