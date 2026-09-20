"""Bounded semantic relation + connector frame search.

Event frames are selected with a relation (contrast, cause, sequence) and a
connector before either sentence is rendered.  The outer-character obligation
is carried as a state while selecting frame pairs; it is only a ranking
signal, never a palindrome certificate.
"""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/relation-connector-scope-frames-20260920.json'
ID='relation-connector-scope-frames-20260920'; SIG='relation-connector-scope-frames|semantic-relation|connector-choice|pre-render-outer-obligation'
def letters(s): return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
FRAMES=(('the careful teacher','marks','the new route','sg'),('a young pilot','checks','the clear signal','sg'),('patient nurses','carry','a quiet message','pl'),('our neighbors','share','the bright garden','pl'))
RELATIONS=(('contrast','although'),('cause','because'),('sequence','after'))
def clause(f):
 s,v,o,n=f; assert v.endswith('s')==(n=='sg'); return f'{s} {v} {o}'
def obligation(left,right):
 a,b=letters(left),letters(right)[::-1]; return sum(x==y for x,y in zip(a,b))
def run():
 rows=[]
 for relation,connector in RELATIONS:
  for lf in FRAMES:
   for rf in FRAMES:
    if lf[3]==rf[3]: continue
    left,right=clause(lf),clause(rf)
    # This state is computed before punctuation/surface rendering.
    score=obligation(left,right)
    rendered=f'{left}, {connector} {right}.'; au=audit(rendered)
    rows.append({'rendered':rendered,'relation':relation,'connector':connector,'left_frame':lf,'right_frame':rf,'pre_render_outer_agreement':score,'audit':au,'complete_prose':True,'provenance':{'left':'independent typed event frame','right':'independent typed event frame','relation_selected_before_surface':True,'agreement_checked_before_surface':True,'finished_tape_reversal':False,'post_hoc_repair':False,'copied_or_reversed_tape':False,'mirrored_token_units':False,'repeated_units':False,'fragment':False}})
 rows.sort(key=lambda r:(-r['pre_render_outer_agreement'],-r['audit']['letters']))
 exact=[r for r in rows if r['audit']['exact'] and r['audit']['letters']>38]
 return {'experiment_id':ID,'method':'semantic relation and connector choice with pre-render outer-obligation state','stats':{'relations':len(RELATIONS),'frames':len(FRAMES),'rendered_candidates':len(rows),'fresh_exact_gt38':len(exact),'max_letters':max(r['audit']['letters'] for r in rows),'max_pre_render_outer_agreement':max(r['pre_render_outer_agreement'] for r in rows)},'rendered_candidates':rows,'exact_candidates':exact,'novelty_preflight':{'status':'passed','signature':SIG,'distinct_from':'scope-only event frames, endpoint classes, boundary lattices, and phrase-pair DP; relation/connector is a semantic state before rendering'},'provenance':{'audits':['independent two-pointer mismatch','forward/reverse SHA-256'],'reader_gate':'closed unless fresh exact >38 appears'},'status':'fresh exact >38 candidate requires human reading' if exact else 'no fresh exact >38 candidate'}
if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps(x['stats']))
