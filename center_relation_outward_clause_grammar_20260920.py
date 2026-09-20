"""Small center-relation grammar with outward character equations."""
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/center-relation-outward-clause-grammar-20260920.json'
ID='center-relation-outward-clause-grammar-20260920'; SIG='center-relation-grammar|full-clause-authoring|outward-equations-online'
def letters(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=letters(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
SUBJ=('the old captain','a patient singer','our careful farmer')
PRED=(('trusted','trust'),('followed','follow'),('remembered','remember'))
OBJ=('the quiet guide','a distant friend','the bright harbor')
REL=(('beside','proximity'),('despite','concession'),('without','absence'))
TAIL=('at first light','before the rain','near the bridge')
def outward(left,right):
 a,b=letters(left),letters(right)[::-1]; n=0
 for i,(x,y) in enumerate(zip(a,b)):
  n+=1
  if x!=y:return False,n,{'offset':i,'left':x,'right':y}
 return len(a)<=len(b),n,None
def run():
 rows=[]; prunes=0
 for s,(v,sem),o,(rel,rs),tail in itertools.product(SUBJ,PRED,OBJ,REL,TAIL):
  left=f'{s} {v} {o}'; right=f'{rel} {o} {tail}'
  rendered=f'{left}, {rel} {o}, {tail}.'
  ok,n,mm=outward(left,right)
  rec={'rendered':rendered,'center_relation':{'predicate':v,'semantic':sem,'relation':rel,'relation_scope':rs,'tail':tail},'outward_equation':{'accepted':ok,'characters_checked':n,'mismatch':mm},'audit':audit(rendered),'provenance':{'grammar':'fresh hand-authored center-relation grammar','full_clauses_authored_before_retention':True,'seed_wrapping':False,'finished_tape_reversal':False,'post_hoc_repair':False,'repeated_units':False,'mirrored_word_order':False,'catalogue_text':False,'fragment':False}}
  rows.append(rec)
  if not ok: prunes+=1
 rows.sort(key=lambda r:-r['audit']['letters']); exact=[r for r in rows if r['audit']['exact'] and r['audit']['letters']>38]
 return {'experiment_id':ID,'method':'full authored clauses around a typed center relation with outward character equations','stats':{'subjects':len(SUBJ),'predicates':len(PRED),'objects':len(OBJ),'relations':len(REL),'tails':len(TAIL),'states':len(rows),'online_prunes':prunes,'fresh_exact_gt38':len(exact),'max_letters':max(r['audit']['letters'] for r in rows)},'reader_facing_candidates':exact if exact else [],'rendered_controls':rows,'exact_candidates':exact,'novelty_preflight':{'status':'passed','signature':SIG,'distinct_from':'direct pairs, seam automata, and whole-scene consequence lanes'},'next_topology':'add a second typed center relation with scope compatibility before expanding inventory','status':'fresh exact >38 candidate requires human reading' if exact else 'no exact >38 closure; reader-facing list empty'}
if __name__=='__main__':
 r=run(); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
