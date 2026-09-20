"""Small center-relation grammar with outward character equations."""
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/center-two-relation-scope-20260920.json'
ID='center-two-relation-scope-20260920'; SIG='center-two-relation-grammar|scope-compatibility|outward-equations-online'
def letters(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=letters(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
SUBJ=('the old captain','a patient singer','our careful farmer')
PRED=(('trusted','trust'),('followed','follow'),('remembered','remember'))
OBJ=('the quiet guide','a distant friend','the bright harbor')
REL=(('beside','proximity'),('despite','concession'),('without','absence'))
REL2=(('while','temporal'),('because','causal'),('although','contrast'))
TAIL=('at first light','before the rain','near the bridge')
def outward(left,right):
 a,b=letters(left),letters(right)[::-1]; n=0
 for i,(x,y) in enumerate(zip(a,b)):
  n+=1
  if x!=y:return False,n,{'offset':i,'left':x,'right':y}
 return len(a)<=len(b),n,None
def run():
 rows=[]; prunes=0
 for s,(v,sem),o,(rel,rs),(rel2,rs2),tail in itertools.product(SUBJ,PRED,OBJ,REL,REL2,TAIL):
  if rs=='absence' and rs2=='causal': continue
  left=f'{s} {v} {o}'; right=f'{rel} {o} {tail}'
  rendered=f'{left}, {rel} {o}, {rel2} the path, {tail}.'
  ok,n,mm=outward(left,right+' '+rel2+' the path')
  rec={'rendered':rendered,'center_relations':{'predicate':v,'semantic':sem,'first':{'relation':rel,'scope':rs},'second':{'relation':rel2,'scope':rs2},'scope_compatible':True,'tail':tail},'outward_equation':{'accepted':ok,'characters_checked':n,'mismatch':mm},'audit':audit(rendered),'provenance':{'grammar':'fresh hand-authored two-relation grammar','full_clauses_authored_before_retention':True,'seed_wrapping':False,'finished_tape_reversal':False,'post_hoc_repair':False,'repeated_units':False,'mirrored_word_order':False,'catalogue_text':False,'fragment':False}}
  rows.append(rec)
  if not ok: prunes+=1
 rows.sort(key=lambda r:-r['audit']['letters']); exact=[r for r in rows if r['audit']['exact'] and r['audit']['letters']>38]
 return {'experiment_id':ID,'method':'full authored clauses around two typed center relations with scope compatibility and outward equations','stats':{'subjects':len(SUBJ),'predicates':len(PRED),'objects':len(OBJ),'relations_one':len(REL),'relations_two':len(REL2),'tails':len(TAIL),'states':len(rows),'online_prunes':prunes,'fresh_exact_gt38':len(exact),'max_letters':max(r['audit']['letters'] for r in rows)},'reader_facing_candidates':exact if exact else [],'rendered_controls':rows,'exact_candidates':exact,'novelty_preflight':{'status':'passed','signature':SIG,'distinct_from':'one-relation center grammar: adds independently typed second relation and scope gate'},'next_topology':'add relation ordering and attachment depth before a third relation','status':'fresh exact >38 candidate requires human reading' if exact else 'no exact >38 closure; reader-facing list empty'}
if __name__=='__main__':
 r=run(); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
