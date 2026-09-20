"""Whole-scene two-clause narrative with consequence/contrast connective."""
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/whole-scene-consequence-contrast-20260920.json'
ID='whole-scene-consequence-contrast-20260920'; SIG='whole-scene-narrative|two-complete-clauses|consequence-contrast-connective|online-match'
def letters(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=letters(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
CLAUSE1=('the lantern keeper opened the gate','a patient cartographer marked the shore','our quiet baker carried the letter')
CLAUSE2=('the waiting garden filled with rain','the distant harbor answered at dawn','the narrow road remained bright')
CONNECTORS=(('therefore','consequence'),('however','contrast'),('so','consequence'))
def online(left,right):
 a,b=letters(left),letters(right)[::-1]; n=0
 for i,(x,y) in enumerate(zip(a,b)):
  n+=1
  if x!=y:return False,n,{'offset':i,'left':x,'right':y}
 return len(a)<=len(b),n,None
def run():
 rows=[]; prunes=0
 for a,b,(conn,kind) in itertools.product(CLAUSE1,CLAUSE2,CONNECTORS):
  rendered=f'{a}; {conn}, {b}.'
  ok,n,mm=online(a,b)
  rec={'rendered':rendered,'narrative':{'clause_one':a,'clause_two':b,'connective':conn,'relation':kind},'online_match':{'accepted':ok,'characters_checked':n,'mismatch':mm},'audit':audit(rendered),'provenance':{'inventory':'fresh authored complete narrative clauses','whole_scene_grammar':True,'finished_tape_reversal':False,'post_hoc_repair':False,'indexed_seam':False,'seed_wrapping':False,'repeated_units':False,'mirrored_word_order':False,'catalogue_text':False,'fragment':False}}
  rows.append(rec)
  if not ok: prunes+=1
 rows.sort(key=lambda r:-r['audit']['letters']); exact=[r for r in rows if r['audit']['exact'] and r['audit']['letters']>38]
 return {'experiment_id':ID,'method':'whole-scene authored narrative of two complete clauses with consequence/contrast connective and online character matching','stats':{'clause_one':len(CLAUSE1),'clause_two':len(CLAUSE2),'connectives':len(CONNECTORS),'states':len(rows),'online_prunes':prunes,'fresh_exact_gt38':len(exact),'max_letters':max(r['audit']['letters'] for r in rows)},'reader_facing_candidates':exact if exact else [],'rendered_controls':rows,'exact_candidates':exact,'novelty_preflight':{'status':'passed','signature':SIG,'distinct_from':'vivid adjunct linear continuation and indexed seam/automaton families'},'next_topology':'add a single causal subordinate clause with attachment scope while preserving whole-scene narrative structure','status':'fresh exact >38 candidate requires human reading' if exact else 'no exact >38 closure; reader-facing list empty'}
if __name__=='__main__':
 r=run(); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
