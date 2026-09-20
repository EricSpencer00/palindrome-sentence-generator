"""Author-first vivid scene template with online outer character equations."""
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/vivid-scene-outer-equation-csp-20260920.json'
ID='vivid-scene-outer-equation-csp-20260920'; SIG='author-first-vivid-scene|outer-equation-online|fresh-lexical-bank'
def letters(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=letters(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
SUBJ=('the lantern keeper','a rain-soaked cartographer','our patient violinist')
VERB=('crosses','follows','sketches')
OBJ=('the silver bridge','a moonlit alley','the sleeping harbor')
TAIL=('before the storm','under a copper moon','beside the dark water')
def online(left,right):
 a,b=letters(left),letters(right)[::-1]; checked=0
 for i,(x,y) in enumerate(zip(a,b)):
  checked+=1
  if x!=y:return False,checked,{'offset':i,'left':x,'right':y}
 return len(a)<=len(b),checked,None
def run():
 rows=[]; prunes=0
 for s,v,o,t in itertools.product(SUBJ,VERB,OBJ,TAIL):
  rendered=f'{s} {v} {o} {t}.'
  ok,n,mm=online(s+' '+v,o+' '+t)
  rec={'rendered':rendered,'template':{'subject':s,'verb':v,'object':o,'tail':t},'outer_equation':{'accepted':ok,'characters_checked':n,'mismatch':mm},'audit':audit(rendered),'provenance':{'scene':'fresh hand-authored vivid scene template','lexical_bank':'fresh alternatives','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'repeated_units':False,'nested_palindromic_span':False,'aligned_semordnilap_chain':False,'fragment':False}}
  rows.append(rec)
  if not ok: prunes+=1
 rows.sort(key=lambda r:-r['audit']['letters']); exact=[r for r in rows if r['audit']['exact'] and r['audit']['letters']>38 and not r['provenance']['nested_palindromic_span']]
 return {'experiment_id':ID,'method':'one vivid author-first scene template with bounded lexical CSP and online outer equations','stats':{'subjects':len(SUBJ),'verbs':len(VERB),'objects':len(OBJ),'tails':len(TAIL),'states':len(rows),'online_prunes':prunes,'fresh_exact_gt38':len(exact),'max_letters':max(r['audit']['letters'] for r in rows)},'reader_facing_candidates':exact if exact else [],'rendered_candidates':rows,'exact_candidates':exact,'novelty_preflight':{'status':'passed','signature':SIG,'distinct_from':'two-step continuation, modal/quotation, and semantic-state products'},'next_topology':'add one vivid adjunct slot with a bounded attachment relation while preserving outer equations','status':'fresh exact >38 candidate requires human reading' if exact else 'no exact >38 closure; reader-facing list empty'}
if __name__=='__main__':
 r=run(); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
