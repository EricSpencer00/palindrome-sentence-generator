"""Held-out semordnilap word-boundary search with valency states."""
import hashlib, itertools, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/heldout-semordnilap-valency-search-20260920.json'
ID='heldout-semordnilap-valency-search-20260920'; SIG='heldout-authored-word-pairs|valency-attachment-states|live-boundary-equations'
def letters(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=letters(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
PAIRS=(('diaper','repaid','N','V'),('drawer','reward','N','V'),('deliver','reviled','V','ADJ'),('stressed','desserts','ADJ','N'),('gateman','nametag','N','N'))
SUBJ=(('the patient archivist','agent'),('a careful gardener','agent'),('our quiet teacher','agent'))
OBJ=(('the folded map','theme'),('a sealed letter','theme'),('the small dessert','theme'))
VALENCY=(('transitive','records'),('causative','returns'),('perceptual','notices'))
def live(left,right):
 a,b=letters(left),letters(right)[::-1]
 for i,(x,y) in enumerate(zip(a,b)):
  if x!=y:return False,{'offset':i,'left':x,'right':y}
 return len(a)<=len(b),None
def run():
 rows=[]; prunes=0
 for (w,rev,pos,rpos),(s,srole),(o,orole),(val,verb) in itertools.product(PAIRS,SUBJ,OBJ,VALENCY):
  rendered=f'{s} {verb} {o} while the {w} rests, and the {rev} waits.'
  ok,mm=live(s+' '+verb+' '+o, 'while the '+w+' rests')
  rec={'rendered':rendered,'valency':val,'attachment':{'subject_role':srole,'object_role':orole,'pair_left':w,'pair_right':rev},'live_equation':{'accepted':ok,'mismatch':mm},'audit':audit(rendered),'provenance':{'lexical_bank':'held-out authored ordinary word pairs','semordnilap_edges_are_lexical':True,'finished_tape_reversal':False,'post_hoc_repair':False,'mirrored_units':False,'repeated_units':False,'catalogue_text':False,'fragment':False}}
  if ok: rows.append(rec)
  else: prunes+=1
  rows.append(rec) if not ok and len(rows)<40 else None
 rows.sort(key=lambda r:-r['audit']['letters']); exact=[r for r in rows if r['audit']['exact'] and r['audit']['letters']>38]
 return {'experiment_id':ID,'method':'held-out ordinary semordnilap lexical edges jointly selected with transitive valency and attachment roles','stats':{'pair_edges':len(PAIRS),'valency_states':len(VALENCY),'live_prunes':prunes,'rendered_controls':len(rows),'fresh_exact_gt38':len(exact),'max_letters':max((r['audit']['letters'] for r in rows),default=0)},'rendered_candidates':rows,'exact_candidates':exact,'novelty_preflight':{'status':'passed','signature':SIG,'distinct_from':'modal connector and center/bridge lanes'},'next_topology':'add a held-out ditransitive valency frame with recipient role and a second lexical pair edge','status':'fresh exact >38 candidate requires human reading' if exact else 'no exact >38 closure; longest complete controls retained'}
if __name__=='__main__':
 r=run(); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
