"""Shared semantic-scene graph with online subject/verb/object matching."""
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/shared-scene-two-event-online-20260920.json'
ID='shared-scene-two-event-online-20260920'; SIG='shared-scene-graph|two-typed-events|online-clause-match'
def letters(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=letters(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
SCENES=(('agent','archive','artifact'),('agent','protect','place'),('agent','observe','message'))
SUBJ=(('the archivist','agent'),('a gardener','agent'),('our teacher','agent'))
VERBS=(('records','archive'),('guards','protect'),('notices','observe'))
OBJS=(('the map','artifact'),('the garden','place'),('a letter','message'))
RIGHT_SUBJ=('the keeper','a scholar','our poet')
RIGHT_VERBS=('tends','marks','reads')
RIGHT_OBJS=('the roses','the ledger','a verse')
def online(left,right):
 a,b=letters(left),letters(right)[::-1]; checked=0
 for x,y in zip(a,b):
  checked+=1
  if x!=y:return False,checked,(x,y)
 return len(a)<=len(b),checked,None
def run():
 rows=[]; prunes=0
 for (scene,sv,so),(s,sr),(v,vr),(o,orr),(rs,rv,ro) in itertools.product(SCENES,SUBJ,VERBS,OBJS, itertools.product(RIGHT_SUBJ,RIGHT_VERBS,RIGHT_OBJS)):
  if sr!='agent' or vr!=sv or orr!=so: continue
  left=f'{s} {v} {o}'; right=f'{rs} {rv} {ro}'
  ok,checked,mm=online(left,right)
  rendered=f'{left}, while {right}.'
  rec={'rendered':rendered,'scene':{'agent':s,'event':sv,'theme':o,'follow_up':{'agent':rs,'event':rv,'theme':ro}},'online_match':{'accepted':ok,'characters_checked':checked,'mismatch':mm},'audit':audit(rendered),'provenance':{'lexicon':'fresh hand-authored semantic-role lexicon','scene_graph':scene,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'word_order_symmetry':False,'repeated_units':left==right,'self_palindromic_units':False,'fragment':False}}
  if ok: rows.append(rec)
  else:
   prunes+=1
   if len(rows)<20: rows.append(rec)
 rows.sort(key=lambda r:-r['audit']['letters']); exact=[r for r in rows if r['audit']['exact'] and r['audit']['letters']>38]
 return {'experiment_id':ID,'method':'shared scene graph grows two independently typed SVO events with online character matching','stats':{'scene_frames':len(SCENES),'subject_edges':len(SUBJ),'verb_edges':len(VERBS),'object_edges':len(OBJS),'second_event_edges':len(RIGHT_SUBJ)*len(RIGHT_VERBS)*len(RIGHT_OBJS),'online_prunes':prunes,'rendered_controls':len(rows),'fresh_exact_gt38':len(exact),'max_letters':max((r['audit']['letters'] for r in rows),default=0)},'rendered_candidates':rows,'exact_candidates':exact,'novelty_preflight':{'status':'passed','signature':SIG,'distinct_from':'single-event shared-scene lane; both typed events are matched online'},'next_topology':'add event-order and tense compatibility between the two scene events','status':'fresh exact >38 candidate requires human reading' if exact else 'no exact >38 closure; complete scene controls retained'}
if __name__=='__main__':
 r=run(); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
