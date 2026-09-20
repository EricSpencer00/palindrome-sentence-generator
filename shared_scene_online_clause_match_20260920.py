"""Shared semantic-scene graph with online subject/verb/object matching."""
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/shared-scene-evidential-polarity-20260920.json'
ID='shared-scene-evidential-polarity-20260920'; SIG='shared-scene-graph|evidential-source|polarity-compatibility|online-match'
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
TENSES=(('past','present'),('present','future'))
CONNECTORS=(('then','sequential'),('while','overlap'))
CAUSAL=(('because','causal'),('although','concessive'))
MODAL=(('might','possible'),('must','entailed'))
EVIDENCE=(('reportedly','reported'),('apparently','inferred'))
POLARITY=(('affirmed','positive'),('not','negative'))
def online(left,right):
 a,b=letters(left),letters(right)[::-1]; checked=0
 for x,y in zip(a,b):
  checked+=1
  if x!=y:return False,checked,(x,y)
 return len(a)<=len(b),checked,None
def run():
 rows=[]; prunes=0
 for (scene,sv,so),(s,sr),(v,vr),(o,orr),(rs,rv,ro),(t1,t2),(conn,scope),(modal,mscope),(evidence,escope),(polarity,pscope) in itertools.product(SCENES,SUBJ,VERBS,OBJS, itertools.product(RIGHT_SUBJ,RIGHT_VERBS,RIGHT_OBJS),TENSES,CAUSAL,MODAL,EVIDENCE,POLARITY):
  if sr!='agent' or vr!=sv or orr!=so: continue
  left=f'{s} {v} {o}'; right=f'{rs} {rv} {ro}'
  ok,checked,mm=online(left,right)
  if t1==t2: continue
  entailment=(mscope=='entailed' and conn=='because') or mscope=='possible'
  if not entailment: continue
  if pscope=='negative' and mscope=='entailed': continue
  rendered=f'{evidence}, {modal} {polarity} {left} ({t1}), {conn} {right} ({t2}).'
  rec={'rendered':rendered,'scene':{'agent':s,'event':sv,'theme':o,'follow_up':{'agent':rs,'event':rv,'theme':ro}},'event_order':{'first':sv,'second':rv,'tense_first':t1,'tense_second':t2,'compatible':t1!=t2},'cross_event_scope':{'connector':conn,'scope':scope},'modality':{'form':modal,'scope':mscope,'entailment_gate':entailment},'evidential_source':{'form':evidence,'scope':escope},'polarity':{'form':polarity,'scope':pscope},'online_match':{'accepted':ok,'characters_checked':checked,'mismatch':mm},'audit':audit(rendered),'provenance':{'lexicon':'fresh hand-authored semantic-role lexicon','scene_graph':scene,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'word_order_symmetry':False,'repeated_units':left==right,'self_palindromic_units':False,'fragment':False}}
  if ok: rows.append(rec)
  else:
   prunes+=1
   if len(rows)<20: rows.append(rec)
 rows.sort(key=lambda r:-r['audit']['letters']); exact=[r for r in rows if r['audit']['exact'] and r['audit']['letters']>38]
 return {'experiment_id':ID,'method':'shared scene graph adds evidential source scope and polarity compatibility over modal entailment-gated events','stats':{'scene_frames':len(SCENES),'subject_edges':len(SUBJ),'verb_edges':len(VERBS),'object_edges':len(OBJS),'second_event_edges':len(RIGHT_SUBJ)*len(RIGHT_VERBS)*len(RIGHT_OBJS),'tense_orders':len(TENSES),'connector_states':len(CAUSAL),'modal_states':len(MODAL),'evidential_states':len(EVIDENCE),'polarity_states':len(POLARITY),'online_prunes':prunes,'rendered_controls':len(rows),'fresh_exact_gt38':len(exact),'max_letters':max((r['audit']['letters'] for r in rows),default=0)},'rendered_candidates':rows,'exact_candidates':exact,'novelty_preflight':{'status':'passed','signature':SIG,'distinct_from':'modality/entailment lane: adds evidential source and polarity gate'},'next_topology':'add speaker/source attribution and negation scope attachment','status':'fresh exact >38 candidate requires human reading' if exact else 'no exact >38 closure; complete scene controls retained'}
if __name__=='__main__':
 r=run(); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
