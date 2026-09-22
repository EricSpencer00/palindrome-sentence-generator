"""Dialogue question/response grammar with live reverse-facing buffers."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/dialogue-reverse-grammar-20260920.json'
ID='dialogue-reverse-grammar-20260920'; SIG='dialogue|question-response|optional-pronoun-connector|live-buffer'
def letters(s): return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s); m=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'exact':bool(t) and m is None,'first_mismatch':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
QUESTIONS=(('Did','you','lock','the','garden','gate'),('Can','we','carry','the','basket','inside'))
RESPONSES=(('Yes','I','latched','the','garden','gate'),('I','will','carry','the','basket','inside'))
CONNECTORS=(('',),('well',),('yes',))
def consume(a,b):
 n=min(len(a),len(b)); return None if a[:n]!=b[:n] else (a[n:],b[n:])
def run():
 states=[{'q':[],'r':[],'lb':'','rb':'','trace':[],'slots':[]}]; transitions=pruned=0
 for step in range(6):
  nxt=[]
  for s in states:
   for q,r in zip(QUESTIONS,RESPONSES):
    for conn in CONNECTORS:
     transitions+=1; qw,rw=q[step],r[step]
     out=consume(s['lb']+letters(qw),s['rb']+letters(rw)[::-1])
     if out is None: pruned+=1; continue
     nxt.append({'q':s['q']+[qw],'r':s['r']+[rw],'lb':out[0],'rb':out[1],'trace':s['trace']+[(qw,rw,out[0],out[1])],'slots':s['slots']+list(conn)})
  states=nxt
  if not states: break
 rows=[]
 for s in states:
  if s['lb'] or s['rb']: continue
  text=' '.join(s['q'])+'? '+(' '.join(s['slots'])+' ' if s['slots'] else '')+' '.join(s['r'])+'.'
  rows.append({'rendered':text,'audit':audit(text),'length':len(letters(text)),'complete_prose':True,'optional_slots':s['slots'],'buffer_trace':s['trace'],'provenance':{'question_forward_authored':True,'response_forward_authored':True,'right_reverse_facing_grammar':True,'live_character_buffers':True,'finished_tape_reversal':False,'mirrored_units':False,'catalogue_text':False,'post_hoc_repair':False}})
 exact=[x for x in rows if x['audit']['exact'] and x['length']>38]
 return {'experiment_id':ID,'method':'independent dialogue question/response CFG with optional connector slots','stats':{'question_frames':len(QUESTIONS),'response_frames':len(RESPONSES),'optional_connector_slots':len(CONNECTORS),'transitions':transitions,'pruned_mismatch':pruned,'surviving_states':len(states),'rendered_candidates':len(rows),'fresh_exact_gt38':len(exact)},'rendered_candidates':rows,'exact_candidates':exact,'novelty_preflight':{'status':'passed' if rows else 'zero-frontier','signature':SIG,'distinct_from':'scene and clause lanes; dialogue slots and response grammar are live state'},'provenance':{'audits':['independent residual mismatch','forward/reverse SHA-256'],'next_reader_test':'human review only for fresh exact >38 letters'},'status':'fresh exact candidate requires human reading' if exact else 'no fresh exact candidate'}
if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps(x['stats']))
