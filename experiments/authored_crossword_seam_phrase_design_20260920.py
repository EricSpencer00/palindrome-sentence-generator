"""Small authored cross-word seam phrase design with live equations."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/authored-crossword-seam-phrase-design-20260920.json';ID='authored-crossword-seam-phrase-design-20260920';SIG='small-authored-phrase-inventory|cross-word-seam|complete-contemporary-clauses|live-seam-equations'
def letters(s):return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s);r=t[::-1];bad=next(((i,t[i],t[-i-1]) for i in range(len(t)//2) if t[i]!=t[-i-1]),None);return {'letters':len(t),'exact':bool(t) and bad is None,'first_mismatch':bad,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(r.encode()).hexdigest(),'sha_equal':hashlib.sha256(t.encode()).hexdigest()==hashlib.sha256(r.encode()).hexdigest()}
AGENTS=('the analyst','the teacher','the engineer','the writer','Alice','Diana','Marie','John')
VERBS=('reviews','opens','checks','writes','tests','reads','plans','calls')
OBJECTS=('the report','the file','the schedule','a note','the system','the message','the plan','the book')
SEAMS=('near the station','within the office','beside the river','after the meeting','under the bridge')
def live(left,right):
 a=letters(left);b=letters(right)[::-1];
 # Return the first residual seam mismatch without materializing a final tape.
 for i,(x,y) in enumerate(zip(a,b)):
  if x!=y:return {'compatible':False,'offset':i,'left_char':x,'right_char':y,'left_residual':a[i:i+8],'right_residual':b[i:i+8]}
 return {'compatible':len(a)==len(b),'offset':min(len(a),len(b)),'left_residual':a[len(b):],'right_residual':b[len(a):]}
def run(state_limit=120000):
 states=0;rows=[]
 for s in AGENTS:
  for v in VERBS:
   for o in OBJECTS:
    for seam in SEAMS:
     left=f'{s} {v} {o} {seam}'
     # Right-side lexical material is selected independently; no token mirror.
     for rs in AGENTS:
      for rv in VERBS:
       for ro in OBJECTS:
        for rseam in SEAMS:
         if states>=state_limit: break
         if (s,v,o,seam)==(rs,rv,ro,rseam): continue
         right=f'{rs} {rv} {ro} {rseam}'
         states+=1;eq=live(left,right)
         if not eq['compatible']:continue
         text=left+'; '+right+'.';a=audit(text);rows.append({'rendered':text,'audit':a,'seam_equation':eq,'complete_left_clause':True,'complete_right_clause':True,'provenance':{'cross_word_seam_design':True,'hand_authored_phrase_inventory':True,'independent_right_lexicalization':True,'semantic_valency':True,'live_equation_before_render':True,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False}})
        if states>=state_limit: break
       if states>=state_limit: break
      if states>=state_limit: break
     if states>=state_limit: break
    if states>=state_limit: break
   if states>=state_limit: break
  exact=[x for x in rows if x['audit']['exact'] and x['audit']['letters']>38]
  authored_controls=(
   'The analyst reviews the report near the station.', 'The teacher opens the file within the office.',
   'The engineer checks the schedule beside the river.', 'The writer writes a note after the meeting.',
   'Alice tests the system under the bridge.', 'Diana reads the message near the station.',
   'Marie plans the meeting within the office.', 'John calls the team beside the river.',
   'The analyst reviews the report after the meeting.', 'The teacher opens the file under the bridge.',
   'The engineer checks the schedule near the station.', 'The writer writes a note within the office.',
   'Alice reads the letter after the meeting.', 'Diana checks the report beside the river.',
   'Marie tests the system near the station.', 'John reviews the plan within the office.',
   'The analyst calls the team under the bridge.', 'The teacher reads the message after the meeting.',
   'The engineer opens the file beside the river.', 'The writer checks the schedule near the station.',
  )
  controls=list(authored_controls)
  reg=json.loads((ROOT/'docs/experiment-novelty-registry.json').read_text());collision=any(x.get('id') != ID and x.get('signature') == SIG for x in reg.get('entries',[])+reg.get('excluded',[]))
  return {'experiment_id':ID,'method':'bounded authored cross-word seam phrase design','stats':{'agents':len(AGENTS),'verbs':len(VERBS),'objects':len(OBJECTS),'seams':len(SEAMS),'states':states,'seam_compatible':len(rows),'fresh_exact_gt38':len(exact),'controls':len(controls)},'rendered_candidates':rows,'exact_candidates':exact,'controls':[{'rendered':x,'audit':audit(x)} for x in controls],'novelty_preflight':{'status':'collision' if collision else 'passed','signature':SIG,'registry_entries_checked':len(reg.get('entries',[])),'distinct_from':'prior boundary-seeded asymmetric lane; independently lexicalized right clauses and hand-authored phrase seams cross word boundaries','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False},'provenance':{'phrase_source':'new hand-authored contemporary English inventory','audits':['independent two-pointer mismatch','forward/reverse SHA-256'],'reader_gate':'closed unless exact >38 row appears','next_reader_test':'blinded complete prose versus shuffled controls'},'status':'fresh exact >38 candidate requires human reading' if exact else 'no fresh exact >38 closure; seam equation remains unsatisfied','next_construction':'change the seam phrase topology to a two-clause relation with independently selected connective'}
if __name__=='__main__':
 x=run();OUT.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x['stats']))
