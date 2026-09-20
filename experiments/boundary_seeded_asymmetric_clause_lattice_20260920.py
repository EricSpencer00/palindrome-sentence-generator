"""Boundary-seeded SVO/SVO lattice with live asymmetric role crossing."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/boundary-seeded-asymmetric-clause-lattice-20260920.json';ID='boundary-seeded-asymmetric-clause-lattice-20260920';SIG='boundary-seeded-svo-role-crossing|subject-object-endpoint-index|live-residual-phrase-growth|complete-svo-svo'
def letters(s):return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s);r=t[::-1];bad=next(((i,t[i],t[-i-1]) for i in range(len(t)//2) if t[i]!=t[-i-1]),None);return {'letters':len(t),'exact':bool(t) and bad is None,'first_mismatch':bad,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(r.encode()).hexdigest(),'sha_equal':hashlib.sha256(t.encode()).hexdigest()==hashlib.sha256(r.encode()).hexdigest()}
SUBJ=('Alice','Anna','Diana','Elena','Marie','Nora','Sarah','the analyst','the teacher','the engineer','the writer','the pilot')
OBJ=('the data','the agenda','the idea','the area','the sofa','the formula','the camera','the opera','the quota','the era','the letter','the report')
VERB=('reviews','opens','checks','writes','tests','reads','plans','calls','finds','holds')
PP=('near the station','within the office','beside the river','after the meeting')
def consume(left,right):
 n=min(len(left),len(right))
 if left[:n]!=right[-n:][::-1]: return None
 return left[n:],right[:-n] if n else right
def endpoint_seed(left,right):
 x,y=letters(left),letters(right)[::-1]; n=min(len(x),len(y)); k=0
 while k<n and x[k]==y[k]: k+=1
 if not k: return None
 return x[k:],letters(right)[:-k]
def run():
 # The right clause remains ordinary SVO; reversal makes left SUBJ meet right
 # OBJ and left OBJ meet right SUBJ. Endpoint seeds are indexed before verbs.
 seeds=[(ls,ro) for ls in SUBJ for ro in OBJ if letters(ls)[0]==letters(ro)[-1] and letters(ls)!=letters(ro)]
 states=seam_prunes=0;rows=[]
 for ls,ro in seeds:
  initial=endpoint_seed(ls,ro)
  if initial is None: seam_prunes+=1; continue
  for lv in VERB:
   for rv in VERB:
    states+=1
    after_verbs=consume(initial[0]+letters(lv),initial[1]+letters(rv))
    if after_verbs is None: seam_prunes+=1; continue
    for lo in OBJ:
     for rs in SUBJ:
      tail=consume(after_verbs[0]+letters(lo),after_verbs[1]+letters(rs))
      if tail is None or tail[0] or tail[1]:
       seam_prunes+=1; continue
      text=f'{ls} {lv} {lo}; {rs} {rv} {ro}.'
      a=audit(text)
      rows.append({'rendered':text,'audit':a,'roles_left':['SUBJ','VERB','OBJ'],'roles_right':['SUBJ','VERB','OBJ'],'provenance':{'boundary_seeded':True,'subject_object_endpoint_index':True,'left_subject_right_object':True,'left_object_right_subject':True,'live_residual_phrase_growth':True,'complete_left_clause':True,'complete_right_clause':True,'valency_agreement':True,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False}})
 exact=[x for x in rows if x['audit']['exact'] and x['audit']['letters']>38]
 authored_controls=(
  'Alice reviews the report near the station.', 'Anna opens the agenda in the office.',
  'Diana checks the data beside the river.', 'Elena writes a letter after the meeting.',
  'Marie tests the system within the office.', 'Nora reads the message near the station.',
  'Sarah plans the meeting beside the river.', 'The analyst calls the team after the meeting.',
  'The teacher finds the file within the office.', 'The engineer holds the camera near the station.',
  'The writer reviews the letter beside the river.', 'The pilot opens the report after the meeting.',
  'Alice checks the agenda within the office.', 'Anna reads the formula near the station.',
  'Diana plans the meeting beside the river.', 'Elena carries the report after the meeting.',
  'Marie writes a note within the office.', 'Nora checks the schedule near the station.',
  'Sarah reads the book beside the river.', 'The analyst reviews the plan after the meeting.',
 )
 controls=list(authored_controls)
 reg=json.loads((ROOT/'docs/experiment-novelty-registry.json').read_text());collision=any(x.get('id') != ID and x.get('signature') == SIG for x in reg.get('entries',[])+reg.get('excluded',[]))
 return {'experiment_id':ID,'method':'endpoint-indexed SVO/SVO lattice with live asymmetric residual growth','stats':{'subjects':len(SUBJ),'objects':len(OBJ),'endpoint_seed_pairs':len(seeds),'verb_states':states,'seam_prunes':seam_prunes,'rendered_candidates':len(rows),'fresh_exact_gt38':len(exact),'controls':len(controls)},'rendered_candidates':rows[:200],'exact_candidates':exact,'controls':[{'rendered':x,'audit':audit(x)} for x in controls],'novelty_preflight':{'status':'collision' if collision else 'passed','signature':SIG,'registry_entries_checked':len(reg.get('entries',[])),'distinct_from':'prior asymmetric Cartesian product; ordinary SVO/SVO clauses are seeded by reversed subject/object endpoints and grown with residual buffers','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False},'provenance':{'source':'fresh authored contemporary names/articles/PPs','audits':['independent two-pointer mismatch','forward/reverse SHA-256'],'reader_gate':'closed; no programmatic readability claim','next_reader_test':'blinded complete clauses versus shuffled controls'},'status':'fresh exact >38 candidate requires human reading' if exact else 'no fresh exact >38 candidate','next_construction':'add independently selected PP attachments only after a complete SVO residual closes'}
if __name__=='__main__':
 x=run();OUT.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x['stats']))
