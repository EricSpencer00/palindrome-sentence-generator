"""Asymmetric semantic-role clause pairing across reversed character spans."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/asymmetric-role-semordnilap-clause-20260920.json';ID='asymmetric-role-semordnilap-clause-20260920';SIG='asymmetric-role-pairing|left-subject-right-object|live-crossword-residuals|complete-contemporary-svo'
def letters(s):return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s);r=t[::-1];bad=next(((i,t[i],t[-i-1]) for i in range(len(t)//2) if t[i]!=t[-i-1]),None);return {'letters':len(t),'exact':bool(t) and bad is None,'first_mismatch':bad,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(r.encode()).hexdigest(),'sha_equal':hashlib.sha256(t.encode()).hexdigest()==hashlib.sha256(r.encode()).hexdigest()}
SUBJ=('Alice','Diana','Marie','John','the analyst','the teacher','the engineer','the writer','the pilot','the designer')
OBJ=('the report','the file','the schedule','a note','the system','the message','the meeting','the plan','the book','the letter')
VERB=('reviews','opens','checks','writes','tests','reads','plans','calls','finds','holds')
PP=('near the station','within the office','beside the river','after the meeting','under the bridge')
def compatible(left,right):
 a=letters(left);b=letters(right)[::-1];return all(x==y for x,y in zip(a,b))
def run():
 states=0;rows=[]
 for s in SUBJ:
  for v in VERB:
   for o in OBJ:
    left=f'{s} {v} {o}'
    for rs in SUBJ:
     for rv in VERB:
      for ro in OBJ:
       # Asymmetric role mapping: left SUBJ is compared against right OBJ,
       # and left OBJ against right SUBJ; right is rendered in OVS order.
       right=f'{ro} {rv} {rs}'
       states+=1
       if not compatible(left,right):continue
       text=left+'; '+right+'.';a=audit(text);rows.append({'rendered':text,'left_roles':['SUBJ','VERB','OBJ'],'right_roles':['OBJ','VERB','SUBJ'],'audit':a,'provenance':{'asymmetric_role_alignment':True,'left_subject_right_object':True,'left_object_right_subject':True,'complete_left_clause':True,'complete_right_clause':True,'live_cross_word_residuals':True,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False}})
       if len(rows)>=200:break
      if len(rows)>=200:break
     if len(rows)>=200:break
    if len(rows)>=200:break
   if len(rows)>=200:break
  if len(rows)>=200:break
 controls=[]
 for i in range(20):
  controls.append(f'{SUBJ[i%len(SUBJ)]} {VERB[i%len(VERB)]} {OBJ[i%len(OBJ)]}; {OBJ[(i+2)%len(OBJ)]} {VERB[(i+3)%len(VERB)]} {SUBJ[(i+4)%len(SUBJ)]}.')
 registry=json.loads((ROOT/'docs/experiment-novelty-registry.json').read_text());collision=any(x.get('id')==ID or x.get('signature')==SIG for x in registry.get('entries',[])+registry.get('excluded',[]))
 exact=[x for x in rows if x['audit']['exact'] and x['audit']['letters']>38]
 return {'experiment_id':ID,'method':'asymmetric semantic-role pairing for complete contemporary clauses','stats':{'subjects':len(SUBJ),'objects':len(OBJ),'verbs':len(VERB),'states':states,'span_compatible_rendered':len(rows),'fresh_exact_gt38':len(exact),'controls':len(controls)},'rendered_candidates':rows,'exact_candidates':exact,'controls':[{'rendered':x,'audit':audit(x)} for x in controls],'novelty_preflight':{'status':'collision' if collision else 'passed','signature':SIG,'registry_entries_checked':len(registry.get('entries',[])),'distinct_from':'prior anaphoric discourse and symmetric envelope lanes; reversed role alignment is asymmetric SVO versus OVS','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False},'provenance':{'source':'fresh authored names/articles/contemporary clause bank','audits':['independent two-pointer mismatch','forward/reverse SHA-256'],'reader_gate':'closed; no programmatic readability claim','next_reader_test':'blinded intact asymmetric clauses versus shuffled controls'},'status':'fresh exact >38 candidate requires human reading' if exact else 'no fresh exact >38 candidate','next_construction':'add SVO+PP asymmetric role paths with explicit attachment binding'}
if __name__=='__main__':
 x=run();OUT.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x['stats']))
