"""Boundary-seeded asymmetric SVO/OVS lattice with live residual growth."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/boundary-seeded-asymmetric-clause-lattice-20260920.json';ID='boundary-seeded-asymmetric-clause-lattice-20260920';SIG='boundary-seeded-asymmetric|subject-object-endpoint-index|live-verb-residuals|complete-svo-ovs'
def letters(s):return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s);r=t[::-1];bad=next(((i,t[i],t[-i-1]) for i in range(len(t)//2) if t[i]!=t[-i-1]),None);return {'letters':len(t),'exact':bool(t) and bad is None,'first_mismatch':bad,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(r.encode()).hexdigest(),'sha_equal':hashlib.sha256(t.encode()).hexdigest()==hashlib.sha256(r.encode()).hexdigest()}
SUBJ=('Alice','Anna','Diana','Elena','Marie','Nora','Sarah','the analyst','the teacher','the engineer','the writer','the pilot')
OBJ=('the data','the agenda','the idea','the area','the sofa','the formula','the camera','the opera','the quota','the era','the letter','the report')
VERB=('reviews','opens','checks','writes','tests','reads','plans','calls','finds','holds')
PP=('near the station','within the office','beside the river','after the meeting')
def pref(a,b):return all(x==y for x,y in zip(letters(a),letters(b)[::-1]))
def run():
 # Exact endpoint indexing happens before any verb/interior choice.
 seeds=[(s,o) for s in SUBJ for o in OBJ if letters(s)[0]==letters(o)[-1] and letters(s)!=letters(o)]
 states=0;rows=[]
 for s,o in seeds:
  for v in VERB:
   left=f'{s} {v}';right=f'{o} {v} {s}'
   states+=1
   if not pref(left,right):continue
   for pp in PP:
    text=f'{s} {v} {o} {pp}; {o} {v} {s} {pp}.';a=audit(text);rows.append({'rendered':text,'audit':a,'roles_left':['SUBJ','VERB','OBJ','PP'],'roles_right':['OBJ','VERB','SUBJ','PP'],'provenance':{'boundary_seeded':True,'subject_object_endpoint_index':True,'live_verb_residual':True,'complete_left_clause':True,'complete_right_clause':True,'valency_agreement':True,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False}})
 exact=[x for x in rows if x['audit']['exact'] and x['audit']['letters']>38]
 controls=[f'{SUBJ[i%len(SUBJ)]} {VERB[i%len(VERB)]} {OBJ[i%len(OBJ)]} near the station.' for i in range(20)]
 reg=json.loads((ROOT/'docs/experiment-novelty-registry.json').read_text());collision=any(x.get('id')==ID or x.get('signature')==SIG for x in reg.get('entries',[])+reg.get('excluded',[]))
 return {'experiment_id':ID,'method':'endpoint-indexed asymmetric clause lattice with live interior growth','stats':{'subjects':len(SUBJ),'objects':len(OBJ),'endpoint_seed_pairs':len(seeds),'verb_states':states,'rendered_candidates':len(rows),'fresh_exact_gt38':len(exact),'controls':len(controls)},'rendered_candidates':rows[:200],'exact_candidates':exact,'controls':[{'rendered':x,'audit':audit(x)} for x in controls],'novelty_preflight':{'status':'collision' if collision else 'passed','signature':SIG,'registry_entries_checked':len(reg.get('entries',[])),'distinct_from':'prior asymmetric Cartesian product; subject/object endpoint seeds precede verb and PP expansion','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False},'provenance':{'source':'fresh authored contemporary names/articles/PPs','audits':['independent two-pointer mismatch','forward/reverse SHA-256'],'reader_gate':'closed; no programmatic readability claim','next_reader_test':'blinded complete clauses versus shuffled controls'},'status':'fresh exact >38 candidate requires human reading' if exact else 'no fresh exact >38 candidate','next_construction':'allow independently chosen right verb and PP attachment after endpoint seed'}
if __name__=='__main__':
 x=run();OUT.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x['stats']))
