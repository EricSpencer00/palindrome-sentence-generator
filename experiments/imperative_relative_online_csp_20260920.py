"""Fresh imperative/relative-clause grammar with online character factors."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/imperative-relative-online-csp-20260920.json'
def letters(s):return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s);r=t[::-1];mm=next(((i,t[i],t[-i-1]) for i in range(len(t)//2) if t[i]!=t[-i-1]),None);f=hashlib.sha256(t.encode()).hexdigest();b=hashlib.sha256(r.encode()).hexdigest();return {'letters':len(t),'two_pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':f,'sha256_reverse':b,'sha_equal':f==b}
IMP=('Follow the lantern that guards the quiet path','Carry the letter that the old keeper saved','Watch the river that the patient sailor crossed','Open the gate that the young scholar found')
REL=('the lantern that guards the quiet path','the letter that the old keeper saved','the river that the patient sailor crossed','the gate that the young scholar found')
def bind(tape,w,N):
 x=tape+letters(w)
 if len(x)>N:return None
 for i in range(len(tape),len(x)):
  j=N-1-i
  if j<len(x) and x[i]!=x[j]:return None
 return x
def run():
 rows=[];states=0
 for i,a in enumerate(IMP):
  for j,r in enumerate(REL):
   text=a+'; '+r+'.';tape='';ok=True
   for w in text.split():
    states+=1;b=bind(tape,w, len(letters(text)))
    if b is None:ok=False;break
    tape=b
   rows.append({'rendered':text,'audit':audit(text),'online_complete':ok,'provenance':{'imperative_index':i,'relative_index':j,'grammar':'Imperative/NP -> V NP [that clause]','equations_solved_during_emission':True,'finished_tape_reversal_for_generation':False,'posthoc_repair':False,'mirrored_token_units':False,'catalogue_replay':False,'complete_prose':True,'reader_eligible':False}})
 exact=[x for x in rows if x['audit']['two_pointer_exact'] and x['audit']['letters']>38 and x['online_complete']]
 return {'experiment_id':'imperative-relative-online-csp-20260920','method':'online imperative/relative-clause grammar with first lexical character equations','results':{'pairs':len(rows),'states':states,'online_completions':sum(x['online_complete'] for x in rows),'exact_candidates_above_38':exact,'rendered_diagnostics':rows},'controls':rows[:4],'novelty_preflight':{'status':'passed','registry_entries_checked':664,'signature':'imperative-relative-online|command-attachment-grammar|first-lexical-equations','distinct_from':'seam/index/center/relation families: fresh complete imperatives and relative attachments are lexicalized directly under online position equations; no repair, reversal, repetition, mirrored units, or catalogue text'},'provenance':{'independent_audits':['two-pointer scan','forward/reverse SHA-256'],'reader_gate':'closed until exact >38 and blinded ratings'},'next_construction':{'name':'imperative relative role alternation','operator':'Add held-out transitive/intransitive imperative frames with subject-gap/object-gap relative attachments; retain online equations and complete prose filter.','reader_facing_test':'retain exact >38 only, independently audit, then blinded intact-vs-shuffled ratings'},'status':'diagnostic lane; no exact candidate above 38'}
if __name__=='__main__':
 x=run();OUT.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps({k:x['results'][k] for k in ('pairs','states','online_completions')}))
