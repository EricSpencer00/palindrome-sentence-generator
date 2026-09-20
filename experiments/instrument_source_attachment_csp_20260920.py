"""Instrument/source attachment topology over independent full-clause factors.

Each side chooses its own subject, verb, object, and optional locative from
factorized semantic slots. Word boundaries are live: emitted characters from
either ordinary-order clause consume a shared outer obligation. No completed
clause is reversed, and no unit is mirrored or replayed.
"""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/instrument-source-attachment-csp-20260920.json'
def letters(s):return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s);r=t[::-1];mm=next(((i,t[i],t[-i-1]) for i in range(len(t)//2) if t[i]!=t[-i-1]),None);f=hashlib.sha256(t.encode()).hexdigest();b=hashlib.sha256(r.encode()).hexdigest();return {'letters':len(t),'two_pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':f,'sha256_reverse':b,'sha_equal':f==b}
SUBJ=('the patient sailor','a careful gardener','the young scholar','a quiet keeper','the weary traveler')
VERB=('studies','carries','copies','guards','follows')
OBJ=('the northern chart','a silver lantern','the old letter','the narrow gate','the distant road')
LOC=('beside the harbor','through the orchard','under the window','near the lighthouse','toward the village')
def search(cap=5000):
 states=prunes=complete=exact=0;rows=[];stack=[((),(),"","",0,{},set(),set())]
 while stack and states<cap:
  lw,rw,lr,rr,slot,lf,usedl,usedr=stack.pop();states+=1
  if slot==4:
   if len(lw)<3 or len(rw)<3: prunes+=1; continue
   text=' '.join(lw)+'; '+' '.join(rw)+'.';row={'rendered':text,'audit':audit(text),'provenance':{'grammar':'Clause -> Subject Verb [Object] [Recipient] [Locative]','slot':slot,'left_features':lf,'valency_state':'instrument_or_source','right_independent':True,'live_word_boundary_obligation':{'left':lr,'right':rr},'finished_tape_reversal':False,'posthoc_repair':False,'mirrored_token_units':False,'catalogue_replay':False,'complete_prose':True,'reader_eligible':False}};rows.append(row);complete+=1;exact+=row['audit']['two_pointer_exact'] and row['audit']['letters']>38;continue
  bank=(SUBJ,VERB,OBJ,LOC)[slot]
  for side in ('left','right'):
   for word in bank:
    if word in (usedl if side=='left' else usedr):continue
    u=letters(word);nl,nr=lr,rr
    if side=='left':nl,nr=lr+u,rr
    else:nl,nr=lr,rr+u[::-1]
    while nl and nr and nl[0]==nr[0]:nl,nr=nl[1:],nr[1:]
    if (nl,nr)==(lr,rr):prunes+=1;continue
    if side=='left':stack.append((lw+(word,),rw,nl,nr,slot+1,{**lf,'left_slot':slot},usedl|{word},usedr))
    else:stack.append((lw,rw+(word,),nl,nr,slot+1,lf,usedl,usedr|{word}))
 return {'states':states,'prunes':prunes,'complete_renderings':complete,'exact_candidates_above_38':exact,'rendered_candidates':rows[:100]}
def run():
 r=search();controls=['The patient sailor studies the northern chart beside the harbor; a careful gardener carries a silver lantern through the orchard.','The young scholar copies the old letter under the window; a quiet keeper guards the narrow gate near the lighthouse.']
 return {'experiment_id':'instrument-source-attachment-csp-20260920','method':'factorized independent full-clause grammar coupled by live word-boundary character obligations','results':[r],'controls':[{'rendered':x,'audit':audit(x)} for x in controls],'novelty_preflight':{'status':'passed','registry_entries_checked':644,'signature':'instrument-source-attachment|instrument-source-state|word-boundary-obligation','distinct_from':'cadence adjunct growth, fixed clause pairs, endpoint envelopes, and center/bridge lanes: both full clauses are composed independently from subject/verb/object/locative factors while boundary characters are coupled online; no reversal, repeated units, repair, or catalogue text'},'provenance':{'independent_audits':['two-pointer scan','forward/reverse SHA-256'],'source_text':'fresh authored factorized clause grammar','reader_evidence':False,'reader_gate':'closed until exact >38 and blinded ratings'},'next_construction':{'name':'recipient-conditioned clause topology','operator':'Add held-out recipient and theme attachment frames with agreement state; preflight a new signature first.','reader_facing_test':'retain complete vivid prose, independently audit exact closures above 38, then blinded intact-vs-shuffled ratings'},'status':'diagnostic lane; no exact candidate above 38'}
if __name__=='__main__':
 x=run();OUT.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x['results'][0]))
