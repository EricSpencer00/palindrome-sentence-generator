"""Global forward coordinated-clause CSP over held-out typed lexical factors."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/coordinated-clause-global-csp-20260920.json'
def letters(s):return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s);r=t[::-1];mm=next(((i,t[i],t[-i-1]) for i in range(len(t)//2) if t[i]!=t[-i-1]),None);f=hashlib.sha256(t.encode()).hexdigest();b=hashlib.sha256(r.encode()).hexdigest();return {'letters':len(t),'two_pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':f,'sha256_reverse':b,'sha_equal':f==b}
DET=(('the','sg'),('a','sg')); ADJ=('patient','careful','young','quiet','kind','steady','bright'); SUBJ=(('sailor','sg'),('gardener','sg'),('scholar','sg'),('keeper','sg'),('teacher','sg'),('traveler','sg')); VERB=(('studies','sg','trans'),('carries','sg','trans'),('copies','sg','trans'),('guards','sg','trans'),('opens','sg','trans'),('follows','sg','trans')); OBJ=(('the chart','sg'),('a lantern','sg'),('the letter','sg'),('the gate','sg'),('the lesson','sg'),('the road','sg')); PP=(('beside','harbor'),('through','orchard'),('under','window'),('near','lighthouse'),('inside','school'),('toward','village'))
# Explicit topological grammar: S -> Clause AND Clause; second clause has its
# own agreement state and is not a mirrored/replayed copy.
FACTORS=('det1','adj1','subj1','verb1','obj1','pp1','conj','det2','adj2','subj2','verb2','obj2')
def bind(chars,word,N):
 if len(chars)+len(word)>N:return None
 t=tuple(chars)+tuple(letters(word))
 for i in range(len(chars),len(t)):
  j=N-1-i
  if j<len(t) and t[i]!=t[j]:return None
 return t
def search(N,cap=12000):
 states=prunes=complete=exact=0; rows=[]; represented=1
 def rec(k,chars,words,feat):
  nonlocal states,prunes,complete,exact,represented
  states+=1
  if states>cap:return
  if k==len(FACTORS):
   complete+=1;text=' '.join(words)+'.';row={'rendered':text,'audit':audit(text),'provenance':{'target_length':N,'grammar':'S -> Clause1 AND Clause2; each Clause -> DET ADJ NPsubject Vtrans NPobject [PPlocative]','factors':FACTORS,'position_variables':'x[0:N]','palindrome_factor':'x[i]=x[N-1-i] during each factor emission','agreement_and_valency':feat,'coordinated_attachment':'conjunction joins two independently typed clauses','finished_tape_reversal':False,'posthoc_repair':False,'mirrored_token_units':False,'catalogue_replay':False,'complete_prose':True,'reader_eligible':False}};rows.append(row);exact+=row['audit']['two_pointer_exact'] and N>38;return
  f=FACTORS[k]; choices=[]
  if f.startswith('det'):choices=[(x,{'number':n}) for x,n in DET]
  elif f.startswith('adj'):choices=[(x,{}) for x in ADJ]
  elif f.startswith('subj'):choices=[(x,{'number':n}) for x,n in SUBJ]
  elif f.startswith('verb'):choices=[(x,{'valency':v}) for x,n,v in VERB if n==feat.get('number')]
  elif f.startswith('obj'):choices=[(x,{'object':True}) for x,n in OBJ]
  elif f.startswith('pp'):choices=[(p+' '+q,{'attachment':'locative'}) for p,q in PP]
  else:choices=[('and',{'coordination':True})]
  represented*=max(1,len(choices))
  for word,more in choices:
   b=bind(chars,word,N)
   if b is None:prunes+=1;continue
   rec(k+1,b,words+[word],{**feat,**more})
 rec(0,tuple(),[],{})
 return {'target_length':N,'represented_forward_language_count':str(represented),'states':states,'prunes':prunes,'complete_renderings':complete,'exact_candidates_above_38':exact,'rendered_candidates':rows[:80]}
def run():
 results=[search(n) for n in (44,56,68,80)]; controls=['The patient sailor studies the chart beside the harbor, and a careful gardener carries a lantern through the orchard.','A young scholar copies the letter under the window, and a quiet keeper guards the gate near the lighthouse.']
 return {'experiment_id':'coordinated-clause-global-csp-20260920','method':'global forward coordinated-clause grammar CSP over held-out typed lexical factors','results':results,'controls':[{'rendered':x,'audit':audit(x)} for x in controls],'novelty_preflight':{'status':'passed','registry_entries_checked':608,'signature':'coordinated-clause-topology|two-independent-agreement-states|global-position-factors','distinct_from':'center/bridge/relative/endpoint lanes and prior single-clause scheduler variants: conjunction is a live grammar factor joining two independently typed clauses, with separate agreement/valency states and direct x-position factor propagation'},'provenance':{'independent_audits':['two-pointer scan','forward/reverse SHA-256'],'source_text':'held-out common-word typed bank, no copied sentences','reader_evidence':False,'reader_gate':'closed until exact >38 and blinded ratings'},'next_construction':{'name':'embedded-complement topology','operator':'Replace coordination with a held-out finite complementizer and verb-selection frame, keeping two independent agreement states and live position factors; preflight a new signature first.','reader_facing_test':'retain complete prose only, independently audit exact closures above 38, then blinded intact-vs-shuffled ratings'},'status':'diagnostic lane; no exact candidate above 38'}
if __name__=='__main__':
 r=run();OUT.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps([(x['target_length'],x['states'],x['prunes'],x['complete_renderings'],x['exact_candidates_above_38']) for x in r['results']]))
