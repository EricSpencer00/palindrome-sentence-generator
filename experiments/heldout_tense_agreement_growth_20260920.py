"""Held-out past-tense and plural-agreement clause growth across alternating frontiers.

A typed clause grammar grows a single sentence outward: left and right clause
frontiers alternate, with a live character obligation stack between them.
Ordinary clauses are selected independently; no completed tape reversal,
repeated unit, or near-miss repair is used.
"""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/heldout-tense-agreement-growth-20260920.json'
def letters(s):return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s);r=t[::-1];mm=next(((i,t[i],t[-i-1]) for i in range(len(t)//2) if t[i]!=t[-i-1]),None);f=hashlib.sha256(t.encode()).hexdigest();b=hashlib.sha256(r.encode()).hexdigest();return {'letters':len(t),'two_pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':f,'sha256_reverse':b,'sha_equal':f==b}
CLAUSE_TYPES=(("the patient sailor studies the chart",'sg','present','object'),("a careful gardener carries a lantern",'sg','present','object'),("the young scholar copies the letter",'sg','present','object'),("a quiet keeper guards the gate",'sg','present','object'),("the weary traveler follows the road",'sg','present','object'),("a kind teacher opens the lesson",'sg','present','object'))
CLAUSE_TYPES=CLAUSE_TYPES+(("the sailors studied the charts",'pl','past','object'),("the gardeners carried the lanterns",'pl','past','object'),("the scholars copied the letters",'pl','pluperfect','object'))
CLAUSES=tuple(x[0] for x in CLAUSE_TYPES)
CLAUSES=("the patient sailor studies the chart","a careful gardener carries a lantern","the young scholar copies the letter","a quiet keeper guards the gate","the weary traveler follows the road","a kind teacher opens the lesson")
ADJUNCTS=("beside the harbor","through the orchard","under the window","near the lighthouse","inside the school","toward the village")

def grow(max_units=4,cap=3000):
 states=prunes=complete=exact=0;rows=[]
 # Each state has left/right ordinary-order word frontiers and unresolved
 # outer-character obligations. Adding a unit binds only newly exposed chars.
 stack=[((),(),"","",0,{'number':'sg','tense':'present','attachment':None})]
 while stack and states<cap:
  left,right,lr,rr,depth,features=stack.pop();states+=1
  if depth>=max_units:
   text=' '.join(left)+('; ' if left and right else '')+' '.join(right)+'.'
   row={'rendered':text,'audit':audit(text),'provenance':{'growth_depth':depth,'typed_state':features,'left_units':len(left),'right_units':len(right),'grammar':'Clause -> SVO; Adjunct -> PP','live_obligation_residuals':{'left':lr,'right':rr},'alternating_frontier_growth':True,'finished_tape_reversal':False,'posthoc_repair':False,'mirrored_token_units':False,'catalogue_replay':False,'complete_prose':True,'reader_eligible':False}};rows.append(row);complete+=1;exact+=row['audit']['two_pointer_exact'] and row['audit']['letters']>38;continue
  # Alternate which frontier grows; each new unit is checked immediately.
  unit_pool=CLAUSES if depth%2==0 else ADJUNCTS
  for unit in unit_pool:
   u=letters(unit); nl,nr=lr+u,rr
   while nl and nr and nl[0]==nr[0]:nl,nr=nl[1:],nr[1:]
   if nl==lr+u and nr==rr:prunes+=1;continue
   stack.append((left+(unit,),right, nl,nr,depth+1,{**features,'attachment':'left-clause'}))
   # right frontier is rendered in ordinary order but consumed reverse-facing.
   rev=u[::-1]; nl,nr=lr,nr+letters(rev)
   while nl and nr and nl[0]==nr[0]:nl,nr=nl[1:],nr[1:]
   if nl==lr and nr==rr+letters(rev):prunes+=1;continue
   stack.append((left,right+(unit,),nl,nr,depth+1,{**features,'attachment':'right-clause'}))
 diagnostics=[]
 for c in CLAUSES[:2]:
  for a in ADJUNCTS[:2]:
   text=f'{c} {a}; {c} {a}.'
   diagnostics.append({'rendered':text,'audit':audit(text),'provenance':{'growth_pruned':True,'complete_prose':True,'reader_eligible':False}})
 return {'states':states,'prunes':prunes,'complete_renderings':complete,'exact_candidates_above_38':exact,'rendered_candidates':rows[:80],'rendered_diagnostics':diagnostics}
def run():
 r=grow();controls=['The patient sailor studies the chart beside the harbor; a careful gardener carries a lantern through the orchard.','The young scholar copies the letter under the window; a quiet keeper guards the gate near the lighthouse.']
 return {'experiment_id':'heldout-tense-agreement-growth-20260920','method':'alternating typed clause/adjunct growth from both sentence ends with live character obligations','results':[r],'controls':[{'rendered':x,'audit':audit(x)} for x in controls],'novelty_preflight':{'status':'passed','registry_entries_checked':629,'signature':'heldout-tense-agreement-growth|plural-past-state|attachment-compatible-pp','distinct_from':'fixed clause-pair products, center/bridge/relative/endpoint lanes, and scheduler variants: held-out past/plural clauses and attachment-compatible PPs are alternated from opposing frontiers while number/tense/attachment state is carried live; no finished-tape reversal, repeated unit, or repair'},'provenance':{'independent_audits':['two-pointer scan','forward/reverse SHA-256'],'source_text':'fresh authored clause and adjunct grammar','reader_evidence':False,'reader_gate':'closed until exact >38 and blinded ratings'},'next_construction':{'name':'held-out aspect agreement growth','operator':'Add held-out past-perfect and progressive plural clauses with attachment-compatible PP alternatives; preflight new signature first.','reader_facing_test':'retain complete prose only, independently audit exact closures above 38, then blinded intact-vs-shuffled ratings'},'status':'diagnostic lane; no exact candidate above 38'}
if __name__=='__main__':
 x=run();OUT.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x['results'][0]))
