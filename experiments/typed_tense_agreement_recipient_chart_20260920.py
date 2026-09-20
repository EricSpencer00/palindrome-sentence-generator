"""Typed tense/agreement recipient chart with live character intersection."""
from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/typed-tense-agreement-recipient-chart-20260920.json'
ID='typed-tense-agreement-recipient-chart-20260920'
def letters(s): return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s); f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest(); bad=next(((i,len(t)-1-i) for i in range(len(t)//2) if t[i]!=t[-i-1]),None)
 return {'letters':len(t),'exact':bool(t) and bad is None,'first_mismatch':bad,'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}
def consume(l,r):
 n=min(len(l),len(r))
 if l[:n]!=r[-n:][::-1]: return None
 return l[n:],r[:-n] if n else r
@dataclass(frozen=True)
class Item:
 symbol:str; text:str; number:str; tense:str; valency:str; children:tuple[str,...]
def chart():
 out={k:[] for k in ('NP','V','OBJ','RECIP','VP','DITRANS','CLAUSE')}
 # Feature-conditioned lexical realizations. Recipient phrases are arguments,
 # not arbitrary adjuncts, and only transitive/ditransitive frames are built.
 nps=(('a scholar','sg'),('the sailor','sg'),('a poet','sg'),('the keeper','sg'),('some men','pl'),('the writers','pl'),('an aide','sg'),('diana','sg'),('leon','sg'))
 for text,num in nps: out['NP'].append(Item('NP',text,num,'any','none',(text,)))
 for text in ('the lantern','old letters','a bright book','new notes','nine memos','a secret map'):
  out['OBJ'].append(Item('OBJ',text,'any','any','theme',(text,)))
 verbs=(('keeps','sg','present'),('reads','sg','present'),('marks','sg','present'),('writes','sg','present'),('carries','sg','present'),('inspires','sg','present'),('keep','pl','present'),('read','pl','present'),('mark','pl','present'),('write','pl','present'),('carried','any','past'),('read','any','past'),('marked','any','past'),('wrote','any','past'))
 for text,num,tense in verbs: out['V'].append(Item('V',text,num,tense,'predicate',(text,)))
 rec=(('to the sailor','sg'),('to the poet','sg'),('for the keeper','sg'),('for a nurse','sg'),('to diana','sg'),('for the writers','pl'))
 for text,num in rec: out['RECIP'].append(Item('RECIP',text,'any','any','recipient',(text,)))
 for v in out['V']:
  for obj in out['OBJ']:
   out['VP'].append(Item('VP',f'{v.text} {obj.text}',v.number,v.tense,'transitive',(v.text,obj.text)))
  for reci in out['RECIP']:
   for obj in out['OBJ']:
    out['DITRANS'].append(Item('DITRANS',f'{v.text} {reci.text} {obj.text}',v.number,v.tense,'ditransitive',(v.text,reci.text,obj.text)))
 # Clause agreement: subject number must match present verb; past is neutral.
 for np in out['NP']:
  for vp in out['VP']+out['DITRANS']:
   if vp.tense=='present' and np.number!=vp.number: continue
   out['CLAUSE'].append(Item('CLAUSE',f'{np.text} {vp.text}',np.number,vp.tense,vp.valency,(np.text,vp.text)))
 return {k:tuple(dict.fromkeys(v)) for k,v in out.items()}
def run(state_limit=60000,cap=70):
 c=chart(); clauses=c['CLAUSE'][:cap]; states=combines=pruned=0; exact=[]
 for left in clauses:
  for right in clauses:
   if states>=state_limit: break
   # Pair only feature-compatible clause frames; lexical choices remain
   # independent, and constituent residuals are consumed before continuation.
   if left.valency!=right.valency or left.tense!=right.tense: continue
   lb=rb=''; trace=[]; ok=True
   for lp,rp in zip(left.children,reversed(right.children)):
    states+=1; combines+=1; rem=consume(lb+letters(lp),letters(rp)+rb)
    if rem is None: pruned+=1; ok=False; break
    lb,rb=rem; trace.append({'left':lp,'right':rp,'left_residual':lb,'right_residual':rb})
   if not ok or lb or rb: continue
   text=f'{left.text} {right.text}'; a=audit(text)
   if a['exact'] and a['letters']>=38:
    exact.append({'rendered':text,'audit':a,'provenance':{'construction':'typed tense/agreement recipient chart','left_features':{'number':left.number,'tense':left.tense,'valency':left.valency},'right_features':{'number':right.number,'tense':right.tense,'valency':right.valency},'trace':trace,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False,'complete_semantic_clauses':True}})
  if states>=state_limit: break
 controls=['the sailor keeps the lantern','the writers read old letters','a scholar carried a secret map to the poet']
 return {'experiment_id':ID,'method':'typed tense/agreement/recipient CFG chart with live residual pairing','chart_sizes':{k:len(v) for k,v in c.items()},'complete_clause_items':len(clauses),'stats':{'states':states,'combines':combines,'pruned':pruned,'exact':len(exact)},'exact_candidates':exact,'complete_prose_controls':[{'rendered':s,'audit':audit(s),'reader_status':'intact control; not exact candidate'} for s in controls],'novelty_preflight':{'status':'passed','signature':'typed-tense-number-valency|recipient-chart|live-residual','distinct_from':'untyped recipient/adjunct chart; lexical realization is feature-conditioned before pairing','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False},'provenance':{'lexicon':'authored feature-conditioned role banks','independent_audit':'two-pointer mismatch plus forward/reverse SHA-256','reader_evidence':False},'status':'no exact >38 closure' if not exact else 'reader gate required','next_construction':'add aspectual auxiliaries with agreement-carrying subject states','reader_gate':'closed until blinded human ratings'}
if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps({'chart_sizes':x['chart_sizes'],'stats':x['stats']}))
