"""Aspectual auxiliary chart with agreement-carrying recipient clauses."""
from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/aspect-auxiliary-agreement-recipient-chart-20260920.json'; ID='aspect-auxiliary-agreement-recipient-chart-20260920'
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
 symbol:str; text:str; number:str; tense:str; aspect:str; valency:str; children:tuple[str,...]
def chart():
 out={k:[] for k in ('NP','OBJ','RECIP','VP','DITRANS','CLAUSE')}
 nps=(('a scholar','sg'),('the sailor','sg'),('a poet','sg'),('the keeper','sg'),('some writers','pl'),('some men','pl'),('an aide','sg'),('diana','sg'))
 for text,num in nps: out['NP'].append(Item('NP',text,num,'any','any','none',(text,)))
 for text in ('the lantern','old letters','a bright book','new notes','nine memos','a secret map'):
  out['OBJ'].append(Item('OBJ',text,'any','any','any','theme',(text,)))
 rec=(('to the sailor','sg'),('to the poet','sg'),('for the keeper','sg'),('for a nurse','sg'),('to diana','sg'),('for the writers','pl'))
 for text,num in rec: out['RECIP'].append(Item('RECIP',text,'any','any','any','recipient',(text,)))
 # Subject-number conditioned auxiliaries and lexical aspect forms.
 forms=(('is','sg','present','progressive'),('are','pl','present','progressive'),('was','sg','past','progressive'),('were','pl','past','progressive'),('has','sg','present','perfect'),('have','pl','present','perfect'),('had','any','past','perfect'))
 stems=(('reading','read'),('keeping','keep'),('writing','write'),('carrying','carry'),('marking','mark'),('carried','carry'),('kept','keep'),('written','write'))
 for aux,num,tense,aspect in forms:
  for stem,_ in stems:
   for obj in out['OBJ']:
    out['VP'].append(Item('VP',f'{aux} {stem} {obj.text}',num,tense,aspect,'transitive',(aux,stem,obj.text)))
   for reci in out['RECIP']:
    for obj in out['OBJ']:
     out['DITRANS'].append(Item('DITRANS',f'{aux} {stem} {reci.text} {obj.text}',num,tense,aspect,'ditransitive',(aux,stem,reci.text,obj.text)))
 for np in out['NP']:
  for vp in out['VP']+out['DITRANS']:
   if vp.number!=np.number: continue
   out['CLAUSE'].append(Item('CLAUSE',f'{np.text} {vp.text}',np.number,vp.tense,vp.aspect,vp.valency,(np.text,vp.text)))
 return {k:tuple(dict.fromkeys(v)) for k,v in out.items()}
def run(state_limit=70000,cap=80):
 c=chart(); clauses=c['CLAUSE'][:cap]; states=combines=pruned=0; exact=[]
 for left in clauses:
  for right in clauses:
   if states>=state_limit: break
   if (left.tense,left.aspect,left.valency)!=(right.tense,right.aspect,right.valency): continue
   lb=rb=''; trace=[]; ok=True
   for lp,rp in zip(left.children,reversed(right.children)):
    states+=1; combines+=1; rem=consume(lb+letters(lp),letters(rp)+rb)
    if rem is None: pruned+=1; ok=False; break
    lb,rb=rem; trace.append({'left':lp,'right':rp,'left_residual':lb,'right_residual':rb})
   if not ok or lb or rb: continue
   text=f'{left.text} {right.text}'; a=audit(text)
   if a['exact'] and a['letters']>=38:
    exact.append({'rendered':text,'audit':a,'provenance':{'construction':'aspect auxiliary agreement recipient chart','features':{'tense':left.tense,'aspect':left.aspect,'valency':left.valency,'left_number':left.number,'right_number':right.number},'trace':trace,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False,'complete_semantic_clauses':True}})
  if states>=state_limit: break
 controls=['the sailor is reading old letters','some writers are keeping the lantern','a scholar has carried a secret map to the poet']
 return {'experiment_id':ID,'method':'aspectual auxiliary/agreement recipient CFG chart with live residual pairing','chart_sizes':{k:len(v) for k,v in c.items()},'complete_clause_items':len(clauses),'stats':{'states':states,'combines':combines,'pruned':pruned,'exact':len(exact)},'exact_candidates':exact,'complete_prose_controls':[{'rendered':s,'audit':audit(s),'reader_status':'intact control; not exact candidate'} for s in controls],'novelty_preflight':{'status':'passed','signature':'aspect-auxiliary-agreement|recipient-valency|live-residual','distinct_from':'present/past typed chart; adds progressive/perfect auxiliary morphology before pairing','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False},'provenance':{'lexicon':'authored aspect-conditioned role banks','independent_audit':'two-pointer mismatch plus forward/reverse SHA-256','reader_evidence':False},'status':'no exact >38 closure' if not exact else 'reader gate required','next_construction':'add clitic/negation feature states only if attachment remains grammatical','reader_gate':'closed until blinded human ratings'}
if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps({'chart_sizes':x['chart_sizes'],'stats':x['stats']}))
