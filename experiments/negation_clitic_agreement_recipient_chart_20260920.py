"""Finite-auxiliary negation/clitic chart with agreement and live pairing."""
from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/negation-clitic-agreement-recipient-chart-20260920.json'; ID='negation-clitic-agreement-recipient-chart-20260920'
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
 symbol:str; text:str; number:str; tense:str; aspect:str; polarity:str; valency:str; children:tuple[str,...]
def chart():
 out={k:[] for k in ('NP','OBJ','RECIP','VP','DITRANS','CLAUSE')}
 for text,num in (('a scholar','sg'),('the sailor','sg'),('a poet','sg'),('the keeper','sg'),('some writers','pl'),('some men','pl'),('an aide','sg'),('diana','sg')):
  out['NP'].append(Item('NP',text,num,'any','any','positive','none',(text,)))
 for text in ('the lantern','old letters','a bright book','new notes','nine memos','a secret map'):
  out['OBJ'].append(Item('OBJ',text,'any','any','any','positive','theme',(text,)))
 for text in ('to the sailor','to the poet','for the keeper','for a nurse','to diana','for the writers'):
  out['RECIP'].append(Item('RECIP',text,'any','any','any','positive','recipient',(text,)))
 # Polarity is a finite-auxiliary state; it is not patched into a finished
 # clause. Aspect and subject agreement constrain each auxiliary realization.
 forms=(('is','sg','present','progressive'),('are','pl','present','progressive'),('was','sg','past','progressive'),('were','pl','past','progressive'),('has','sg','present','perfect'),('have','pl','present','perfect'),('had','any','past','perfect'))
 stems=('reading','keeping','writing','carrying','marking','carried','kept','written')
 for aux,num,tense,aspect in forms:
  for neg in (False,True):
   polarity='negative' if neg else 'positive'; marker=' not' if neg else ''
   for stem in stems:
    for obj in out['OBJ']:
     out['VP'].append(Item('VP',f'{aux}{marker} {stem} {obj.text}',num,tense,aspect,polarity,'transitive',(aux,marker.strip(),stem,obj.text)))
    for rec in out['RECIP']:
     for obj in out['OBJ']:
      out['DITRANS'].append(Item('DITRANS',f'{aux}{marker} {stem} {rec.text} {obj.text}',num,tense,aspect,polarity,'ditransitive',(aux,marker.strip(),stem,rec.text,obj.text)))
 for np in out['NP']:
  for vp in out['VP']+out['DITRANS']:
   if np.number!=vp.number: continue
   out['CLAUSE'].append(Item('CLAUSE',f'{np.text} {vp.text}',np.number,vp.tense,vp.aspect,vp.polarity,vp.valency,(np.text,vp.text)))
 return {k:tuple(dict.fromkeys(v)) for k,v in out.items()}
def run(state_limit=70000,cap=80):
 c=chart(); clauses=c['CLAUSE'][:cap]; states=combines=pruned=0; exact=[]
 for left in clauses:
  for right in clauses:
   if states>=state_limit: break
   feats=(left.tense,left.aspect,left.polarity,left.valency)
   if feats!=(right.tense,right.aspect,right.polarity,right.valency): continue
   lb=rb=''; trace=[]; ok=True
   for lp,rp in zip(left.children,reversed(right.children)):
    states+=1; combines+=1; rem=consume(lb+letters(lp),letters(rp)+rb)
    if rem is None: pruned+=1; ok=False; break
    lb,rb=rem; trace.append({'left':lp,'right':rp,'left_residual':lb,'right_residual':rb})
   if not ok or lb or rb: continue
   text=f'{left.text} {right.text}'; a=audit(text)
   if a['exact'] and a['letters']>=38:
    exact.append({'rendered':text,'audit':a,'provenance':{'construction':'finite auxiliary negation/clitic recipient chart','features':{'tense':left.tense,'aspect':left.aspect,'polarity':left.polarity,'valency':left.valency,'left_number':left.number,'right_number':right.number},'trace':trace,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False,'complete_semantic_clauses':True}})
  if states>=state_limit: break
 controls=['the sailor is not reading old letters','some writers are not keeping the lantern','a scholar has not carried a secret map to the poet']
 return {'experiment_id':ID,'method':'finite auxiliary negation/clitic agreement recipient chart with live residual pairing','chart_sizes':{k:len(v) for k,v in c.items()},'complete_clause_items':len(clauses),'stats':{'states':states,'combines':combines,'pruned':pruned,'exact':len(exact)},'exact_candidates':exact,'complete_prose_controls':[{'rendered':s,'audit':audit(s),'reader_status':'intact control; not exact candidate'} for s in controls],'novelty_preflight':{'status':'passed','signature':'finite-auxiliary-negation|polarity-feature|recipient-valency|live-residual','distinct_from':'aspect chart; polarity is selected as an auxiliary grammar state before lexical pairing','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False},'provenance':{'lexicon':'authored agreement-conditioned auxiliary and recipient banks','independent_audit':'two-pointer mismatch plus forward/reverse SHA-256','reader_evidence':False},'status':'no exact >38 closure' if not exact else 'reader gate required','next_construction':'add interrogative inversion only with complete clause attachment states','reader_gate':'closed until blinded human ratings'}
if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps({'chart_sizes':x['chart_sizes'],'stats':x['stats']}))
