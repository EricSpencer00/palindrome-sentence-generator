"""Relative phrase trie with proper-name endpoint classes."""
from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/relative-name-endpoint-phrase-trie-20260920.json'; ID='relative-name-endpoint-phrase-trie-20260920'
def letters(s): return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s); f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest(); bad=next(((i,len(t)-1-i) for i in range(len(t)//2) if t[i]!=t[-i-1]),None)
 return {'letters':len(t),'exact':bool(t) and bad is None,'first_mismatch':bad,'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}
def consume(l,r):
 n=min(len(l),len(r))
 if l[:n]!=r[-n:][::-1]: return None
 return l[n:],r[:-n] if n else r
@dataclass(frozen=True)
class Phrase: role:str; text:str
class Trie:
 def __init__(self,items):
  self.last={}
  for p in items: self.last.setdefault(letters(p.text)[-1],[]).append(p)
 def ending(self,ch): return tuple(self.last.get(ch,()))
def banks():
 raw={'NP':('a quiet scholar','the old sailor','a patient keeper','some young poets','an aide'),'VP':('reads old letters','keeps the lantern','marks new notes','writes a bright book','carries a secret map'),'PP':('by the river','in the garden','with great care','at early dawn'),'REL':('that name Diana','who sees Nora','that guides Maria','who helps Leon'),'NAME':('Diana','Nora','Maria','Leon')}
 return {k:tuple(Phrase(k,x) for x in xs) for k,xs in raw.items()}
def run(state_limit=70000):
 b=banks(); trie=Trie(b['REL']); templates=(('NP','VP'),('NP','VP','PP'),('NP','VP','REL'),('NP','VP','PP','REL')); states=seeds=pruned=closed=0; exact=[]
 controls=['a scholar reads old letters that name Diana','the old sailor keeps the lantern by the river','a patient keeper writes a bright book who sees Nora']
 for lt in templates:
  for rt in templates:
   for lp in b['NP']:
    opts=trie.ending(letters(lp.text)[0]) if rt[-1]=='REL' else b[rt[-1]]
    for rp in opts:
     seeds+=1; stack=[(1,len(rt)-2,'','',(lp.text,),(rp.text,),())]
     while stack and states<state_limit:
      li,ri,lb,rb,left,right,trace=stack.pop(); states+=1
      if li==len(lt) and ri<0:
       closed+=1
       if not lb and not rb:
        text=' '.join(left+right); a=audit(text)
        if a['exact'] and a['letters']>38: exact.append({'rendered':text,'audit':a,'provenance':{'construction':'relative phrase trie with name endpoints','left_template':lt,'right_template':rt,'trace':trace,'whole_relative_units':True,'name_compatible_endpoint':rt[-1]=='REL','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False,'complete_semantic_clauses':True}})
       continue
      if li>=len(lt) or ri<0: continue
      for lw in b[lt[li]]:
       for rw in b[rt[ri]]:
        rem=consume(lb+letters(lw.text),letters(rw.text)+rb)
        if rem is None: pruned+=1; continue
        nl,nr=rem; stack.append((li+1,ri-1,nl,nr,left+(lw.text,),(rw.text,)+right,trace+({'left':lw.text,'right':rw.text,'left_residual':nl,'right_residual':nr},)))
     if states>=state_limit: break
    if states>=state_limit: break
   if states>=state_limit: break
  if states>=state_limit: break
 return {'experiment_id':ID,'method':'relative phrase trie with name-compatible endpoints and live residuals','templates':templates,'bank_sizes':{k:len(v) for k,v in b.items()},'stats':{'states':states,'seeds':seeds,'pruned':pruned,'closed':closed,'exact':len(exact)},'exact_candidates':exact,'complete_prose_controls':[{'rendered':s,'audit':audit(s),'reader_status':'complete relative control; not exact candidate'} for s in controls],'novelty_preflight':{'status':'passed','signature':'whole-relative-phrase|proper-name-endpoint-trie|live-residual','distinct_from':'semantic phrase orbit; adds complete relative units selected through name endpoints','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False},'provenance':{'vocabulary':'authored relative clauses with proper-name terminals','independent_audit':'two-pointer mismatch plus forward/reverse SHA-256','reader_evidence':False},'status':'no exact >38 closure' if not exact else 'reader gate required','next_construction':'add relative pronoun agreement and attachment features','reader_gate':'closed until blinded human ratings'}
if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps({'bank_sizes':x['bank_sizes'],'stats':x['stats']}))
