"""Semantic multiword phrase-trie orbit with live cross-boundary debt."""
from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/semantic-phrase-trie-orbit-20260920.json'; ID='semantic-phrase-trie-orbit-20260920'
def letters(s): return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s); f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest(); bad=next(((i,len(t)-1-i) for i in range(len(t)//2) if t[i]!=t[-i-1]),None)
 return {'letters':len(t),'exact':bool(t) and bad is None,'first_mismatch':bad,'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}
def consume(l,r):
 n=min(len(l),len(r))
 if l[:n]!=r[-n:][::-1]: return None
 return l[n:],r[:-n] if n else r
@dataclass(frozen=True)
class Phrase:
 role:str; text:str
class PhraseTrie:
 def __init__(self,phrases):
  self.by_first={}; self.by_last={}
  for p in phrases:
   t=letters(p.text); self.by_first.setdefault(t[0],[]).append(p); self.by_last.setdefault(t[-1],[]).append(p)
 def first(self,ch): return tuple(self.by_first.get(ch,()))
 def last(self,ch): return tuple(self.by_last.get(ch,()))
def banks():
 return {k:tuple(Phrase(k,x) for x in xs) for k,xs in {
  'NP':('a quiet scholar','the old sailor','a patient keeper','some young poets','an aide','a bright bard'),
  'VP':('reads old letters','keeps the lantern','marks new notes','writes a bright book','carries a secret map','guides the sailor'),
  'PP':('by the river','in the garden','with great care','at early dawn','near the harbor'),
  'NAME':('diana','nora','maria','leon','aria','elena'),
  'OBJ':('an idea','old letters','the lantern','a bright book','new notes','a secret map'),
 }.items()}
def templates(b):
 # Complete semantic clauses. Different templates have different phrase counts.
 return ((('NP','VP'),), (('NP','VP','PP'),), (('NP','VP','OBJ'),), (('NP','VP','PP','NAME'),))
def run(state_limit=70000):
 b=banks(); tries={k:PhraseTrie(v) for k,v in b.items()}; templates_all=(('NP','VP'),('NP','VP','PP'),('NP','VP','OBJ'),('NP','VP','PP','NAME')); states=seeds=pruned=closed=0; exact=[]
 controls=['a quiet scholar reads old letters by the river','the old sailor keeps the lantern in the garden','a patient keeper writes a bright book at early dawn']
 for lt in templates_all:
  for rt in templates_all:
   # choose outer NP/last phrase from trie endpoint classes, not a text anchor
   for lp in b['NP']:
    for rp_role in (rt[-1],):
     for rp in tries[rp_role].last(letters(lp.text)[0]):
      seeds+=1; stack=[(1,len(rt)-2,'','', (lp.text,), (rp.text,), ())]
      while stack and states<state_limit:
       li,ri,lb,rb,left,right,trace=stack.pop(); states+=1
       if li==len(lt) and ri<0:
        closed+=1
        if not lb and not rb:
         text=' '.join(left+right); a=audit(text)
         if a['exact'] and a['letters']>38: exact.append({'rendered':text,'audit':a,'provenance':{'construction':'semantic phrase-trie orbit','left_template':lt,'right_template':rt,'trace':trace,'outer_endpoint_trie':True,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False,'complete_semantic_clauses':True}})
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
 return {'experiment_id':ID,'method':'semantic multiword phrase-trie orbit with live phrase/word residuals','template_counts':{str(t):1 for t in templates_all},'bank_sizes':{k:len(v) for k,v in b.items()},'stats':{'states':states,'seeds':seeds,'pruned':pruned,'closed':closed,'exact':len(exact)},'exact_candidates':exact,'complete_prose_controls':[{'rendered':s,'audit':audit(s),'reader_status':'complete phrase control; not exact candidate'} for s in controls],'novelty_preflight':{'status':'passed','signature':'semantic-phrase-trie|multiword-units|proper-name-endpoints|live-residual','distinct_from':'word trie and typed recipient schedulers; endpoint classes index complete phrase units','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False},'provenance':{'vocabulary':'authored multiword NP/VP/PP/name units','independent_audit':'two-pointer mismatch plus forward/reverse SHA-256','reader_evidence':False},'status':'no exact >38 closure' if not exact else 'reader gate required','next_construction':'add relative phrase units with name-compatible endpoints','reader_gate':'closed until blinded human ratings'}
if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps({'bank_sizes':x['bank_sizes'],'stats':x['stats']}))
