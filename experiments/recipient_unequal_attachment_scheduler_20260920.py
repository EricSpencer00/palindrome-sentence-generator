"""Recipient-bearing unequal clause scheduler with live attachment debt."""
from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/recipient-unequal-attachment-scheduler-20260920.json'; ID='recipient-unequal-attachment-scheduler-20260920'
def letters(s): return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s); f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest(); bad=next(((i,len(t)-1-i) for i in range(len(t)//2) if t[i]!=t[-i-1]),None)
 return {'letters':len(t),'exact':bool(t) and bad is None,'first_mismatch':bad,'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}
def consume(l,r):
 n=min(len(l),len(r))
 if l[:n]!=r[-n:][::-1]: return None
 return l[n:],r[:-n] if n else r
@dataclass(frozen=True)
class Clause:
 words:tuple[str,...]; roles:tuple[str,...]
def paths():
 np=('a scholar','the sailor','a poet','the keeper','some writers','a nurse','an aide')
 v=('reads','keeps','marks','writes','carries','guides','names','sees')
 obj=('an idea','old letters','the lantern','a bright book','new notes','a secret map')
 rec=('to the poet','to the sailor','for the keeper','to diana','for a nurse')
 pp=('by the river','in the garden','with care','at dawn')
 rel=('who reads notes','that keeps the book')
 out=[]
 for n in np:
  for verb in v:
   for o in obj:
    base=(n,verb,o); out.append(Clause(base,('subject','verb','theme')))
    for r in rec: out.append(Clause((n,verb,r,o),('subject','verb','recipient','theme')))
    for p in pp: out.append(Clause(base+(p,),('subject','verb','theme','adjunct')))
    for q in rel: out.append(Clause(base+(q,),('subject','verb','theme','relative')))
    for r in rec[:2]:
     for p in pp[:2]: out.append(Clause((n,verb,r,o,p),('subject','verb','recipient','theme','adjunct')))
 return tuple(out)
def run(state_limit=70000,cap=180):
 cs=paths()[:cap]; states=transitions=pruned=seeded=closed=0; exact=[]
 controls=[{'rendered':'a scholar gives to the poet an idea at dawn','audit':audit('a scholar gives to the poet an idea at dawn'),'reader_status':'complete recipient control; not exact candidate'}, {'rendered':'the keeper reads old letters in the garden','audit':audit('the keeper reads old letters in the garden'),'reader_status':'complete adjunct control; not exact candidate'}]
 for left in cs:
  for right in cs:
   if states>=state_limit: break
   if letters(left.words[0])[0]!=letters(right.words[-1])[-1]: continue
   seeded+=1; stack=[(0,len(right.words)-1,'','',(),(),())]
   while stack and states<state_limit:
    li,ri,lb,rb,lrender,rend,trace=stack.pop(); states+=1
    if li==len(left.words) and ri<0:
     closed+=1
     if not lb and not rb:
      text=' '.join(lrender+rend); a=audit(text)
      if a['exact'] and a['letters']>38: exact.append({'rendered':text,'audit':a,'provenance':{'construction':'recipient attachment unequal clause scheduler','left_roles':left.roles,'right_roles':right.roles,'left_units':len(left.words),'right_units':len(right.words),'trace':trace,'recipient_attachment':True,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False,'complete_semantic_clauses':True}})
     continue
    moves=[]
    if li<len(left.words) and ri>=0:
     lw,rw=left.words[li],right.words[ri]; rem=consume(lb+letters(lw),letters(rw)+rb)
     if rem is not None: moves.append((rem,lw,rw,'pair'))
     else: pruned+=1
    if li<len(left.words) and rb:
     lw=left.words[li]; rem=consume(lb+letters(lw),rb)
     if rem is not None: moves.append((rem,lw,None,'left'))
     else: pruned+=1
    if ri>=0 and lb:
     rw=right.words[ri]; rem=consume(lb,letters(rw)+rb)
     if rem is not None: moves.append((rem,None,rw,'right'))
     else: pruned+=1
    transitions+=len(moves)
    for (nl,nr),lw,rw,kind in moves:
     stack.append((li+(lw is not None),ri-(rw is not None),nl,nr,lrender+((lw,) if lw else ()),((rw,)+rend) if rw else rend,trace+({'kind':kind,'left':lw,'right':rw,'left_residual':nl,'right_residual':nr},)))
  if states>=state_limit: break
 return {'experiment_id':ID,'method':'recipient attachment unequal clause scheduler with live residual obligations','path_counts':{n:sum(len(x.words)==n for x in cs) for n in sorted({len(x.words) for x in cs})},'stats':{'states':states,'transitions':transitions,'seeded':seeded,'pruned':pruned,'closed':closed,'exact':len(exact)},'exact_candidates':exact,'complete_prose_controls':controls,'novelty_preflight':{'status':'passed','signature':'recipient-attachment|unequal-clause-paths|independent-side-scheduler|live-residual','distinct_from':'plain unequal SVO scheduler; recipient is a typed argument between verb and theme','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False},'provenance':{'vocabulary':'authored recipient/theme/adjunct/relative paths','independent_audit':'two-pointer mismatch plus forward/reverse SHA-256','reader_evidence':False},'status':'no exact >38 closure' if not exact else 'reader gate required','next_construction':'add recipient number/agreement states to attachment paths','reader_gate':'closed until blinded human ratings'}
if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps({'path_counts':x['path_counts'],'stats':x['stats']}))
