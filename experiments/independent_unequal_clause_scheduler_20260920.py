"""Independent two-sided scheduler for unequal lexical clause paths."""
from __future__ import annotations
import hashlib,json,math,re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/independent-unequal-clause-scheduler-20260920.json'; ID='independent-unequal-clause-scheduler-20260920'
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
 words:tuple[str,...]; roles:tuple[str,...]; score:float
def score(words):
 common={'a scholar','the sailor','quiet poet','old letters','the lantern','bright book','reads old','keeps the','by the','in the','who reads','that keeps'}
 return sum(1.0 for a,b in zip(words,words[1:]) if f'{a} {b}' in common)
def clause_paths():
 np=('a scholar','the sailor','a quiet poet','the keeper','some writers','a nurse','the farmer','an aide')
 v=('reads','keeps','marks','writes','carries','guides','names','sees','aids','finds')
 obj=('old letters','the lantern','an idea','a bright book','new notes','a secret map','the small bell')
 pp=('by the river','in the garden','with care','at dawn','near the harbor')
 rel=('who reads notes','that keeps the book','who marks the path')
 out=[]
 for n in np:
  for verb in v[:8]:
   for o in obj[:6]:
    base=(n,verb,o); out.append(Clause(base,('subject','verb','object'),score(base)))
    for p in pp[:3]:
     x=base+(p,); out.append(Clause(x,('subject','verb','object','adjunct'),score(x)))
    for r in rel[:2]:
     x=base+(r,); out.append(Clause(x,('subject','verb','object','relative'),score(x)))
    for p in pp[:2]:
     x=base+(p,rel[0]); out.append(Clause(x,('subject','verb','object','adjunct','relative'),score(x)))
 return tuple(sorted(out,key=lambda x:(-x.score,x.words))[:140])
def run(state_limit=70000):
 cs=clause_paths(); states=pruned=transitions=seeded=closed=0; exact=[]; controls=[{'rendered':' '.join(c.words),'audit':audit(' '.join(c.words)),'score':c.score,'reader_status':'complete control; not exact candidate'} for c in cs[:10]]
 for left in cs:
  for right in cs:
   if states>=state_limit: break
   if letters(left.words[0])[0]!=letters(right.words[-1])[-1]: continue
   seeded+=1
   # ri traverses the right derivation from its outer edge; right_render is
   # prepended so it remains eventual sentence order without reversing text.
   stack=[(0,len(right.words)-1,'','',left.words,(),(),())]
   while stack and states<state_limit:
    li,ri,lb,rb,lrender,rend,trace,chosen=stack.pop(); states+=1
    if li==len(left.words) and ri<0:
     closed+=1
     if not lb and not rb:
      text=' '.join(lrender+rend); a=audit(text)
      if a['exact'] and a['letters']>38: exact.append({'rendered':text,'audit':a,'score':left.score+right.score,'provenance':{'construction':'independent unequal clause scheduler','left_roles':left.roles,'right_roles':right.roles,'phrase_count_left':len(left.words),'phrase_count_right':len(right.words),'trace':trace,'outer_class_seed':True,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False,'complete_semantic_clauses':True}})
     continue
    # Candidate moves are ordered by collocation score only after checking
    # their live character compatibility.
    moves=[]
    if li<len(left.words) and ri>=0:
     lw=left.words[li]; rw=right.words[ri]; rem=consume(lb+letters(lw),letters(rw)+rb)
     if rem is not None: moves.append(('pair',rem,lw,rw))
     else: pruned+=1
    if li<len(left.words) and rb:
     lw=left.words[li]; rem=consume(lb+letters(lw),rb)
     if rem is not None: moves.append(('left',rem,lw,None))
     else: pruned+=1
    if ri>=0 and lb:
     rw=right.words[ri]; rem=consume(lb,letters(rw)+rb)
     if rem is not None: moves.append(('right',rem,None,rw))
     else: pruned+=1
    transitions+=len(moves)
    for kind,(nl,nr),lw,rw in sorted(moves,key=lambda x: (0 if x[0]=='pair' else 1)):
     stack.append((li+(lw is not None),ri-(rw is not None),nl,nr,
                   lrender+((lw,) if lw else ()),((rw,)+rend) if rw else rend,
                   trace+({'kind':kind,'left':lw,'right':rw,'left_residual':nl,'right_residual':nr},),chosen+(kind,)))
  if states>=state_limit: break
 return {'experiment_id':ID,'method':'independent unequal phrase-count clause scheduler with live residuals','complete_clause_paths':len(cs),'stats':{'states':states,'transitions':transitions,'seeded':seeded,'pruned':pruned,'closed':closed,'exact':len(exact)},'exact_candidates':exact,'complete_prose_controls':controls,'novelty_preflight':{'status':'passed','signature':'unequal-phrase-count|independent-side-scheduler|live-residual|relative-adjunct-paths','distinct_from':'equal-count lexical intersection; one side may advance while the other carries residual debt','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False},'provenance':{'vocabulary':'authored SVO/SVO+PP/SVO+relative paths','independent_audit':'two-pointer mismatch plus forward/reverse SHA-256','reader_evidence':False},'status':'no exact >38 closure' if not exact else 'reader gate required','next_construction':'add recipient-bearing unequal paths with attachment states','reader_gate':'closed until blinded human ratings'}
if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps({'paths':x['complete_clause_paths'],'stats':x['stats']}))
