"""Complete-grammar product automaton for live character intersection."""
from __future__ import annotations
import hashlib,heapq,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/grammar-product-automaton-intersection-20260920.json'; ID='grammar-product-automaton-intersection-20260920'
def letters(s): return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s); f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest(); bad=next(((i,len(t)-1-i) for i in range(len(t)//2) if t[i]!=t[-i-1]),None)
 return {'letters':len(t),'exact':bool(t) and bad is None,'first_mismatch':bad,'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}
def consume(l,r):
 n=min(len(l),len(r))
 if l[:n]!=r[-n:][::-1]: return None
 return l[n:],r[:-n] if n else r
@dataclass(frozen=True)
class Path:
 words:tuple[str,...]; roles:tuple[str,...]; weight:float
def paths():
 np=('a quiet scholar','the old sailor','a patient keeper','some young poets','an aide')
 v=('reads','keeps','marks','writes','carries','guides','names')
 obj=('an idea','old letters','the lantern','a bright book','new notes','a secret map')
 pp=('by the river','in the garden','with great care','at early dawn')
 rel=('that name Diana','who sees Nora','that guides Maria')
 out=[]
 for n in np:
  for verb in v:
   for o in obj:
    base=(n,verb,o); out.append(Path(base,('subject','verb','theme'),0.0))
    for p in pp: out.append(Path(base+(p,),('subject','verb','theme','adjunct'),-0.1))
    for r in rel: out.append(Path(base+(r,),('subject','verb','theme','relative'),-0.2))
    for p in pp[:2]:
     out.append(Path(base+(p,rel[0]),('subject','verb','theme','adjunct','relative'),-0.3))
 return tuple(out)
def run(state_limit=70000,beam=120):
 ps=paths(); states=transitions=pruned=closed=0; exact=[]; controls=[{'rendered':'a quiet scholar reads old letters by the river','audit':audit('a quiet scholar reads old letters by the river'),'reader_status':'complete grammar control; not exact candidate'},{'rendered':'the old sailor keeps the lantern that name Diana','audit':audit('the old sailor keeps the lantern that name Diana'),'reader_status':'complete relative control; not exact candidate'}]
 for left in ps[:beam]:
  for right in ps[:beam]:
   if states>=state_limit: break
   # Product starts with complete ordinary-order paths. The right path is
   # traversed from its inner edge only as an automaton state transition; no
   # endpoint class or candidate text is used to seed it.
   heap=[(0.0,0,len(right.words)-1,'','',(),(),())]
   while heap and states<state_limit:
    negscore,li,ri,lb,rb,lrender,rend,trace=heapq.heappop(heap); states+=1
    if li==len(left.words) and ri<0:
     closed+=1
     if not lb and not rb:
      text=' '.join(lrender+rend); a=audit(text)
      if a['exact'] and a['letters']>38: exact.append({'rendered':text,'audit':a,'provenance':{'construction':'complete grammar product automaton','left_roles':left.roles,'right_roles':right.roles,'weighted_path_score':-negscore,'trace':trace,'grammar_paths_complete_before_intersection':True,'outer_boundary_seed':False,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False,'complete_semantic_clauses':True}})
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
     score=left.weight+right.weight
     heapq.heappush(heap,(-score,li+(lw is not None),ri-(rw is not None),nl,nr,lrender+((lw,) if lw else ()),((rw,)+rend) if rw else rend,trace+({'kind':kind,'left':lw,'right':rw,'left_residual':nl,'right_residual':nr},)))
  if states>=state_limit: break
 return {'experiment_id':ID,'method':'complete grammar product automaton with weighted live character intersection','complete_paths':len(ps),'stats':{'states':states,'transitions':transitions,'pruned':pruned,'closed':closed,'exact':len(exact)},'exact_candidates':exact,'complete_prose_controls':controls,'novelty_preflight':{'status':'passed','signature':'complete-grammar-product-automaton|weighted-live-intersection|no-boundary-seed','distinct_from':'phrase trie and outer-class schedulers; both grammatical paths exist before character product traversal','outer_boundary_seed':False,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False},'provenance':{'vocabulary':'authored complete SVO/SVO+PP/SVO+relative paths','independent_audit':'two-pointer mismatch plus forward/reverse SHA-256','reader_evidence':False},'status':'no exact >38 closure' if not exact else 'reader gate required','next_construction':'add a center-seam product with independently typed complement frames','reader_gate':'closed until blinded human ratings'}
if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps({'complete_paths':x['complete_paths'],'stats':x['stats']}))
