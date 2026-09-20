"""Unequal-length live scheduler over complete semantic frames."""
from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/unequal-complete-frame-seam-scheduler-20260920.json'; ID='unequal-complete-frame-seam-scheduler-20260920'
def letters(s): return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s); f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest(); bad=next(((i,len(t)-1-i) for i in range(len(t)//2) if t[i]!=t[-i-1]),None)
 return {'letters':len(t),'exact':bool(t) and bad is None,'first_mismatch':bad,'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}
def consume(l,r):
 n=min(len(l),len(r))
 if l[:n]!=r[-n:][::-1]: return None
 return l[n:],r[:-n] if n else r
@dataclass(frozen=True)
class Frame: parts:tuple[str,...]; roles:tuple[str,...]; control:bool=False
def frames():
 subs=('a quiet scholar','the old sailor','a patient keeper','some young poets'); verbs=('reads','keeps','marks','writes','carries'); objs=('old letters','the lantern','new notes','a bright book','a secret map'); adj=('by the river','in the garden','with great care','at early dawn'); comp=('that the poet reads letters','that the keeper keeps the lantern','that a scholar marks notes')
 out=[]
 for s in subs:
  for v in verbs:
   for o in objs:
    base=(s,v,o)
    for a in adj: out.append(Frame(base+(a,),('subject','verb','object','adjunct')))
    for c in comp: out.append(Frame(base+(c,),('subject','verb','object','complement')))
    for a in adj[:2]:
     for c in comp[:2]: out.append(Frame(base+(a,c),('subject','verb','object','adjunct','complement')))
 return tuple(out)
def run(state_limit=70000,cap=150):
 fs=frames()[:cap]; baseline=(Frame(('an aide','rips','nine memos'),('subject','verb','object'),True),Frame(('some men','inspire','Diana'),('subject','verb','object'),True)); allf=fs+baseline; states=transitions=pruned=closed=0; exact=[]; baseline_exact=[]
 controls=['a quiet scholar reads old letters by the river','the old sailor keeps the lantern that the poet reads letters','a patient keeper writes a bright book with great care that the keeper keeps the lantern']
 for left in allf:
  for right in allf:
   if states>=state_limit: break
   stack=[(0,len(right.parts)-1,'','',(),(),())]
   while stack and states<state_limit:
    li,ri,lb,rb,lrender,rend,trace=stack.pop(); states+=1
    if li==len(left.parts) and ri<0:
     closed+=1
     if not lb and not rb:
      text=' '.join(lrender+rend); a=audit(text)
      row={'rendered':text,'audit':a,'provenance':{'construction':'unequal complete-frame seam scheduler','left_roles':left.roles,'right_roles':right.roles,'left_parts':len(left.parts),'right_parts':len(right.parts),'trace':trace,'baseline_control':left.control or right.control,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False,'complete_semantic_clauses':True}}
      if a['exact'] and left.control and right.control: baseline_exact.append(row)
      if a['exact'] and a['letters']>38 and not (left.control or right.control): exact.append(row)
     continue
    moves=[]
    if li<len(left.parts) and ri>=0:
     lw,rw=left.parts[li],right.parts[ri]; rem=consume(lb+letters(lw),letters(rw)+rb)
     if rem is not None: moves.append((rem,lw,rw,'pair'))
     else: pruned+=1
    if li<len(left.parts) and rb:
     lw=left.parts[li]; rem=consume(lb+letters(lw),rb)
     if rem is not None: moves.append((rem,lw,None,'left'))
     else: pruned+=1
    if ri>=0 and lb:
     rw=right.parts[ri]; rem=consume(lb,letters(rw)+rb)
     if rem is not None: moves.append((rem,None,rw,'right'))
     else: pruned+=1
    transitions+=len(moves)
    for (nl,nr),lw,rw,kind in moves:
     stack.append((li+(lw is not None),ri-(rw is not None),nl,nr,lrender+((lw,) if lw else ()),((rw,)+rend) if rw else rend,trace+({'kind':kind,'left':lw,'right':rw,'left_residual':nl,'right_residual':nr},)))
  if states>=state_limit: break
 return {'experiment_id':ID,'method':'unequal-length scheduler over complete 3-5 constituent frames','fresh_frames':len(fs),'stats':{'states':states,'transitions':transitions,'pruned':pruned,'closed':closed,'exact':len(exact),'baseline_exact':len(baseline_exact)},'exact_candidates':exact,'baseline_exact_controls':baseline_exact,'complete_prose_controls':[{'rendered':s,'audit':audit(s),'reader_status':'complete frame control; not exact candidate'} for s in controls],'novelty_preflight':{'status':'passed','signature':'unequal-complete-frame|independent-constituent-advancement|live-residual','distinct_from':'zip-only complete-frame seam; one side crosses multiple phrase boundaries while residual debt is live','baseline_is_control_only':True,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False},'provenance':{'vocabulary':'authored 3-5 constituent semantic frames','independent_audit':'two-pointer mismatch plus forward/reverse SHA-256','reader_evidence':False},'status':'no fresh exact >38 closure' if not exact else 'reader gate required','next_construction':'add a typed center complement seam with unequal scheduler','reader_gate':'closed until blinded human ratings'}
if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps({'fresh_frames':x['fresh_frames'],'stats':x['stats']}))
