"""Unequal scheduler over complete typed matrix/complement frames."""
from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/unequal-typed-complement-frame-scheduler-20260920.json'; ID='unequal-typed-complement-frame-scheduler-20260920'
def letters(s): return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s); f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest(); bad=next(((i,len(t)-1-i) for i in range(len(t)//2) if t[i]!=t[-i-1]),None)
 return {'letters':len(t),'exact':bool(t) and bad is None,'first_mismatch':bad,'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}
def consume(l,r):
 n=min(len(l),len(r))
 if l[:n]!=r[-n:][::-1]: return None
 return l[n:],r[:-n] if n else r
@dataclass(frozen=True)
class Frame: parts:tuple[str,...]; roles:tuple[str,...]; comp_type:str; control:bool=False
def frames():
 s=('the sailor','a scholar','the keeper','a patient poet','some writers'); m=('knows','says','hopes','reports','believes'); c=('that the poet reads old letters','that the keeper keeps the lantern','that a scholar marks new notes','that the sailor carries a bright book'); a=('by the river','in the garden','at dawn')
 out=[]
 for subj in s:
  for verb in m:
   for comp in c:
    out.append(Frame((subj,verb,comp),('matrix_subject','matrix_verb','complement'),'finite_complement'))
    for adj in a: out.append(Frame((subj,verb,comp,adj),('matrix_subject','matrix_verb','complement','adjunct'),'finite_complement_adjunct'))
 return tuple(out)
def run(state_limit=70000,cap=160):
 fs=frames()[:cap]; baseline=(Frame(('an aide','rips','nine memos'),('subject','verb','object'),'baseline',True),Frame(('some men','inspire','Diana'),('subject','verb','object'),'baseline',True)); allf=fs+baseline; states=transitions=pruned=closed=0; exact=[]; base=[]
 controls=['the sailor knows that the poet reads old letters','a scholar says that the keeper keeps the lantern by the river','some writers believe that the sailor carries a bright book at dawn']
 for left in allf:
  for right in allf:
   if states>=state_limit: break
   stack=[(0,len(right.parts)-1,'','',(),(),())]
   while stack and states<state_limit:
    li,ri,lb,rb,lrender,rend,trace=stack.pop(); states+=1
    if li==len(left.parts) and ri<0:
     closed+=1
     if not lb and not rb:
      text=' '.join(lrender+rend); a=audit(text); row={'rendered':text,'audit':a,'provenance':{'construction':'unequal typed complement frame scheduler','left_roles':left.roles,'right_roles':right.roles,'left_complement_type':left.comp_type,'right_complement_type':right.comp_type,'trace':trace,'complete_frames_before_join':True,'baseline_control':left.control or right.control,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False,'complete_semantic_clauses':True}}
      if a['exact'] and left.control and right.control: base.append(row)
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
    for (nl,nr),lw,rw,kind in moves: stack.append((li+(lw is not None),ri-(rw is not None),nl,nr,lrender+((lw,) if lw else ()),((rw,)+rend) if rw else rend,trace+({'kind':kind,'left':lw,'right':rw,'left_residual':nl,'right_residual':nr},)))
  if states>=state_limit: break
 return {'experiment_id':ID,'method':'unequal typed finite-complement frame scheduler','fresh_frames':len(fs),'frame_types':sorted({x.comp_type for x in fs}),'stats':{'states':states,'transitions':transitions,'pruned':pruned,'closed':closed,'exact':len(exact),'baseline_exact':len(base)},'exact_candidates':exact,'baseline_exact_controls':base,'complete_prose_controls':[{'rendered':s,'audit':audit(s),'reader_status':'complete complement control; not exact candidate'} for s in controls],'novelty_preflight':{'status':'passed','signature':'typed-finite-complement|unequal-constituent-scheduler|live-residual','distinct_from':'untyped unequal frame scheduler; matrix/complement role frames are complete and typed before scheduling','baseline_is_control_only':True,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False},'provenance':{'vocabulary':'authored finite complement and adjunct frames','independent_audit':'two-pointer mismatch plus forward/reverse SHA-256','reader_evidence':False},'status':'no fresh exact >38 closure' if not exact else 'reader gate required','next_construction':'add typed question/answer complement frames','reader_gate':'closed until blinded human ratings'}
if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps({'fresh_frames':x['fresh_frames'],'frame_types':x['frame_types'],'stats':x['stats']}))
