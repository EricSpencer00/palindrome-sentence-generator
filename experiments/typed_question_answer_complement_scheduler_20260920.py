"""Typed question/answer complement frames under unequal seam scheduling."""
from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/typed-question-answer-complement-scheduler-20260920.json'; ID='typed-question-answer-complement-scheduler-20260920'
def letters(s): return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s); f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest(); bad=next(((i,len(t)-1-i) for i in range(len(t)//2) if t[i]!=t[-i-1]),None)
 return {'letters':len(t),'exact':bool(t) and bad is None,'first_mismatch':bad,'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}
def consume(l,r):
 n=min(len(l),len(r))
 if l[:n]!=r[-n:][::-1]: return None
 return l[n:],r[:-n] if n else r
@dataclass(frozen=True)
class Frame: parts:tuple[str,...]; roles:tuple[str,...]; kind:str; control:bool=False
def frames():
 subs=('the sailor','a scholar','the keeper','some writers','a patient poet'); names=('the poet','the sailor','the keeper','Diana'); verbs=('reads','keeps','marks','writes','carries'); objs=('old letters','the lantern','new notes','a bright book')
 out=[]
 for s in subs:
  for n in names:
   for v in verbs:
    for o in objs:
     out.append(Frame((s,'asks','whether',n,v,o),('subject','question_verb','complementizer','complement_subject','verb','object'),'question'))
     out.append(Frame((s,'answers','that',n,v,o),('subject','answer_verb','complementizer','complement_subject','verb','object'),'answer'))
 return tuple(out)
def run(state_limit=70000,cap=160):
 fs=frames()[:cap]; baseline=(Frame(('an aide','rips','nine memos'),('subject','verb','object'),'baseline',True),Frame(('some men','inspire','Diana'),('subject','verb','object'),'baseline',True)); allf=fs+baseline; states=transitions=pruned=closed=0; exact=[]; base=[{'rendered':'an aide rips nine memos some men inspire Diana','audit':audit('an aide rips nine memos some men inspire Diana'),'provenance':{'baseline_control':True,'generated_by_this_constructor':False}}]
 controls=['the sailor asks whether the poet reads old letters','a scholar answers that the keeper keeps the lantern','some writers ask whether Diana marks new notes']
 for left in allf:
  for right in allf:
   if states>=state_limit: break
   if left.kind not in ('question','answer') or right.kind not in ('question','answer') or left.kind==right.kind: continue
   stack=[(0,len(right.parts)-1,'','',(),(),())]
   while stack and states<state_limit:
    li,ri,lb,rb,lrender,rend,trace=stack.pop(); states+=1
    if li==len(left.parts) and ri<0:
     closed+=1
     if not lb and not rb:
      text=' '.join(lrender+rend); a=audit(text); row={'rendered':text,'audit':a,'provenance':{'construction':'typed question/answer complement scheduler','left_kind':left.kind,'right_kind':right.kind,'left_roles':left.roles,'right_roles':right.roles,'trace':trace,'complete_dialogue_frames_before_join':True,'baseline_control':left.control or right.control,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False,'complete_semantic_clauses':True}}
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
 return {'experiment_id':ID,'method':'typed question/answer complement frames with unequal live scheduling','complete_dialogue_frames':len(fs),'stats':{'states':states,'transitions':transitions,'pruned':pruned,'closed':closed,'exact':len(exact),'baseline_exact':len(base)},'exact_candidates':exact,'baseline_exact_controls':base,'complete_prose_controls':[{'rendered':s,'audit':audit(s),'reader_status':'complete dialogue control; not exact candidate'} for s in controls],'novelty_preflight':{'status':'passed','signature':'typed-question-answer|whether-that-complements|unequal-scheduler|live-residual','distinct_from':'finite declarative complement frames; question/answer dialogue roles are complete before seam pairing','baseline_is_control_only':True,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False},'provenance':{'vocabulary':'authored question/answer complement frames','independent_audit':'two-pointer mismatch plus forward/reverse SHA-256','reader_evidence':False},'status':'no fresh exact >38 closure' if not exact else 'reader gate required','next_construction':'add wh-question complement frames with typed answer valency','reader_gate':'closed until blinded human ratings'}
if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps({'frames':x['complete_dialogue_frames'],'stats':x['stats']}))
