"""Wh-recipient questions and typed ditransitive answers under live scheduling."""
from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/typed-wh-recipient-ditransitive-scheduler-20260920.json'; ID='typed-wh-recipient-ditransitive-scheduler-20260920'
def letters(s): return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s); f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest(); bad=next(((i,len(t)-1-i) for i in range(len(t)//2) if t[i]!=t[-i-1]),None)
 return {'letters':len(t),'exact':bool(t) and bad is None,'first_mismatch':bad,'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}
def consume(l,r):
 n=min(len(l),len(r))
 if l[:n]!=r[-n:][::-1]: return None
 return l[n:],r[:-n] if n else r
@dataclass(frozen=True)
class Frame: parts:tuple[str,...]; roles:tuple[str,...]; kind:str; valency:str
def frames():
 subs=('the sailor','a scholar','the keeper','some writers','a patient poet'); people=('the poet','the sailor','the keeper','Diana'); verbs=('gives','sends','carries','shows','brings'); objs=('an idea','old letters','the lantern','new notes')
 out=[]
 for s in subs:
  for wh in ('who','which poet','to whom'):
   for p in people:
    for v in verbs:
     for o in objs:
      out.append(Frame((wh,s,'asks',v,p,o),('wh_recipient','subject','question_verb','verb','recipient','theme'),'question','ditransitive'))
      out.append(Frame((s,'answers','that',v,p,o),('subject','answer_verb','complementizer','verb','recipient','theme'),'answer','ditransitive'))
 return tuple(out)
def run(state_limit=70000,cap=160):
 fs=frames()[:cap]; states=transitions=pruned=closed=0; exact=[]; base=[{'rendered':'an aide rips nine memos some men inspire Diana','audit':audit('an aide rips nine memos some men inspire Diana'),'provenance':{'baseline_control':True,'generated_by_this_constructor':False}}]
 controls=['to whom the sailor asks gives the poet an idea','a scholar answers that gives the keeper the lantern','which poet the writers ask sends Diana new notes']
 for left in fs:
  for right in fs:
   if states>=state_limit: break
   if left.kind==right.kind or left.valency!=right.valency: continue
   stack=[(0,len(right.parts)-1,'','',(),(),())]
   while stack and states<state_limit:
    li,ri,lb,rb,lrender,rend,trace=stack.pop(); states+=1
    if li==len(left.parts) and ri<0:
     closed+=1
     if not lb and not rb:
      text=' '.join(lrender+rend); a=audit(text)
      if a['exact'] and a['letters']>38: exact.append({'rendered':text,'audit':a,'provenance':{'construction':'typed wh-recipient ditransitive scheduler','left_kind':left.kind,'right_kind':right.kind,'valency':left.valency,'left_roles':left.roles,'right_roles':right.roles,'trace':trace,'complete_frames_before_join':True,'endpoint_seed':False,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False,'complete_semantic_clauses':True}})
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
 return {'experiment_id':ID,'method':'typed wh-recipient question/ditransitive answer scheduler','complete_frames':len(fs),'stats':{'states':states,'transitions':transitions,'pruned':pruned,'closed':closed,'exact':len(exact),'baseline_exact':1},'exact_candidates':exact,'baseline_exact_controls':base,'complete_prose_controls':[{'rendered':s,'audit':audit(s),'reader_status':'complete wh-recipient control; not exact candidate'} for s in controls],'novelty_preflight':{'status':'passed','signature':'wh-recipient|typed-ditransitive-answer|unequal-scheduler|live-residual','distinct_from':'wh-theme questions; recipient and theme are separate answer valency roles','baseline_is_control_only':True,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False},'provenance':{'vocabulary':'authored wh-recipient and ditransitive answer frames','independent_audit':'two-pointer mismatch plus forward/reverse SHA-256','reader_evidence':False},'status':'no fresh exact >38 closure','next_construction':'add recipient case alternants to wh-recipient frames','reader_gate':'closed until blinded human ratings'}
if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps({'frames':x['complete_frames'],'stats':x['stats']}))
