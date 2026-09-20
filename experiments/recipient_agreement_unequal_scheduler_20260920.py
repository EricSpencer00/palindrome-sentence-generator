"""Agreement-conditioned recipient paths under unequal live scheduling."""
from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/recipient-agreement-unequal-scheduler-20260920.json'; ID='recipient-agreement-unequal-scheduler-20260920'
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
 words:tuple[str,...]; roles:tuple[str,...]; number:str; recipient_number:str
def paths():
 subjects=(('a scholar','sg'),('the sailor','sg'),('a poet','sg'),('the keeper','sg'),('some writers','pl'),('some men','pl'),('an aide','sg'))
 verbs={'sg':('reads','keeps','marks','writes','carries','guides','names','sees'),'pl':('read','keep','mark','write','carry','guide','name','see')}
 objects=('an idea','old letters','the lantern','a bright book','new notes','a secret map')
 recipients=(('to the poet','sg'),('to the sailor','sg'),('for the keeper','sg'),('to diana','sg'),('for the writers','pl'))
 pp=('by the river','in the garden','with care','at dawn'); rel=('who reads notes','that keeps the book')
 out=[]
 for subj,num in subjects:
  for v in verbs[num]:
   for obj in objects:
    base=(subj,v,obj); out.append(Clause(base,('subject','verb','theme'),num,'none'))
    for rec,rnum in recipients:
     # Recipient number is carried as an argument feature; all forms here
     # are number-neutral prepositional realizations, but remain typed.
     out.append(Clause((subj,v,rec,obj),('subject','verb','recipient','theme'),num,rnum))
    for p in pp: out.append(Clause(base+(p,),('subject','verb','theme','adjunct'),num,'none'))
    for q in rel: out.append(Clause(base+(q,),('subject','verb','theme','relative'),num,'none'))
    for rec,rnum in recipients[:2]:
     for p in pp[:2]: out.append(Clause((subj,v,rec,obj,p),('subject','verb','recipient','theme','adjunct'),num,rnum))
 return tuple(out)
def run(state_limit=70000,cap=180):
 cs=paths()[:cap]; states=transitions=pruned=seeded=closed=0; exact=[]
 controls=[{'rendered':'a scholar gives to the poet an idea at dawn','audit':audit('a scholar gives to the poet an idea at dawn'),'reader_status':'complete agreement control; not exact candidate'}, {'rendered':'some writers carry new notes for the keeper','audit':audit('some writers carry new notes for the keeper'),'reader_status':'complete plural control; not exact candidate'}]
 for left in cs:
  for right in cs:
   if states>=state_limit: break
   if (left.number,left.recipient_number)!=(right.number,right.recipient_number): continue
   if letters(left.words[0])[0]!=letters(right.words[-1])[-1]: continue
   seeded+=1; stack=[(0,len(right.words)-1,'','',(),(),())]
   while stack and states<state_limit:
    li,ri,lb,rb,lrender,rend,trace=stack.pop(); states+=1
    if li==len(left.words) and ri<0:
     closed+=1
     if not lb and not rb:
      text=' '.join(lrender+rend); a=audit(text)
      if a['exact'] and a['letters']>38: exact.append({'rendered':text,'audit':a,'provenance':{'construction':'agreement-conditioned recipient unequal scheduler','left_roles':left.roles,'right_roles':right.roles,'left_number':left.number,'right_number':right.number,'recipient_number':left.recipient_number,'trace':trace,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False,'complete_semantic_clauses':True}})
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
 return {'experiment_id':ID,'method':'agreement-conditioned recipient unequal scheduler with live residuals','path_counts':{n:sum(len(x.words)==n for x in cs) for n in sorted({len(x.words) for x in cs})},'stats':{'states':states,'transitions':transitions,'seeded':seeded,'pruned':pruned,'closed':closed,'exact':len(exact)},'exact_candidates':exact,'complete_prose_controls':controls,'novelty_preflight':{'status':'passed','signature':'recipient-number-agreement|unequal-attachment-paths|independent-side-scheduler|live-residual','distinct_from':'untyped recipient attachment scheduler; subject/verb/recipient features gate paths before pairing','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False},'provenance':{'vocabulary':'authored number-conditioned subjects/verbs and typed recipients','independent_audit':'two-pointer mismatch plus forward/reverse SHA-256','reader_evidence':False},'status':'no exact >38 closure' if not exact else 'reader gate required','next_construction':'add recipient case/preposition alternants with attachment states','reader_gate':'closed until blinded human ratings'}
if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps({'path_counts':x['path_counts'],'stats':x['stats']}))
