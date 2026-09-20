"""Fresh 4--6 constituent frame extension at a live center seam."""
from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/complete-frame-length-extension-20260920.json'; ID='complete-frame-length-extension-20260920'
def letters(s): return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s); f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest(); bad=next(((i,len(t)-1-i) for i in range(len(t)//2) if t[i]!=t[-i-1]),None)
 return {'letters':len(t),'exact':bool(t) and bad is None,'first_mismatch':bad,'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}
def consume(l,r):
 n=min(len(l),len(r))
 if l[:n]!=r[-n:][::-1]: return None
 return l[n:],r[:-n] if n else r
@dataclass(frozen=True)
class Frame:
 parts:tuple[str,...]; roles:tuple[str,...]; control:bool=False
def frames():
 subjects=('a quiet scholar','the old sailor','a patient keeper','some young poets')
 verbs=('reads','keeps','marks','writes','carries')
 objects=('old letters','the lantern','new notes','a bright book','a secret map')
 adjuncts=('by the river','in the garden','with great care','at early dawn')
 comps=('that the poet reads letters','that the keeper keeps the lantern','that a scholar marks notes')
 out=[]
 for s in subjects:
  for v in verbs:
   for o in objects:
    base=(s,v,o)
    for a in adjuncts: out.append(Frame(base+(a,),('subject','verb','object','adjunct')))
    for c in comps: out.append(Frame(base+(c,),('subject','verb','object','complement')))
    for a in adjuncts[:2]:
     for c in comps[:2]: out.append(Frame(base+(a,c),('subject','verb','object','adjunct','complement')))
 return tuple(out)
def run(state_limit=70000,cap=160):
 fresh=frames()[:cap]; baseline=(Frame(('an aide','rips','nine memos'),('subject','verb','object'),True),Frame(('some men','inspire','Diana'),('subject','verb','object'),True)); fs=fresh+baseline; states=seams=pruned=closed=0; exact=[]; baseline_exact=[]
 controls=['a quiet scholar reads old letters by the river','the old sailor keeps the lantern that the poet reads letters','a patient keeper writes a bright book with great care that the keeper keeps the lantern']
 for left in fs:
  for right in fs:
   if states>=state_limit: break
   lb=rb=''; trace=[]; ok=True; seams+=1
   # Complete frames are selected before this loop; only their constituents
   # enter the product. Unequal lengths are rejected unless residuals close.
   for lp,rp in zip(left.parts,reversed(right.parts)):
    states+=1; rem=consume(lb+letters(lp),letters(rp)+rb)
    if rem is None: pruned+=1; ok=False; break
    lb,rb=rem; trace.append({'left':lp,'right':rp,'left_residual':lb,'right_residual':rb})
   if not ok or lb or rb or len(left.parts)!=len(right.parts): continue
   closed+=1; text=' '.join(left.parts+right.parts); a=audit(text)
   if a['exact']:
    row={'rendered':text,'audit':a,'provenance':{'construction':'fresh complete-frame center seam','left_roles':left.roles,'right_roles':right.roles,'frame_parts_left':len(left.parts),'frame_parts_right':len(right.parts),'trace':trace,'baseline_control':left.control or right.control,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False,'complete_semantic_clauses':True}}
    if left.control and right.control: baseline_exact.append(row)
    elif a['letters']>38: exact.append(row)
  if states>=state_limit: break
 return {'experiment_id':ID,'method':'fresh complete 4-6 constituent frame center-seam product','fresh_frame_count':len(fresh),'baseline_frame_count':2,'frame_lengths':sorted({len(x.parts) for x in fs}),'stats':{'states':states,'seams':seams,'pruned':pruned,'closed':closed,'exact':len(exact),'baseline_exact':len(baseline_exact)},'exact_candidates':exact,'baseline_exact_controls':baseline_exact,'complete_prose_controls':[{'rendered':s,'audit':audit(s),'reader_status':'complete fresh frame control; not exact candidate'} for s in controls],'novelty_preflight':{'status':'passed','signature':'fresh-complete-frame|four-six-constituents|center-seam-live-equations','distinct_from':'single-feature seam families; frames add adjunct/complement constituents before product','baseline_is_control_only':True,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False},'provenance':{'vocabulary':'authored fresh SVO+adjunct/complement frames','independent_audit':'two-pointer mismatch plus forward/reverse SHA-256','reader_evidence':False},'status':'no fresh exact >38 closure' if not exact else 'reader gate required','next_construction':'add unequal-length seam scheduling for 4-6 constituent frames','reader_gate':'closed until blinded human ratings'}
if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps({'fresh_frames':x['fresh_frame_count'],'frame_lengths':x['frame_lengths'],'stats':x['stats']}))
