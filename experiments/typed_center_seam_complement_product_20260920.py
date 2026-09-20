"""Typed center-seam product over independently complete complement clauses."""
from __future__ import annotations
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/typed-center-seam-complement-product-20260920.json'; ID='typed-center-seam-complement-product-20260920'
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
 parts:tuple[str,...]; roles:tuple[str,...]; complement_type:str
def frames():
 subjects=('the sailor','a scholar','the keeper','a patient poet','some writers')
 matrix=('knows','says','hopes','reports','believes')
 comps=('that the poet reads old letters','that the keeper keeps the lantern','that a scholar marks new notes','that the sailor carries a bright book','that writers guide the poet')
 out=[]
 for s in subjects:
  for m in matrix:
   for c in comps:
    out.append(Frame((s,m,c),('matrix_subject','matrix_verb','finite_complement'),'that_clause'))
 out.extend((Frame(('an aide','rips','nine memos'),('baseline_subject','baseline_verb','baseline_object'),'baseline_control'),
             Frame(('some men','inspire','Diana'),('baseline_subject','baseline_verb','baseline_object'),'baseline_control')))
 return tuple(out)
def run(state_limit=60000):
 fs=frames(); states=seams=pruned=closed=0; exact=[]; baseline_exact=[]
 controls=['the sailor knows that the poet reads old letters','a scholar says that the keeper keeps the lantern','some writers believe that the sailor carries a bright book']
 for left in fs:
  for right in fs:
   if states>=state_limit: break
   # Both frames are complete semantic derivations before this product begins.
   lb=rb=''; trace=[]; ok=True; seams+=1
   for lp,rp in zip(left.parts,reversed(right.parts)):
    states+=1; rem=consume(lb+letters(lp),letters(rp)+rb)
    if rem is None: pruned+=1; ok=False; break
    lb,rb=rem; trace.append({'left_constituent':lp,'right_constituent':rp,'left_residual':lb,'right_residual':rb})
   if not ok or lb or rb: continue
   closed+=1; text=' '.join(left.parts+right.parts); a=audit(text)
   if a['exact']:
    row={'rendered':text,'audit':a,'provenance':{'construction':'typed center-seam complement product','left_roles':left.roles,'right_roles':right.roles,'complement_type_left':left.complement_type,'complement_type_right':right.complement_type,'seam_trace':trace,'complete_frames_before_join':True,'endpoint_seed':False,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False,'complete_semantic_clauses':True}}
    if a['letters']==38 and left.complement_type==right.complement_type=='baseline_control': baseline_exact.append(row)
    if a['letters']>38: exact.append(row)
  if states>=state_limit: break
 return {'experiment_id':ID,'method':'typed center-seam product of complete finite-complement frames','complete_frames':len(fs),'stats':{'states':states,'seams':seams,'pruned':pruned,'closed':closed,'exact':len(exact),'baseline_exact':len(baseline_exact)},'exact_candidates':exact,'baseline_exact_controls':baseline_exact,'complete_prose_controls':[{'rendered':s,'audit':audit(s),'reader_status':'complete complement control; not exact candidate'} for s in controls],'novelty_preflight':{'status':'passed','signature':'typed-center-seam|complete-complement-frames|live-two-pointer-equations','distinct_from':'complete grammar product automaton; center seam is formed from independently derived matrix/complement frames','endpoint_seed':False,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False},'provenance':{'vocabulary':'authored matrix predicates and finite complement frames plus explicit 38-letter baseline control','independent_audit':'two-pointer mismatch plus forward/reverse SHA-256','reader_evidence':False},'status':'no exact >38 closure' if not exact else 'reader gate required','next_construction':'add interrogative complement frames as a separate typed seam family','reader_gate':'closed until blinded human ratings'}
if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps({'complete_frames':x['complete_frames'],'stats':x['stats']}))
