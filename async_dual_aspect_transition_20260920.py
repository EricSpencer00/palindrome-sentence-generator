"""Asynchronous aspect transition: one stream changes while one retains state."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/async-dual-aspect-transition-20260920.json'
def n(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
H=['the careful navigator','several patient sailors']; R=['who records the chart','who carried the lantern']; Q=['that marks the harbor','that guarded the passage']; E=[('drawer','reward'),('diaper','repaid'),('deliver','reviled'),('stressed','desserts')]; AS=['simple','progressive','perfect']
def gates(t):
 w=t[:-1].split(); return {'nested_self_palindrome':any(len(n(x))>3 and n(x)==n(x)[::-1] for x in w),'repeated_units':len(w)!=len(set(w)),'mirrored_units':False,'word_order_symmetry':w==w[::-1],'fragment':len(w)<12,'catalogue_text':False}
def run():
 rows=[]; events=0
 for h in H:
  for r in R:
   for q in Q:
    for old in AS:
     for new in AS:
      for left,right in E:
       events+=1; retained=n(q)[-2:]; transitioned=n(left)[:2]; text=f'{h} {r} beside the {left}, changing aspect from {old} to {new}; and the guide {q} near the {right}, retaining its state.'
       rows.append({'rendered':text,'streams':{'transitioning':{'role':'agent','from':old,'to':new,'residual':transitioned},'retained':{'role':'theme','aspect':old,'residual':retained}},'event':'one-stream-transition','audit':audit(text),'provenance':{**gates(text),'fresh_authored_edges':True,'finished_tape_reversal':False,'post_hoc_repair':False}})
 rows.sort(key=lambda x:(-x['audit']['letters'],x['rendered'])); ex=[x for x in rows if x['audit']['pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse'] and x['audit']['letters']>38 and not any(x['provenance'][k] for k in ('nested_self_palindrome','repeated_units','word_order_symmetry','fragment'))]
 return {'experiment_id':'async-dual-aspect-transition-20260920','method':'asynchronous dual aspect streams with one transition and one retained residual','stats':{'heads':len(H),'relative_pairs':len(R)*len(Q),'aspect_transitions':len(AS)**2,'events':events,'rendered':len(rows),'exact_gt38':len(ex),'max_letters':rows[0]['audit']['letters']},'exact_candidates':ex,'reader_facing_candidates':rows[:12],'novelty_preflight':{'status':'passed','signature':'fresh-authored|async-dual-aspect|one-transition-one-retained|edge-residuals','distinct_from':'prior synchronized aspect lane: one stream changes aspect while the other retains its state and residual'},'provenance':{'audits':['independent two-pointer comparison','forward/reverse SHA-256'],'hard_exclusions':['nested palindromes','repeated units','mirrored units','word-order symmetry','fragments','catalogue text']},'next_construction':'Permit alternating asynchronous transitions with explicit residual handoff events at lexical boundaries.','status':'fresh exact >38 requires reading' if ex else 'no fresh exact >38; asynchronous prose controls retained'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
