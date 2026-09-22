"""Diagnostic cross-word boundary grammar DP.

States are pairs of independently generated grammar chunks. A transition
records a boundary diagnostic, but does not enforce a global outside-in
residual; this lane is quarantined as a prose-control diagnostic.
"""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/cross-word-boundary-grammar-dp-20260920.json'
ID='cross-word-boundary-grammar-dp-20260920'; SIG='cross-word-boundary-grammar-dp|cross-word-boundary-transition|live-residual|fresh-frames'
def letters(s): return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
LEFT=(('the', 'steady', 'pilot', 'charts', 'a', 'hidden', 'island'),('a','patient','doctor','keeps','the','night','watch'),('our','young','writer','follows','a','winding','river'))
RIGHT=(('each','quiet','sailor','studies','the','open','map'),('every','careful','gardener','tends','a','winter','garden'),('the','old','teacher','answers','the','final','question'))
def boundary_step(lw,rw,residual):
 """Consume cross-word boundary classes; residual is an explicit state."""
 a,b=letters(lw),letters(rw)
 pair=min(len(a),len(b)); matched=sum(a[i]==b[-1-i] for i in range(pair))
 return residual + (pair-matched), matched
def run():
 rows=[]; states=0
 for left in LEFT:
  for right in RIGHT:
   residual=0; matched=0; trace=[]
   for lw,rw in zip(left,right):
    residual,got=boundary_step(lw,rw,residual); matched+=got; states+=1
    trace.append({'left_word':lw,'right_word':rw,'matched_boundary_chars':got,'residual':residual})
   l=' '.join(left); r=' '.join(right); rendered=f'{l}, while {r}.'; au=audit(rendered)
   rows.append({'rendered':rendered,'audit':au,'cross_word_trace':trace,'final_residual':residual,'matched_chars':matched,'complete_prose':True,'provenance':{'left':'fresh typed grammar frame','right':'fresh typed grammar frame','cross_word_transitions':True,'finished_tape_reversal':False,'post_hoc_repair':False,'copied_or_reversed_tape':False,'mirrored_token_units':False,'repeated_units':False,'fragment':False}})
 exact=[r for r in rows if r['audit']['exact'] and r['audit']['letters']>38]
 return {'experiment_id':ID,'method':'diagnostic cross-word boundary grammar DP; residual score is not a global palindrome obligation','stats':{'left_frames':len(LEFT),'right_frames':len(RIGHT),'transition_states':states,'rendered_candidates':len(rows),'fresh_exact_gt38':len(exact),'max_letters':max(r['audit']['letters'] for r in rows)},'rendered_candidates':rows,'exact_candidates':exact,'novelty_preflight':{'status':'quarantined-diagnostic','signature':SIG,'distinct_from':'phrase-pair product, endpoint class, and inner class gates','global_residual_enforced':False},'provenance':{'audits':['independent two-pointer mismatch','forward/reverse SHA-256'],'reader_gate':'closed; prose controls only'},'status':'diagnostic prose controls; not a palindrome constructor'}
if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps(x['stats']))
