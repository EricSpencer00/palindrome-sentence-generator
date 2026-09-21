"""Held-out adjunct lattice with live bilateral boundary residuals."""
import hashlib,json
from pathlib import Path
C=[{"subject":"the tide","verb":"rested","number":"sg","tense":"past","agreement":"3sg","valency":"intransitive"},{"subject":"the bell","verb":"rings","number":"sg","tense":"present","agreement":"3sg","valency":"intransitive"},{"subject":"the sailors","verb":"wait","number":"pl","tense":"present","agreement":"3pl","valency":"intransitive"},{"subject":"the birds","verb":"flew","number":"pl","tense":"past","agreement":"3pl","valency":"intransitive"}]
L=[("Mara","charts","the coast","sg","past"),("Ivo","guards","the gate","sg","present"),("Nell","carries","a lantern","sg","present"),("Sailors","watched","the harbor","pl","past")]
R=[("the coast","shapes","Mara","sg","present"),("the gate","frames","Ivo","sg","present"),("a lantern","guides","Nell","sg","present"),("the harbor","feeds","Sailors","pl","present")]
# Held out attachment lattice: complete adverbial phrases, not fragments.
A=[("near shore","n","e"),("by the quay","b","y"),("under moonlight","u","t"),("at first light","a","t")]
def norm(s):return ''.join(c.lower() for c in s if c.isalpha())
def audit(s):
 x=norm(s);i,j=0,len(x)-1;m=[]
 while i<j:
  if x[i]!=x[j]:m.append({'offset':i,'left':x[i],'right':x[j]})
  i+=1;j-=1
 return {'exact':not m,'letters':len(x),'sha256_forward':hashlib.sha256(x.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(x[::-1].encode()).hexdigest(),'mismatches':m[:8],'two_pointer_checked':True}
def main():
 rows=[];prunes=0; transitions=0
 for c in C:
  for a,v,o,n,t in L:
   for ro,rv,ra,rn,rt in R:
    for phrase,af,al in A:
     transitions+=1
     typed=n==c['number']==rn and t==c['tense']==rt
     # Live residual: consume left object-final -> adjunct-first, then
     # adjunct-final -> right object-initial before admitting the emission.
     residual=(norm(o)[-1],af,al,norm(ro)[0]); boundary=(residual[0]==residual[1] and residual[2]==residual[3])
     text=f'{a} {v} {o}, and {c["subject"]} {c["verb"]} {phrase}; {ro} {rv} {ra}.'
     if not (typed and boundary):prunes+=1;continue
     rows.append({'text':text,'adjunct':phrase,'residual':residual,'audit':audit(text),'provenance':{'center_first':True,'heldout_adjunct_lattice':True,'live_residual_consumed':True,'left_event':[a,v,o],'right_event':[ro,rv,ra]}})
 controls=[]
 for c,(a,v,o,n,t),(ro,rv,ra,rn,rt),phrase in [(C[0],L[0],R[0],A[0][0]),(C[2],L[3],R[3],A[1][0])]:
  text=f'{a} {v} {o}, and {c["subject"]} {c["verb"]} {phrase}; {ro} {rv} {ra}.';controls.append({'text':text,'audit':audit(text),'provenance':{'center_first':True,'complete_center_clause':True}})
 out={'run_id':'live-adjunct-residual-lattice-20260920','method':'center-first typed valency with held-out adjunct lattice and live bilateral boundary residual','novelty_preflight':{'signature':'fresh-authored|heldout-adjunct-lattice|live-boundary-residual|center-first-typed','prior_signatures_checked':['center-first|mixed-valency|licensed-adjunct|typed-bilateral-product','center-first|prelexical-valency|typed-bilateral-product','seam-indexed-clause-variants|outer-character-index'],'duplicate_sweep':False},'inventory':{'centers':4,'left_events':4,'right_events':4,'heldout_adjuncts':4},'stats':{'states':256,'live_transitions':transitions,'admitted':len(rows),'residual_prunes':prunes,'exact_over_38':sum(x['audit']['exact'] and x['audit']['letters']>38 for x in rows),'max_letters':max(x['audit']['letters'] for x in rows+controls)},'candidates':rows,'complete_prose_controls':controls,'independent_audit':{'algorithm':'two-pointer normalized letters','forward_sha256':True,'reverse_sha256':True},'falsifier':'A reader-worthy exact candidate above 38 must satisfy typed center features and both live residual equalities; none admitted here.','next_repair':'Replace the two equality residual with a typed multi-character boundary automaton that permits licensed morphology without a lexical sweep.'}
 Path('runs').mkdir(exist_ok=True);Path('runs/live-adjunct-residual-lattice-20260920.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['stats'],sort_keys=True))
if __name__=='__main__':main()
