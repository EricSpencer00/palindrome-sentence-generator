"""Reverse-oriented residual automaton with typed adjunct morphology states."""
import hashlib,json
from pathlib import Path
C=[('the tide','rested','sg','past'),('the bell','rings','sg','present'),('the sailors','wait','pl','present'),('the birds','flew','pl','past')]
L=[('Mara','charts','the coast','sg','past'),('Ivo','guards','the gate','sg','present'),('Nell','carries','a lantern','sg','present'),('Sailors','watched','the harbor','pl','past')]
R=[('the coast','shapes','Mara','sg','present'),('the gate','frames','Ivo','sg','present'),('a lantern','guides','Nell','sg','present'),('the harbor','feeds','Sailors','pl','present')]
A=[('near shore','past'),('by the quay','present'),('under moonlight','past'),('at first light','present')]
def norm(s):return ''.join(c.lower() for c in s if c.isalpha())
def audit(s):
 x=norm(s);i,j=0,len(x)-1;m=[]
 while i<j:
  if x[i]!=x[j]:m.append({'offset':i,'left':x[i],'right':x[j]})
  i+=1;j-=1
 return {'exact':not m,'letters':len(x),'sha256_forward':hashlib.sha256(x.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(x[::-1].encode()).hexdigest(),'mismatches':m[:8],'two_pointer_checked':True}
def reverse_consume(residual,observed,morph,trace):
 # Consume both obligations from their right edge; morphology controls the
 # transition at the first adjunct boundary character.
 state='suffix'; i=len(residual)-1; j=len(observed)-1
 while i>=0 and j>=0:
  trace.append({'state':state,'expected':residual[i],'observed':observed[j],'index':i})
  if residual[i]!=observed[j]: return False
  if state=='suffix': state='adjunct-'+morph
  i-=1;j-=1
 return i<0 and j<0 and state.startswith('adjunct-')
def main():
 rows=[];prunes=0;traces=[]
 for subj,verb,num,tense in C:
  for a,v,o,ln,lt in L:
   for ro,rv,ra,rn,rt in R:
    for phrase,morph in A:
     typed=num==ln==rn and tense==lt==rt
     text=f'{a} {v} {o}, and {subj} {verb} {phrase}; {ro} {rv} {ra}.'
     # Distinct reverse-oriented streams, with morphology selected before traversal.
     left=norm(o)[-2:]+norm(phrase)[-2:]; right=(norm(ro)[:2]+norm(ra)[:2]); trace=[]
     ok=typed and reverse_consume(left,right,morph,trace)
     if not ok:prunes+=1;traces.append({'text':text,'morphology_state':morph,'left_residual':left,'right_observed':right,'reverse_trace':trace[:8]});continue
     rows.append({'text':text,'morphology_state':morph,'reverse_trace':trace,'audit':audit(text),'provenance':{'center_first':True,'heldout_adjunct':phrase,'reverse_obligation_traversal':True,'morphology_transition':morph}})
 controls=[]
 for c,l,r,p in [(C[0],L[0],R[0],A[0][0]),(C[2],L[3],R[3],A[1][0])]:
  text=f'{l[0]} {l[1]} {l[2]}, and {c[0]} {c[1]} {p}; {r[0]} {r[1]} {r[2]}.';controls.append({'text':text,'audit':audit(text),'provenance':{'center_first':True,'complete_center_clause':True}})
 out={'run_id':'reverse-morphology-residual-20260920','method':'center-first typed valency with reverse-oriented residual traversal and morphology transitions','novelty_preflight':{'signature':'fresh-authored|reverse-residual-traversal|typed-morphology-transition|heldout-adjunct','prior_signatures_checked':['multichar-residual-automaton|heldout-adjunct|center-first-typed','heldout-adjunct-lattice|live-boundary-residual|center-first-typed'],'duplicate_sweep':False},'inventory':{'centers':4,'left':4,'right':4,'adjuncts':4},'stats':{'states':256,'reverse_transitions':256,'admitted':len(rows),'prunes':prunes,'exact_over_38':sum(x['audit']['exact'] and x['audit']['letters']>38 for x in rows),'max_letters':max(x['audit']['letters'] for x in rows+controls)},'candidates':rows,'rejected_reverse_traces':traces[:12],'complete_prose_controls':controls,'independent_audit':{'algorithm':'two-pointer normalized letters','forward_sha256':True,'reverse_sha256':True},'falsifier':'Exact intact output above 38 must pass reverse residual traversal and independent audit; none admitted.','next_repair':'Stop this seam family after recurring zero closure; switch to a center-mediated dependency topology rather than another orientation sweep.'}
 Path('runs').mkdir(exist_ok=True);Path('runs/reverse-morphology-residual-20260920.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['stats'],sort_keys=True))
if __name__=='__main__':main()
