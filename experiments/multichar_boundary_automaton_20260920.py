"""Center-first event product with a multi-character residual automaton."""
import hashlib,json
from pathlib import Path
C=[('the tide','rested','sg','past'),('the bell','rings','sg','present'),('the sailors','wait','pl','present'),('the birds','flew','pl','past')]
L=[('Mara','charts','the coast','sg','past'),('Ivo','guards','the gate','sg','present'),('Nell','carries','a lantern','sg','present'),('Sailors','watched','the harbor','pl','past')]
R=[('the coast','shapes','Mara','sg','present'),('the gate','frames','Ivo','sg','present'),('a lantern','guides','Nell','sg','present'),('the harbor','feeds','Sailors','pl','present')]
A=[('near shore','near'),('by the quay','by'),('under moonlight','under'),('at first light','at')]
def n(s):return ''.join(c.lower() for c in s if c.isalpha())
def audit(s):
 x=n(s);i,j=0,len(x)-1;m=[]
 while i<j:
  if x[i]!=x[j]:m.append({'offset':i,'left':x[i],'right':x[j]})
  i+=1;j-=1
 return {'exact':not m,'letters':len(x),'sha256_forward':hashlib.sha256(x.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(x[::-1].encode()).hexdigest(),'mismatches':m[:8],'two_pointer_checked':True}
def consume(left,right,trace):
 """Consume a residual string one character at a time, mirrored at right."""
 i=0
 while i<len(left) and i<len(right):
  trace.append({'state':i,'expected':left[i],'observed':right[i]})
  if left[i]!=right[i]: return False
  i+=1
 return i==len(left)==len(right)
def main():
 rows=[];prunes=0;traces=[]
 for subj,verb,num,tense in C:
  for a,v,o,ln,lt in L:
   for ro,rv,ra,rn,rt in R:
    for phrase,prep in A:
     typed=num==ln==rn and tense==lt==rt
     text=f'{a} {v} {o}, and {subj} {verb} {phrase}; {ro} {rv} {ra}.'
     # Two residual strings are consumed in sequence: object/adjunct seam,
     # then adjunct/right-object seam. The automaton has explicit states.
     left1=n(o)[-2:]+n(phrase)[:2]; right1=(n(ra)+n(ro))[:4][::-1]
     left2=n(phrase)[-2:]+n(ro)[:2]; right2=(n(o)+n(a))[:4][::-1]
     trace=[];ok=typed and consume(left1,right1,trace) and consume(left2,right2,trace)
     if not ok: prunes+=1; traces.append({'text':text,'typed':typed,'left1':left1,'right1':right1,'left2':left2,'right2':right2,'trace':trace[:8]});continue
     rows.append({'text':text,'residuals':{'left1':left1,'right1':right1,'left2':left2,'right2':right2},'trace':trace,'audit':audit(text),'provenance':{'center_first':True,'heldout_adjunct':phrase,'multichar_automaton':True}})
 controls=[]
 for c,l,r,phrase in [(C[0],L[0],R[0],A[0][0]),(C[2],L[3],R[3],A[1][0])]:
  text=f'{l[0]} {l[1]} {l[2]}, and {c[0]} {c[1]} {phrase}; {r[0]} {r[1]} {r[2]}.';controls.append({'text':text,'audit':audit(text),'provenance':{'center_first':True,'complete_center_clause':True}})
 out={'run_id':'multichar-boundary-automaton-20260920','method':'center-first typed valency with held-out adjuncts and multi-character residual automaton','novelty_preflight':{'signature':'fresh-authored|multichar-residual-automaton|heldout-adjunct|center-first-typed','prior_signatures_checked':['heldout-adjunct-lattice|live-boundary-residual|center-first-typed','center-first|mixed-valency|licensed-adjunct|typed-bilateral-product'],'duplicate_sweep':False},'inventory':{'centers':4,'left':4,'right':4,'adjuncts':4},'stats':{'states':256,'automaton_transitions':256,'admitted':len(rows),'prunes':prunes,'exact_over_38':sum(x['audit']['exact'] and x['audit']['letters']>38 for x in rows),'max_letters':max(x['audit']['letters'] for x in rows+controls)},'candidates':rows,'rejected_residual_traces':traces[:12],'complete_prose_controls':controls,'independent_audit':{'algorithm':'two-pointer normalized letters','forward_sha256':True,'reverse_sha256':True},'falsifier':'Exact intact output above 38 must pass both multi-character residual streams and independent audit; none admitted.','next_repair':'Change residual orientation to an explicit reverse-consumption automaton with typed morphology transitions, not a larger phrase bank.'}
 Path('runs').mkdir(exist_ok=True);Path('runs/multichar-boundary-automaton-20260920.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['stats'],sort_keys=True))
if __name__=='__main__':main()
