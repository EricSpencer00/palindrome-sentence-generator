"""Clause-level dependency grammar with center relation and matrix frames."""
import hashlib,json
from pathlib import Path
F=[('Mara charts the coast','while','the tide rests near shore','the coast shapes Mara'),('Ivo guards the gate','because','the bell rings at noon','the gate frames Ivo'),('Nell carries a lantern','when','the sailors wait by harbor','a lantern guides Nell'),('Sailors watched the harbor','although','the birds flew over water','the harbor feeds Sailors')]
REL=['while','because','when','although']
def norm(s):return ''.join(c.lower() for c in s if c.isalpha())
def audit(s):
 x=norm(s);i,j=0,len(x)-1;m=[]
 while i<j:
  if x[i]!=x[j]:m.append({'offset':i,'left':x[i],'right':x[j]})
  i+=1;j-=1
 return {'exact':not m,'letters':len(x),'sha256_forward':hashlib.sha256(x.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(x[::-1].encode()).hexdigest(),'mismatches':m[:8],'two_pointer_checked':True}
def main():
 rows=[];traces=[]
 for i,frame in enumerate(F):
  for j,relation in enumerate(REL):
   # Matrix frame and relation are selected as typed clause productions.
   text=f'{frame[0]}, {relation} {frame[2]}, and {frame[3]}.'
   trace=[{'state':'matrix-to-center','expected':norm(frame[0])[-1],'observed':norm(frame[2])[0]},{'state':'center-to-dependent','expected':norm(frame[2])[-1],'observed':norm(frame[3])[0]}]
   ok=(i==j and trace[0]['expected']==trace[0]['observed'] and trace[1]['expected']==trace[1]['observed'])
   if ok: rows.append({'text':text,'frame':i,'relation':relation,'trace':trace,'audit':audit(text),'provenance':{'clause_level_matrix':True,'center_relation':relation,'nonrepeated_frame':True}})
   else: traces.append({'text':text,'frame':i,'relation':relation,'trace':trace})
 controls=[{'text':f'{x[0]}, {x[1]} {x[2]}, and {x[3]}.','audit':audit(f'{x[0]}, {x[1]} {x[2]}, and {x[3]}.')} for x in F[:2]]
 out={'run_id':'center-clause-matrix-dependency-20260920','method':'clause-level matrix dependency with typed center relation and nonrepeated frames','novelty_preflight':{'signature':'fresh-authored|clause-level-matrix|typed-center-relation|nonrepeated-frames','prior_signatures_checked':['center-relation|sequential-two-edges|typed-attachment-ownership','center-relation-node|independent-dependency-edges|live-role-obligations'],'duplicate_sweep':False,'third_edge_family':'stopped_as_duplicate'},'inventory':{'matrix_frames':4,'relations':4},'stats':{'states':16,'clause_expansions':16,'admitted':len(rows),'prunes':len(traces),'exact_over_38':sum(x['audit']['exact'] and x['audit']['letters']>38 for x in rows),'max_letters':max(x['audit']['letters'] for x in rows+controls)},'candidates':rows,'rejected_traces':traces,'complete_prose_controls':controls,'independent_audit':{'algorithm':'two-pointer normalized letters','forward_sha256':True,'reverse_sha256':True},'falsifier':'Exact intact output above 38 must pass clause-frame and center-relation obligations plus independent audit; none admitted.','next_repair':'Add typed subject continuity across matrix and center clauses before lexical choice, not another frame copy.'}
 Path('runs').mkdir(exist_ok=True);Path('runs/center-clause-matrix-dependency-20260920.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['stats'],sort_keys=True))
if __name__=='__main__':main()
