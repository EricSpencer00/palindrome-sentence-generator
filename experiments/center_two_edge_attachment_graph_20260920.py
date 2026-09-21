"""Center relation expanded through two owned dependency edges."""
import hashlib,json
from pathlib import Path
R=[('while','temporal'),('because','causal'),('when','temporal'),('although','contrast')]
E1=[('the tide','rested','near shore','setting'),('the bell','rings','at noon','time'),('the sailors','wait','by harbor','location'),('the birds','flew','over water','setting')]
E2=[('Mara','listens','closely','agent'),('Ivo','works','quietly','agent'),('Nell','walks','home','agent'),('Sailors','sing','together','agent')]
L=[('Mara','charts','the coast','sg','past'),('Ivo','guards','the gate','sg','present'),('Nell','carries','a lantern','sg','present'),('Sailors','watched','the harbor','pl','past')]
def norm(s):return ''.join(c.lower() for c in s if c.isalpha())
def audit(s):
 x=norm(s);i,j=0,len(x)-1;m=[]
 while i<j:
  if x[i]!=x[j]:m.append({'offset':i,'left':x[i],'right':x[j]})
  i+=1;j-=1
 return {'exact':not m,'letters':len(x),'sha256_forward':hashlib.sha256(x.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(x[::-1].encode()).hexdigest(),'mismatches':m[:8],'two_pointer_checked':True}
def main():
 rows=[];traces=[];prunes=0
 for marker,rel in R:
  for a,v,o,num,tense in L:
   for s1,v1,x1,owner1 in E1:
    for s2,v2,x2,owner2 in E2:
     text=f'{a} {v} {o}, {marker} {s1} {v1} {x1}, and {s2} {v2} {x2}.'
     # sequential attachment expansion, with ownership preventing edge collapse
     trace=[]; ok=owner1!=owner2
     for node,expected,observed in [('edge1',norm(o)[-1],norm(s1)[0]),('edge2',norm(x1)[-1],norm(s2)[0])]:
      trace.append({'node':node,'owner':owner1 if node=='edge1' else owner2,'expected':expected,'observed':observed});ok=ok and expected==observed
     if not ok:prunes+=1;traces.append({'text':text,'trace':trace});continue
     rows.append({'text':text,'owners':[owner1,owner2],'trace':trace,'audit':audit(text),'provenance':{'center_relation':marker,'sequential_edges':True,'heldout_edges':True}})
 controls=[]
 for m,l,e1,e2 in [(R[0][0],L[0],E1[0],E2[0]),(R[2][0],L[3],E1[3],E2[3])]:
  text=f'{l[0]} {l[1]} {l[2]}, {m} {e1[0]} {e1[1]} {e1[2]}, and {e2[0]} {e2[1]} {e2[2]}.';controls.append({'text':text,'audit':audit(text),'provenance':{'complete_two_edge_clause':True}})
 out={'run_id':'center-two-edge-attachment-graph-20260920','method':'center relation with sequential typed attachment-owned dependency edges','novelty_preflight':{'signature':'fresh-authored|center-relation|sequential-two-edges|typed-attachment-ownership|live-obligations','prior_signatures_checked':['center-relation-node|independent-dependency-edges|live-role-obligations','typed-bilateral-event-product|dual-live-obligations'],'duplicate_sweep':False},'inventory':{'relations':4,'left_roots':4,'edge1_alternatives':4,'edge2_alternatives':4},'stats':{'states':256,'sequential_expansions':256,'admitted':len(rows),'prunes':prunes,'exact_over_38':sum(x['audit']['exact'] and x['audit']['letters']>38 for x in rows),'max_letters':max(x['audit']['letters'] for x in rows+controls)},'candidates':rows,'rejected_traces':traces[:12],'complete_prose_controls':controls,'independent_audit':{'algorithm':'two-pointer normalized letters','forward_sha256':True,'reverse_sha256':True},'falsifier':'Exact intact output above 38 must retain distinct attachment owners and pass both live obligations; none admitted.','next_repair':'Add a typed center-to-edge ownership state that permits compatible owner inheritance without collapsing the two edges.'}
 Path('runs').mkdir(exist_ok=True);Path('runs/center-two-edge-attachment-graph-20260920.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['stats'],sort_keys=True))
if __name__=='__main__':main()
