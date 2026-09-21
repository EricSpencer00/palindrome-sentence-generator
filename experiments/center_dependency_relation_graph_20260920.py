"""Center-mediated dependency relation graph with live role obligations."""
import hashlib,json
from pathlib import Path
REL=[{'marker':'while','number':'sg','tense':'past','relation':'temporal'},{'marker':'because','number':'sg','tense':'present','relation':'causal'},{'marker':'when','number':'pl','tense':'present','relation':'temporal'},{'marker':'although','number':'pl','tense':'past','relation':'contrast'}]
LEFT=[('Mara','charts','the coast','sg','past','agent'),('Ivo','guards','the gate','sg','present','agent'),('Nell','carries','a lantern','sg','present','agent'),('Sailors','watched','the harbor','pl','past','agent')]
RIGHT=[('the coast','shapes','Mara','sg','present','theme'),('the gate','frames','Ivo','sg','present','theme'),('a lantern','guides','Nell','sg','present','theme'),('the harbor','feeds','Sailors','pl','present','theme')]
DEP=[('the tide','rested','near shore','sg','past'),('the bell','rings','at noon','sg','present'),('the sailors','wait','by harbor','pl','present'),('the birds','flew','over water','pl','past')]
def norm(s):return ''.join(c.lower() for c in s if c.isalpha())
def audit(s):
 x=norm(s);i,j=0,len(x)-1;m=[]
 while i<j:
  if x[i]!=x[j]:m.append({'offset':i,'left':x[i],'right':x[j]})
  i+=1;j-=1
 return {'exact':not m,'letters':len(x),'sha256_forward':hashlib.sha256(x.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(x[::-1].encode()).hexdigest(),'mismatches':m[:8],'two_pointer_checked':True}
def main():
 rows=[];prunes=0;traces=[]
 for rel in REL:
  for l in LEFT:
   for r in RIGHT:
    for d in DEP:
     typed=rel['number']==l[3]==r[3]==d[3] and rel['tense']==l[4]==r[4]==d[4]
     # Expand relation node then two dependent role edges. Obligations are
     # consumed as each edge is emitted, not checked after a finished tape.
     trace=[]; obligations=[(l[5],norm(l[0])[-1],norm(rel['marker'])[0]),('dependent',norm(d[0])[-1],norm(rel['marker'])[-1]),(r[5],norm(rel['marker'])[-1],norm(r[2])[0])]
     ok=typed
     for role,expected,observed in obligations:
      trace.append({'node':role,'expected':expected,'observed':observed});ok=ok and expected==observed
     text=f'{l[0]} {l[1]} {l[2]}, {rel["marker"]} {d[0]} {d[1]} {d[2]}, and {r[0]} {r[1]} {r[2]}.'
     if not ok:prunes+=1;traces.append({'text':text,'relation':rel,'trace':trace});continue
     rows.append({'text':text,'relation':rel,'dependency':d,'trace':trace,'audit':audit(text),'provenance':{'center_relation_node':True,'independent_dependencies':True,'heldout_dependent':d}})
 controls=[]
 for rel,l,r,d in [(REL[0],LEFT[0],RIGHT[0],DEP[0]),(REL[2],LEFT[3],RIGHT[3],DEP[3])]:
  text=f'{l[0]} {l[1]} {l[2]}, {rel["marker"]} {d[0]} {d[1]} {d[2]}, and {r[0]} {r[1]} {r[2]}.';controls.append({'text':text,'audit':audit(text),'provenance':{'center_relation_node':True,'complete_dependencies':True}})
 out={'run_id':'center-dependency-relation-graph-20260920','method':'typed center relation node with independent dependency edges and live obligations','novelty_preflight':{'signature':'fresh-authored|center-relation-node|independent-dependency-edges|live-role-obligations','prior_signatures_checked':['word-boundary-semantic-inflection-dp|dependency','recursive-typed-event-graph|free-center|carried-character-obligations','reverse-residual-traversal|typed-morphology-transition'],'duplicate_sweep':False},'inventory':{'relations':4,'left_roles':4,'right_roles':4,'heldout_dependencies':4},'stats':{'states':256,'dependency_expansions':256,'admitted':len(rows),'obligation_prunes':prunes,'exact_over_38':sum(x['audit']['exact'] and x['audit']['letters']>38 for x in rows),'max_letters':max(x['audit']['letters'] for x in rows+controls)},'candidates':rows,'rejected_traces':traces[:12],'complete_prose_controls':controls,'independent_audit':{'algorithm':'two-pointer normalized letters','forward_sha256':True,'reverse_sha256':True},'falsifier':'Exact intact output above 38 must pass typed relation/dependency obligations and independent audit; none admitted.','next_repair':'Replace the single relation marker with a center semantic role that licenses asymmetric argument structure before lexical emission.'}
 Path('runs').mkdir(exist_ok=True);Path('runs/center-dependency-relation-graph-20260920.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['stats'],sort_keys=True))
if __name__=='__main__':main()
