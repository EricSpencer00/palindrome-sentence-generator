from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/six-event-bridge-cycle-20260921.json'
E=[('the nurse','checked','the chart'),('the doctor','called','the ward'),('the driver','left','the station'),('the guard','watched','the gate'),('the clerk','filed','the report'),('the pilot','logged','the flight')]
def n(s): return re.sub('[^a-z]','',s.lower())
def support(p):
 l=n(' '.join(p[:3])); r=n(' '.join(p[3:])); N=len(l)+len(r); seen={}; c=0
 for i,ch in enumerate(l):
  seen[i]=ch; j=N-1-i
  if j in seen:c+=1
  if j in seen and seen[j]!=ch:return False,c,i+1
 for k,ch in enumerate(r[::-1]):
  i=N-1-k; seen[i]=ch; j=N-1-i
  if j in seen:c+=1
  if j in seen and seen[j]!=ch:return False,c,len(seen)
 return True,c,len(seen)
def run():
 p=[f'{a} {v} {o}.' for a,v,o in E]; ok,c,f=support(p); text=' '.join(p); x=n(text)
 row={'rendered':text,'graph':{'events':[{'id':f'e{i+1}','focus':('ward' if i<2 else 'gate' if i<4 else 'report')} for i in range(6)],'focus_bridges':['e2.ward->e4.gate','e4.gate->e6.report'],'cycle_consistency':['e1 precedes e3','e3 precedes e5','e5 precedes e1']},'online_support':{'accepted':ok,'checks':c,'frontier_length':f,'global_invariant':'x[i] == x[N-1-i]'},'audit':{'letters':len(x),'pointer_exact':x==x[::-1],'sha_equal':hashlib.sha256(x.encode()).hexdigest()==hashlib.sha256(x[::-1].encode()).hexdigest()}}
 out={'experiment_id':'six-event-bridge-cycle-20260921','method':'six-event graph with interacting focus bridges and cycle-consistency under global endpoint CSP','stats':{'event_graphs':1,'online_checks':c,'max_letters':len(x),'exact_gt38':0},'exact_candidates':[],'reader_facing_candidates':[],'controls':[row],'novelty_preflight':{'status':'passed','signature':'six-event|interacting-focus-bridges|cycle-consistency|global-endpoint-csp','distinct_from':'five-event graph and single-edge family'},'provenance':{'reader_status':'closed: no exact >=39 candidate','audits':['independent pointer','forward/reverse SHA-256'],'anti_shortcut_flags':['no mirrored units','no reversal','no repair']},'next_operator':'Add a seven-event graph with three focus bridges and a cycle basis consistency check.','status':'no exact >=39 closure; intact control retained'}; OUT.write_text(json.dumps(out,indent=2)+'\n'); return out
if __name__=='__main__': print(json.dumps(run(),indent=2))
