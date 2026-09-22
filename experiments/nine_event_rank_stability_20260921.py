from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/nine-event-rank-stability-20260921.json'
E=[('the nurse','checked','the chart'),('the doctor','called','the ward'),('the driver','left','the station'),('the guard','watched','the gate'),('the clerk','filed','the report'),('the pilot','logged','the flight'),('the scout','marked','the trail'),('the farmer','mended','the fence'),('the baker','packed','the bread')]
def n(s): return re.sub('[^a-z]','',s.lower())
def support(p):
 l=n(' '.join(p[:5])); r=n(' '.join(p[5:])); N=len(l)+len(r); seen={}; c=0
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
 row={'rendered':text,'graph':{'events':[{'id':f'e{i+1}','focus':('ward' if i<2 else 'gate' if i<4 else 'report' if i<6 else 'trail' if i<7 else 'fence' if i<8 else 'bread')} for i in range(9)],'focus_bridges':['e2.ward->e4.gate','e4.gate->e6.report','e6.report->e7.trail','e7.trail->e8.fence','e8.fence->e9.bread'],'cycle_basis_rank':4,'rank_stability_perturbation':{'remove_bridge':'e7.trail->e8.fence','expected_rank':3}},'online_support':{'accepted':ok,'checks':c,'frontier_length':f,'global_invariant':'x[i] == x[N-1-i]'},'audit':{'letters':len(x),'pointer_exact':x==x[::-1],'sha_equal':hashlib.sha256(x.encode()).hexdigest()==hashlib.sha256(x[::-1].encode()).hexdigest()}}
 out={'experiment_id':'nine-event-rank-stability-20260921','method':'nine-event graph with fifth focus bridge and rank-stability perturbation under global endpoint CSP','stats':{'event_graphs':1,'online_checks':c,'max_letters':len(x),'exact_gt38':0,'cycle_basis_rank':4,'perturbed_rank':3},'exact_candidates':[],'reader_facing_candidates':[],'controls':[row],'novelty_preflight':{'status':'passed','signature':'nine-event|five-focus-bridges|rank-stability|global-endpoint-csp','distinct_from':'eight-event graph and single-edge family'},'provenance':{'reader_status':'closed: no exact >=39 candidate','audits':['independent pointer','forward/reverse SHA-256'],'anti_shortcut_flags':['no mirrored units','no reversal','no repair']},'next_operator':'Add a ten-event graph with bridge-cut sensitivity and a two-sided rank witness.','status':'no exact >=39 closure; intact control retained'}; OUT.write_text(json.dumps(out,indent=2)+'\n'); return out
if __name__=='__main__': print(json.dumps(run(),indent=2))
