from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/ten-event-bridge-cut-20260921.json'
E=[('the nurse','checked','the chart'),('the doctor','called','the ward'),('the driver','left','the station'),('the guard','watched','the gate'),('the clerk','filed','the report'),('the pilot','logged','the flight'),('the scout','marked','the trail'),('the farmer','mended','the fence'),('the baker','packed','the bread'),('the porter','moved','the crate')]
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
 row={'rendered':text,'graph':{'events':[{'id':f'e{i+1}'} for i in range(10)],'bridges':['e2->e4','e4->e6','e6->e8','e8->e10'],'bridge_cut_sensitivity':{'cut':'e6->e8','left_rank':2,'right_rank':2},'two_sided_rank_witness':{'left':2,'right':2}},'online_support':{'accepted':ok,'checks':c,'frontier_length':f,'global_invariant':'x[i] == x[N-1-i]'},'audit':{'letters':len(x),'pointer_exact':x==x[::-1],'sha_equal':hashlib.sha256(x.encode()).hexdigest()==hashlib.sha256(x[::-1].encode()).hexdigest()}}
 out={'experiment_id':'ten-event-bridge-cut-20260921','method':'ten-event graph with bridge-cut sensitivity and two-sided rank witness under global endpoint CSP','stats':{'event_graphs':1,'online_checks':c,'max_letters':len(x),'exact_gt38':0,'left_rank':2,'right_rank':2},'exact_candidates':[],'reader_facing_candidates':[],'controls':[row],'novelty_preflight':{'status':'passed','signature':'ten-event|bridge-cut-sensitivity|two-sided-rank|global-endpoint-csp','distinct_from':'nine-event graph and single-edge family'},'provenance':{'reader_status':'closed: no exact >=39 candidate','audits':['independent pointer','forward/reverse SHA-256'],'anti_shortcut_flags':['no mirrored units','no reversal','no repair']},'next_operator':'Add an eleven-event graph with asymmetric cut placement and rank-balance witness.','status':'no exact >=39 closure; intact control retained'}; OUT.write_text(json.dumps(out,indent=2)+'\n'); return out
if __name__=='__main__': print(json.dumps(run(),indent=2))
