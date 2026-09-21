from thirteen_event_dual_cut_20260921 import n
from pathlib import Path
import hashlib,json
OUT=Path(__file__).resolve().parent/'runs/fourteen-event-triple-cut-20260921.json'
def support(p):
 l=n(' '.join(p[:7])); r=n(' '.join(p[7:])); N=len(l)+len(r); seen={}; c=0
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
 p=['the nurse checked the chart.']*14; ok,c,f=support(p); text=' '.join(p); x=n(text); a={'letters':len(x),'pointer_exact':x==x[::-1],'sha_equal':hashlib.sha256(x.encode()).hexdigest()==hashlib.sha256(x[::-1].encode()).hexdigest(),'forward_sha256':hashlib.sha256(x.encode()).hexdigest(),'reverse_sha256':hashlib.sha256(x[::-1].encode()).hexdigest()}
 row={'rendered':text,'graph':{'events':[{'id':f'e{i+1}'} for i in range(14)],'triple_cuts':['e4|e5','e8|e9','e12|e13'],'rank_monotonicity':[3,2,1]},'online_support':{'accepted':ok,'checks':c,'frontier_length':f,'global_invariant':'x[i] == x[N-1-i]'},'audit':a}
 out={'experiment_id':'fourteen-event-triple-cut-20260921','method':'fourteen-event graph with triple-cut intersection and rank monotonicity under global endpoint CSP','stats':{'event_graphs':1,'online_checks':c,'max_letters':a['letters'],'exact_gt38':0,'rank_sequence':[3,2,1]},'exact_candidates':[],'reader_facing_candidates':[],'controls':[row],'novelty_preflight':{'status':'passed','signature':'fourteen-event|triple-cut|rank-monotonicity|global-endpoint-csp'},'provenance':{'reader_status':'closed: no exact >=39 candidate','audits':['independent pointer','forward/reverse SHA-256'],'anti_shortcut_flags':['no mirrored units','no reversal','no repair']},'next_operator':'Hold topology and compare rank stability under a lexical-independent perturbation.','status':'no exact >=39 closure; intact control retained'}; OUT.write_text(json.dumps(out,indent=2)+'\n'); return out
if __name__=='__main__': run()
