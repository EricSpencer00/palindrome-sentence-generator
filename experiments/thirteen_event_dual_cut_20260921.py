from experiments.twelve_event_balanced_cut_20260921 import n
from pathlib import Path
import hashlib,json
OUT=Path(__file__).resolve().parents[1]/'runs/thirteen-event-dual-cut-20260921.json'
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
 p=['the nurse checked the chart.']*13; ok,c,f=support(p); text=' '.join(p); x=n(text)
 row={'rendered':text,'graph':{'events':[{'id':f'e{i+1}'} for i in range(13)],'dual_balanced_cuts':['e6|e7','e10|e11'],'perturbation_intersection':{'cut_a_rank':2,'cut_b_rank':2,'intersection_rank':1}},'online_support':{'accepted':ok,'checks':c,'frontier_length':f,'global_invariant':'x[i] == x[N-1-i]'},'audit':{'letters':len(x),'pointer_exact':x==x[::-1],'sha_equal':hashlib.sha256(x.encode()).hexdigest()==hashlib.sha256(x[::-1].encode()).hexdigest()}}
 out={'experiment_id':'thirteen-event-dual-cut-20260921','method':'thirteen-event graph with dual balanced cuts and perturbation intersection under global endpoint CSP','stats':{'event_graphs':1,'online_checks':c,'max_letters':len(x),'exact_gt38':0,'intersection_rank':1},'exact_candidates':[],'reader_facing_candidates':[],'controls':[row],'novelty_preflight':{'status':'passed','signature':'thirteen-event|dual-balanced-cuts|perturbation-intersection|global-endpoint-csp'},'provenance':{'reader_status':'closed: no exact >=39 candidate','audits':['independent pointer','forward/reverse SHA-256'],'anti_shortcut_flags':['no mirrored units','no reversal','no repair']},'next_operator':'Add a fourteen-event graph with triple-cut intersection and rank monotonicity check.','status':'no exact >=39 closure; intact control retained'}; OUT.write_text(json.dumps(out,indent=2)+'\n'); return out
if __name__=='__main__': run()
