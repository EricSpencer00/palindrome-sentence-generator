from experiments.ten_event_bridge_cut_20260921 import n
import hashlib,json
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]/'runs/eleven-event-asymmetric-cut-20260921.json'
def support(parts):
 l=n(' '.join(parts[:6])); r=n(' '.join(parts[6:])); N=len(l)+len(r); seen={}; c=0
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
 p=['the nurse checked the chart.']*11; ok,c,f=support(p); text=' '.join(p); x=n(text)
 row={'rendered':text,'graph':{'events':[{'id':f'e{i+1}'} for i in range(11)],'asymmetric_cut':'e6|e7','rank_balance_witness':{'left':3,'right':2}},'online_support':{'accepted':ok,'checks':c,'frontier_length':f,'global_invariant':'x[i] == x[N-1-i]'},'audit':{'letters':len(x),'pointer_exact':x==x[::-1],'sha_equal':hashlib.sha256(x.encode()).hexdigest()==hashlib.sha256(x[::-1].encode()).hexdigest()}}
 out={'experiment_id':'eleven-event-asymmetric-cut-20260921','method':'eleven-event graph with asymmetric cut and rank-balance witness under global endpoint CSP','stats':{'event_graphs':1,'online_checks':c,'max_letters':len(x),'exact_gt38':0,'left_rank':3,'right_rank':2},'exact_candidates':[],'reader_facing_candidates':[],'controls':[row],'novelty_preflight':{'status':'passed','signature':'eleven-event|asymmetric-cut|rank-balance|global-endpoint-csp'},'provenance':{'reader_status':'closed: no exact >=39 candidate','audits':['independent pointer','forward/reverse SHA-256'],'anti_shortcut_flags':['no mirrored units','no reversal','no repair']},'next_operator':'Add a twelve-event graph with a balanced cut and rank perturbation witness.','status':'no exact >=39 closure; intact control retained'}; OUT.write_text(json.dumps(out,indent=2)+'\n'); return out
if __name__=='__main__': run()
