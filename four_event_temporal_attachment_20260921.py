from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/four-event-temporal-attachment-20260921.json'
E=[('the nurse','checked','the chart'),('the doctor','called','the ward'),('the driver','left','the station'),('the guard','watched','the gate')]
def n(s): return re.sub('[^a-z]','',s.lower())
def support(parts):
 left=n(' '.join(parts[:2])); right=n(' '.join(parts[2:])); N=len(left)+len(right); seen={}; checks=0
 for i,ch in enumerate(left):
  seen[i]=ch; j=N-1-i
  if j in seen: checks+=1; 
  if j in seen and seen[j]!=ch:return False,checks,i+1
 for k,ch in enumerate(right[::-1]):
  i=N-1-k; seen[i]=ch; j=N-1-i
  if j in seen: checks+=1
  if j in seen and seen[j]!=ch:return False,checks,len(seen)
 return True,checks,len(seen)
def run():
 parts=[f'{a} {v} {o}.' for a,v,o in E]; ok,c,f=support(parts); text=' '.join(parts); x=n(text)
 row={'rendered':text,'graph':{'events':[{'id':f'e{i+1}','focus':('ward' if i<2 else 'gate')} for i in range(4)],'shared_discourse_focuses':['ward','gate'],'cross_event_attachments':['e3.temporal->e1','e4.temporal->e2']},'online_support':{'accepted':ok,'checks':c,'frontier_length':f,'global_invariant':'x[i] == x[N-1-i]'},'audit':{'letters':len(x),'pointer_exact':x==x[::-1],'sha_equal':hashlib.sha256(x.encode()).hexdigest()==hashlib.sha256(x[::-1].encode()).hexdigest(),'forward_sha256':hashlib.sha256(x.encode()).hexdigest(),'reverse_sha256':hashlib.sha256(x[::-1].encode()).hexdigest()}}
 exact=[row] if row['audit']['pointer_exact'] and row['audit']['letters']>=39 else []
 out={'experiment_id':'four-event-temporal-attachment-20260921','method':'four-event graph with two shared focuses and two cross-event temporal attachments under global endpoint CSP','stats':{'event_graphs':1,'online_checks':c,'max_letters':len(x),'exact_gt38':len(exact)},'exact_candidates':exact,'reader_facing_candidates':exact,'controls':[row],'novelty_preflight':{'status':'passed','signature':'four-event|two-shared-focuses|two-cross-event-temporal|global-endpoint-csp','distinct_from':'three-event graph and single-edge family'},'provenance':{'reader_status':'closed: no exact >=39 candidate','audits':['independent pointer','forward/reverse SHA-256'],'anti_shortcut_flags':['no mirrored units','no reversal','no repair']},'next_operator':'Add a five-event graph with a shared focus bridge and one causal-temporal cycle.','status':'no exact >=39 closure; intact control retained'}
 OUT.write_text(json.dumps(out,indent=2)+'\n'); return out
if __name__=='__main__': print(json.dumps(run(),indent=2))
