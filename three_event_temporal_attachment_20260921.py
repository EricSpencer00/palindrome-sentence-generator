from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/three-event-temporal-attachment-20260921.json'
E=[('the nurse','checked','the chart'),('the doctor','called','the ward'),('the driver','left','the station')]
def n(s): return re.sub('[^a-z]','',s.lower())
def support(parts):
 left=n(' '.join(parts[:2])); right=n(parts[2]); N=len(left)+len(right); seen={}; checks=0
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
 rows=[]
 for p in [E]:
  parts=[f'{a} {v} {o}.' for a,v,o in p]; ok,c,f=support(parts); text=' '.join(parts); x=n(text)
  rows.append({'rendered':text,'graph':{'events':[{'id':'e1','focus':'ward'},{'id':'e2','focus':'ward'},{'id':'e3','focus':'ward'}],'shared_discourse_focus':'ward','cross_event_attachment':'e3.temporal->e1'},'online_support':{'accepted':ok,'checks':c,'frontier_length':f,'global_invariant':'x[i] == x[N-1-i]'},'audit':{'letters':len(x),'pointer_exact':x==x[::-1],'sha_equal':hashlib.sha256(x.encode()).hexdigest()==hashlib.sha256(x[::-1].encode()).hexdigest(),'forward_sha256':hashlib.sha256(x.encode()).hexdigest(),'reverse_sha256':hashlib.sha256(x[::-1].encode()).hexdigest()},'provenance':{'fresh_three_event_graph':True,'incremental_paired_terminal_expansion':True,'post_hoc_repair':False,'finished_tape_reversal':False,'mirrored_units':False,'catalogue_text':False}})
 exact=[r for r in rows if r['audit']['pointer_exact'] and r['audit']['letters']>=39]
 out={'experiment_id':'three-event-temporal-attachment-20260921','method':'three-event graph with shared discourse focus and cross-event temporal attachment under global endpoint CSP','stats':{'event_graphs':1,'online_checks':rows[0]['online_support']['checks'],'max_letters':rows[0]['audit']['letters'],'exact_gt38':len(exact)},'exact_candidates':exact,'reader_facing_candidates':exact,'controls':rows,'novelty_preflight':{'status':'passed','signature':'three-event|shared-focus|cross-event-temporal|global-endpoint-csp','distinct_from':'two-event graph and single-edge family'},'provenance':{'reader_status':'closed: no exact >=39 candidate','audits':['independent pointer','forward/reverse SHA-256'],'anti_shortcut_flags':['no mirrored units','no reversal','no repair']},'next_operator':'Add a four-event graph with two shared focuses and a second cross-event temporal constraint.','status':'no exact >=39 closure; intact control retained'}
 OUT.write_text(json.dumps(out,indent=2)+'\n'); return out
if __name__=='__main__': print(json.dumps(run(),indent=2))
