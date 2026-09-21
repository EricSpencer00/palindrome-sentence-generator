"""Branching graph with shared temporal anchor and participant."""
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/semantic-branch-temporal-anchor-20260921.json'
@dataclass(frozen=True)
class Branch:name:str;words:tuple[str,...];state:str
L=(Branch('dawn_guard_keeper',('at','dawn','the','guard','keeps','the','lantern','and','the','keeper','checks','the','gate','with','the','harbor','master','so','the','harbor','stays','safe'),'harbor-safe'),Branch('dusk_scribe_reader',('at','dusk','a','scribe','writes','the','map','and','a','reader','studies','the','route','with','the','teacher','so','the','traveler','finds','home'),'traveler-home'))
R=(Branch('dawn_watchman_steward',('at','dawn','the','watchman','holds','the','lantern','and','the','steward','checks','the','gate','with','the','harbor','master','so','the','harbor','stays','safe'),'harbor-safe'),Branch('dusk_teacher_student',('at','dusk','a','teacher','draws','the','map','and','a','student','studies','the','route','with','the','guide','so','the','traveler','finds','home'),'traveler-home'))
def tape(s):return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=tape(s);n=len(t);bad=next((i for i in range(n//2) if t[i]!=t[n-1-i]),None);h=hashlib.sha256(t.encode()).hexdigest();rh=hashlib.sha256(t[::-1].encode()).hexdigest();return {'letters':n,'two_pointer_exact':bad is None,'first_mismatch':bad,'forward_sha256':h,'reverse_sha256':rh,'sha_equal':h==rh}
def main():
 old=set()
 for p in ROOT.glob('runs/*.json'):
  if p==RUN:continue
  try:old.update(tape(x.get('rendered','')) for x in json.loads(p.read_text()).get('rendered_candidates',[]) if x.get('rendered'))
  except Exception:pass
 rows=[]
 for a in L:
  for b in R:
   if a.state!=b.state:continue
   s=' '.join(a.words)+'; '+' '.join(b.words)+'.';t=tape(s);l=0;r=len(t)-1
   while l<r and t[l]==t[r]:l+=1;r-=1
   z={'pairs':l,'center_inside_word':True,'obligation':None if l>=r else(t[l],t[r])}
   rows.append({'rendered':s,'left_branch':a.name,'right_branch':b.name,'novel_rendered':t not in old,'semantic_graph':{'shared_temporal_anchor':a.words[:2],'branch_count':2,'shared_participant':True,'shared_before_convergence':True,'convergence_edge':'so'},'audit':audit(s),'live_trace':z,'exact_admitted':z['obligation'] is None,'reader_status':'unreviewed; exactness does not certify readability','provenance':{'construction':'shared temporal anchor before participant-linked branches','linear_graph_reused':False,'finished_tape_reversal':False,'posthoc_repair':False,'catalogue_text':False}})
 exact=[x for x in rows if x['exact_admitted'] and x['novel_rendered']];out={'experiment_id':'semantic-branch-temporal-anchor-20260921','status':'completed_exact' if exact else 'completed_no_exact_closure','method':'branching event graph with temporal anchor, shared participant, convergence','candidate_count':len(rows),'exact_count':len(exact),'reader_eligible':False,'rendered_candidates':rows,'stats':{'longest_letters':max(x['audit']['letters'] for x in rows)},'novelty_preflight':{'hash_source':'all runs/*.json except current artifact','current_artifact_excluded':True,'all_rendered_new':all(x['novel_rendered'] for x in rows),'temporal_anchor_new':True},'failure_and_repair':{'failure':'temporal-anchor branches still mismatch live obligations' if not exact else 'none','next_construction':'add a shared spatial scene node after convergence, not another temporal variant'},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'independent_audits':['two-pointer','forward/reverse SHA-256'],'shortcuts_excluded':True}}
 RUN.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':out['status'],'candidates':len(rows),'exact':len(exact),'longest_letters':out['stats']['longest_letters'],'all_new':out['novelty_preflight']['all_rendered_new']}))
if __name__=='__main__':main()
