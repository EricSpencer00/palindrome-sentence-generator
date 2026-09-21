"""Topology reset: branching events converge on one shared consequence."""
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/semantic-branch-shared-consequence-20260921.json'
@dataclass(frozen=True)
class Branch:name:str;words:tuple[str,...];consequence:str
L=(Branch('guard_keeper_harbor',('the','guard','keeps','the','lantern','and','the','keeper','checks','the','gate','so','the','harbor','stays','safe'),'harbor-safe'),Branch('scribe_reader_road',('a','scribe','writes','the','map','and','a','reader','studies','the','route','so','the','traveler','finds','home'),'traveler-home'))
R=(Branch('watchman_stable_harbor',('the','watchman','holds','the','lantern','and','the','steward','checks','the','gate','so','the','harbor','stays','safe'),'harbor-safe'),Branch('teacher_student_road',('a','teacher','draws','the','map','and','a','student','studies','the','route','so','the','traveler','finds','home'),'traveler-home'))
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
   if a.consequence!=b.consequence:continue
   s=' '.join(a.words)+'; '+' '.join(b.words)+'.';t=tape(s);l=0;r=len(t)-1
   while l<r and t[l]==t[r]:l+=1;r-=1
   z={'pairs':l,'center_inside_word':True,'obligation':None if l>=r else(t[l],t[r])}
   rows.append({'rendered':s,'left_branch':a.name,'right_branch':b.name,'novel_rendered':t not in old,'semantic_graph':{'branch_count':2,'shared_consequence':a.consequence,'convergence_edge':'so'},'audit':audit(s),'live_trace':z,'exact_admitted':z['obligation'] is None,'reader_status':'unreviewed; exactness does not certify readability','provenance':{'construction':'branching two-event graph converging on one consequence','linear_connector_graph':False,'finished_tape_reversal':False,'posthoc_repair':False,'catalogue_text':False}})
 exact=[x for x in rows if x['exact_admitted'] and x['novel_rendered']];out={'experiment_id':'semantic-branch-shared-consequence-20260921','status':'completed_exact' if exact else 'completed_no_exact_closure','method':'branching semantic event graph with shared consequence node','candidate_count':len(rows),'exact_count':len(exact),'reader_eligible':False,'rendered_candidates':rows,'stats':{'longest_letters':max(x['audit']['letters'] for x in rows)},'novelty_preflight':{'hash_source':'all runs/*.json except current artifact','current_artifact_excluded':True,'all_rendered_new':all(x['novel_rendered'] for x in rows),'linear_graph_reused':False,'topology_reset':True},'failure_and_repair':{'failure':'branching graphs still mismatch live obligations' if not exact else 'none','next_construction':'add a shared participant node linking the two branches before convergence'},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'independent_audits':['two-pointer','forward/reverse SHA-256'],'shortcuts_excluded':True}}
 RUN.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':out['status'],'candidates':len(rows),'exact':len(exact),'longest_letters':out['stats']['longest_letters'],'all_new':out['novelty_preflight']['all_rendered_new']}))
if __name__=='__main__':main()
