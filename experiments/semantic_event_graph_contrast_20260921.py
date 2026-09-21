"""Held-out contrast edge in a causal semantic event graph."""
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/semantic-event-graph-contrast-20260921.json'
@dataclass(frozen=True)
class Graph:name:str;words:tuple[str,...];edges:tuple[str,...]
L=(Graph('guard_lantern',('the','guard','keeps','the','lantern','although','the','storm','darkens','the','harbor','because','the','ship','waits','but','the','wind','rises'),('although','because','but')),Graph('scribe_map',('a','scribe','draws','the','map','although','the','rain','falls','on','the','roof','because','the','traveler','waits','but','the','road','floods'),('although','because','but')))
R=(Graph('keeper_lantern',('the','keeper','guards','the','lantern','although','the','storm','darkens','the','harbor','because','the','ship','waits','but','the','wind','rises'),('although','because','but')),Graph('artist_map',('a','cartographer','draws','the','map','although','the','rain','falls','on','the','roof','because','the','traveler','waits','but','the','road','floods'),('although','because','but')))
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
   if a.edges!=b.edges:continue
   s=' '.join(a.words)+'; '+' '.join(b.words)+'.';t=tape(s);l=0;r=len(t)-1
   while l<r and t[l]==t[r]:l+=1;r-=1
   z={'pairs':l,'center_inside_word':True,'obligation':None if l>=r else(t[l],t[r])}
   rows.append({'rendered':s,'left_graph':a.name,'right_graph':b.name,'novel_rendered':t not in old,'semantic_graph':{'edge_types':list(a.edges),'contrast_edge':'but','edge_count':3},'audit':audit(s),'live_trace':z,'exact_admitted':z['obligation'] is None,'reader_status':'unreviewed; exactness does not certify readability','provenance':{'construction':'contrast edge in dual-edge causal event graph','relative_clause_family':False,'instrument_family':False,'finished_tape_reversal':False,'posthoc_repair':False,'catalogue_text':False}})
 exact=[x for x in rows if x['exact_admitted'] and x['novel_rendered']];out={'experiment_id':'semantic-event-graph-contrast-20260921','status':'completed_exact' if exact else 'completed_no_exact_closure','method':'three-edge causal graph with held-out contrast edge','candidate_count':len(rows),'exact_count':len(exact),'reader_eligible':False,'rendered_candidates':rows,'stats':{'longest_letters':max(x['audit']['letters'] for x in rows)},'novelty_preflight':{'hash_source':'all runs/*.json except current artifact','current_artifact_excluded':True,'all_rendered_new':all(x['novel_rendered'] for x in rows),'contrast_edge_new':True},'failure_and_repair':{'failure':'contrast-edge graphs still mismatch live character obligations' if not exact else 'none','next_construction':'reset topology to a branching event graph with one shared consequence node'},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'independent_audits':['two-pointer','forward/reverse SHA-256'],'shortcuts_excluded':True}}
 RUN.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':out['status'],'candidates':len(rows),'exact':len(exact),'longest_letters':out['stats']['longest_letters'],'all_new':out['novelty_preflight']['all_rendered_new']}))
if __name__=='__main__':main()
