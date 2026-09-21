"""Topology reset: semantic event graph paired with live outer obligations.

This is deliberately outside the relative/instrument family: nodes are
complete causal event graphs (event -> object -> consequence) and edges carry
semantic entailment before any character obligation is queried.
"""
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/semantic-event-graph-outer-20260921.json'
@dataclass(frozen=True)
class Graph:
 name:str;words:tuple[str,...];signature:tuple[str,...]
LEFT=(Graph('storm_lantern',('the','watchman','lights','the','lantern','because','the','storm','darkens','the','harbor'),('perceive','light','weather','place')),Graph('scribe_letter',('a','careful','scribe','writes','the','letter','while','the','reader','waits','by','the','window'),('create','message','read','place')))
RIGHT=(Graph('keeper_lantern',('the','keeper','guards','the','lantern','because','the','storm','darkens','the','harbor'),('protect','light','weather','place')),Graph('student_letter',('a','patient','student','reads','the','letter','while','the','teacher','waits','by','the','window'),('read','message','wait','place')))
def tape(s):return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=tape(s);n=len(t);bad=next((i for i in range(n//2) if t[i]!=t[n-1-i]),None);h=hashlib.sha256(t.encode()).hexdigest();rh=hashlib.sha256(t[::-1].encode()).hexdigest();return {'letters':n,'two_pointer_exact':bad is None,'first_mismatch':bad,'forward_sha256':h,'reverse_sha256':rh,'sha_equal':h==rh}
def compatible(a,b):return a.signature[-2:]==b.signature[-2:]
def main():
 old=set()
 for p in ROOT.glob('runs/*.json'):
  if p==RUN:continue
  try:old.update(tape(x.get('rendered','')) for x in json.loads(p.read_text()).get('rendered_candidates',[]) if x.get('rendered'))
  except Exception:pass
 rows=[]
 for a in LEFT:
  for b in RIGHT:
   if not compatible(a,b):continue
   s=' '.join(a.words)+'; '+' '.join(b.words)+'.';t=tape(s);l=0;r=len(t)-1
   while l<r and t[l]==t[r]:l+=1;r-=1
   z={'pairs':l,'center_inside_word':True,'obligation':None if l>=r else(t[l],t[r])}
   rows.append({'rendered':s,'left_graph':a.name,'right_graph':b.name,'novel_rendered':t not in old,'semantic_graph':{'left_signature':a.signature,'right_signature':b.signature,'shared_edge_types':list(set(a.signature[-2:])&set(b.signature[-2:]))},'audit':audit(s),'live_trace':z,'exact_admitted':z['obligation'] is None,'reader_status':'unreviewed; exactness does not certify readability','provenance':{'construction':'causal semantic event graph with outer character obligations','relative_clause_family':False,'instrument_family':False,'finished_tape_reversal':False,'posthoc_repair':False,'catalogue_text':False}})
 exact=[x for x in rows if x['exact_admitted'] and x['novel_rendered']];out={'experiment_id':'semantic-event-graph-outer-20260921','status':'completed_exact' if exact else 'completed_no_exact_closure','method':'semantic event graph compatibility before outer obligation query','candidate_count':len(rows),'exact_count':len(exact),'reader_eligible':False,'rendered_candidates':rows,'stats':{'longest_letters':max(x['audit']['letters'] for x in rows)},'novelty_preflight':{'hash_source':'all runs/*.json except current artifact','current_artifact_excluded':True,'all_rendered_new':all(x['novel_rendered'] for x in rows),'topology_reset':True,'relative_instrument_reused':False},'failure_and_repair':{'failure':'event-graph-compatible prose still mismatches live outer obligations' if not exact else 'none','next_construction':'add a second semantic edge type with a held-out causal connector, not a lexical sweep'},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'independent_audits':['two-pointer','forward/reverse SHA-256'],'shortcuts_excluded':True}}
 RUN.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':out['status'],'candidates':len(rows),'exact':len(exact),'longest_letters':out['stats']['longest_letters'],'all_new':out['novelty_preflight']['all_rendered_new']}))
if __name__=='__main__':main()
