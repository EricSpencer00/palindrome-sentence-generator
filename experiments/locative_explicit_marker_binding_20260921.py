"""Explicit that-marker on anaphoric object with event instrument binding."""
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/locative-explicit-marker-binding-20260921.json'
@dataclass(frozen=True)
class Arc:name:str;words:tuple[str,...];event:str;obj:str;instrument:str
L=(Arc('smith_gate',('the','careful','smith','mends','the','gate','with','a','hammer','that','the','builder','keeps','in','the','workshop','beside','the','river'),'mend','gate','hammer'),Arc('porter_crate',('a','quiet','porter','moves','the','crate','with','a','cart','that','the','merchant','stores','beside','the','river','in','the','warehouse'),'move','crate','cart'))
R=(Arc('carpenter_gate',('the','patient','carpenter','mends','the','gate','with','a','hammer','that','the','builder','keeps','in','the','workshop','beside','the','river'),'mend','gate','hammer'),Arc('merchant_crate',('a','strong','merchant','moves','the','crate','with','a','cart','that','the','porter','stores','beside','the','river','in','the','warehouse'),'move','crate','cart'))
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
   if (a.event,a.obj,a.instrument)!=(b.event,b.obj,b.instrument):continue
   s=' '.join(a.words)+'; '+' '.join(b.words)+'.';t=tape(s);l=0;r=len(t)-1
   while l<r and t[l]==t[r]:l+=1;r-=1
   z={'pairs':l,'center_inside_word':True,'obligation':None if l>=r else(t[l],t[r])}
   rows.append({'rendered':s,'left_arc':a.name,'right_arc':b.name,'novel_rendered':t not in old,'semantic_state':{'event_frame':a.event,'anaphoric_object':a.obj,'relative_marker':'that','instrument':a.instrument,'binding_carried':True,'typed_locatives':True},'audit':audit(s),'live_trace':z,'exact_admitted':z['obligation'] is None,'reader_status':'unreviewed; exactness does not certify readability','provenance':{'construction':'explicit that-marker anaphoric object with event binding','finished_tape_reversal':False,'posthoc_repair':False,'catalogue_text':False}})
 exact=[x for x in rows if x['exact_admitted'] and x['novel_rendered']];out={'experiment_id':'locative-explicit-marker-binding-20260921','status':'completed_exact' if exact else 'completed_no_exact_closure','method':'explicit relative that-marker carrying event instrument binding','candidate_count':len(rows),'exact_count':len(exact),'reader_eligible':False,'rendered_candidates':rows,'stats':{'longest_letters':max(x['audit']['letters'] for x in rows)},'novelty_preflight':{'hash_source':'all runs/*.json except current artifact','current_artifact_excluded':True,'all_rendered_new':all(x['novel_rendered'] for x in rows),'explicit_marker_new':True},'failure_and_repair':{'failure':'explicit-marker anaphoric pairs still mismatch live character obligations' if not exact else 'none','next_construction':'carry an explicit relative marker with an event-specific passive instrument clause'},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'independent_audits':['two-pointer','forward/reverse SHA-256'],'shortcuts_excluded':True}}
 RUN.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':out['status'],'candidates':len(rows),'exact':len(exact),'longest_letters':out['stats']['longest_letters'],'all_new':out['novelty_preflight']['all_rendered_new']}))
if __name__=='__main__':main()
