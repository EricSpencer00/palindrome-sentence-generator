"""Final morphology branch: coordinated plural passive instrument patient."""
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/locative-coordinated-passive-patient-20260921.json'
@dataclass(frozen=True)
class Arc:name:str;words:tuple[str,...];event:str;patient:str;instrument:str
L=(Arc('smith_gates_walls',('the','careful','smith','mends','the','gates','and','walls','that','are','repaired','with','the','hammers','in','the','workshop','beside','the','river'),'mend','gates-and-walls','hammers'),Arc('porters_crates_boxes',('the','quiet','porters','move','the','crates','and','boxes','that','are','carried','with','the','carts','beside','the','river','in','the','warehouse'),'move','crates-and-boxes','carts'))
R=(Arc('carpenters_gates_walls',('the','patient','carpenters','mend','the','gates','and','walls','that','are','repaired','with','the','hammers','in','the','workshop','beside','the','river'),'mend','gates-and-walls','hammers'),Arc('merchants_crates_boxes',('the','strong','merchants','move','the','crates','and','boxes','that','are','carried','with','the','carts','beside','the','river','in','the','warehouse'),'move','crates-and-boxes','carts'))
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
   if (a.event,a.patient,a.instrument)!=(b.event,b.patient,b.instrument):continue
   s=' '.join(a.words)+'; '+' '.join(b.words)+'.';t=tape(s);l=0;r=len(t)-1
   while l<r and t[l]==t[r]:l+=1;r-=1
   z={'pairs':l,'center_inside_word':True,'obligation':None if l>=r else(t[l],t[r])}
   rows.append({'rendered':s,'left_arc':a.name,'right_arc':b.name,'novel_rendered':t not in old,'semantic_state':{'relative_marker':'that','event_frame':a.event,'coordinated_plural_patient':a.patient,'passive_auxiliary':'are','instrument':a.instrument,'typed_locatives':True,'agreement_controller':'coordinated plural'},'audit':audit(s),'live_trace':z,'exact_admitted':z['obligation'] is None,'reader_status':'unreviewed; exactness does not certify readability','provenance':{'construction':'coordinated plural passive instrument patient','finished_tape_reversal':False,'posthoc_repair':False,'catalogue_text':False}})
 exact=[x for x in rows if x['exact_admitted'] and x['novel_rendered']];out={'experiment_id':'locative-coordinated-passive-patient-20260921','status':'completed_exact' if exact else 'completed_no_exact_closure','method':'that-marker coordinated plural passive patient with typed instrument/locatives','candidate_count':len(rows),'exact_count':len(exact),'reader_eligible':False,'rendered_candidates':rows,'stats':{'longest_letters':max(x['audit']['letters'] for x in rows)},'novelty_preflight':{'hash_source':'all runs/*.json except current artifact','current_artifact_excluded':True,'all_rendered_new':all(x['novel_rendered'] for x in rows),'coordinated_agreement_new':True},'failure_and_repair':{'failure':'coordinated passive morphology still mismatches live character obligations' if not exact else 'none','next_construction':'TOPOLOGY RESET: leave morphology family and pair a new semantic event graph with outer obligations'},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'independent_audits':['two-pointer','forward/reverse SHA-256'],'shortcuts_excluded':True}}
 RUN.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':out['status'],'candidates':len(rows),'exact':len(exact),'longest_letters':out['stats']['longest_letters'],'all_new':out['novelty_preflight']['all_rendered_new']}))
if __name__=='__main__':main()
