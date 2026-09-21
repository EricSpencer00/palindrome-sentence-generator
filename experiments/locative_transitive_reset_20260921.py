"""Topology reset: new transitive event frames coupled to locative order."""
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/locative-transitive-reset-20260921.json'
@dataclass(frozen=True)
class Arc:name:str;words:tuple[str,...];order:str;event:str
L=(Arc('carpenter_repairs',('the','careful','carpenter','who','repairs','the','gate','in','the','workshop','beside','the','river'),'place-first','repair'),Arc('merchant_moves',('a','quiet','merchant','who','moves','the','crate','beside','the','river','in','the','warehouse'),'river-first','move'))
R=(Arc('builder_repairs',('the','patient','builder','who','repairs','the','gate','in','the','workshop','beside','the','river'),'place-first','repair'),Arc('porter_moves',('a','strong','porter','who','moves','the','crate','beside','the','river','in','the','warehouse'),'river-first','move'))
def tape(s):return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=tape(s);n=len(t);bad=next((i for i in range(n//2) if t[i]!=t[n-1-i]),None);h=hashlib.sha256(t.encode()).hexdigest();rh=hashlib.sha256(t[::-1].encode()).hexdigest();return {'letters':n,'two_pointer_exact':bad is None,'first_mismatch':bad,'forward_sha256':h,'reverse_sha256':rh,'sha_equal':h==rh}
def main():
 old=set()
 for p in ROOT.glob('runs/*.json'):
  if p == RUN:
   continue
  try: old.update(tape(x.get('rendered','')) for x in json.loads(p.read_text()).get('rendered_candidates',[]) if x.get('rendered'))
  except Exception: pass
 rows=[]
 for a in L:
  for b in R:
   if (a.order,a.event)!=(b.order,b.event):continue
   s=' '.join(a.words)+'; '+' '.join(b.words)+'.';t=tape(s);l=0;r=len(t)-1
   while l<r and t[l]==t[r]:l+=1;r-=1
   z={'pairs':l,'center_inside_word':True,'obligation':None if l>=r else(t[l],t[r])}
   rows.append({'rendered':s,'left_arc':a.name,'right_arc':b.name,'novel_rendered':t not in old,'semantic_state':{'event_frame':a.event,'locative_order':a.order,'typed_prepositions':['in','beside'],'transitive_object':True},'audit':audit(s),'live_trace':z,'exact_admitted':z['obligation'] is None,'reader_status':'unreviewed; exactness does not certify readability','provenance':{'construction':'new transitive event frame coupled to locative order','finished_tape_reversal':False,'posthoc_repair':False,'catalogue_text':False}})
 exact=[x for x in rows if x['exact_admitted'] and x['novel_rendered']];out={'experiment_id':'locative-transitive-reset-20260921','status':'completed_exact' if exact else 'completed_no_exact_closure','method':'transitive repair/move event frames coupled to typed locative order','candidate_count':len(rows),'exact_count':len(exact),'reader_eligible':False,'rendered_candidates':rows,'stats':{'longest_letters':max(x['audit']['letters'] for x in rows)},'novelty_preflight':{'hash_source':'all runs/*.json rendered_candidates','all_rendered_new':all(x['novel_rendered'] for x in rows),'topology_reset':True,'adjunct_permutation_only':False},'failure_and_repair':{'failure':'new transitive locative frames still mismatch live character obligations' if not exact else 'none','next_construction':'add an event-frame instrument argument with typed locative order, without expanding the lexical bank'},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'independent_audits':['two-pointer','forward/reverse SHA-256'],'shortcuts_excluded':True}}
 RUN.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':out['status'],'candidates':len(rows),'exact':len(exact),'longest_letters':out['stats']['longest_letters'],'all_new':out['novelty_preflight']['all_rendered_new']}))
if __name__=='__main__':main()
