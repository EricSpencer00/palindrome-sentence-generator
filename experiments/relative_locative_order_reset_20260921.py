"""Bounded adjunct-order topology test; resets if order is merely lexical repetition."""
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/relative-locative-order-reset-20260921.json';PRIOR=ROOT/'runs/relative-two-locatives-20260921.json'
@dataclass(frozen=True)
class Arc:name:str;words:tuple[str,...];order:str
L=(Arc('makers_place_first',('the','skilled','makers','who','work','with','the','tools','in','the','workshop','beside','the','river'),'place-first'),Arc('artists_river_first',('the','patient','artists','who','paint','with','the','brushes','beside','the','river','in','the','studio'),'river-first'))
R=(Arc('workers_place_first',('the','quiet','workers','who','build','with','the','tools','in','the','workshop','beside','the','river'),'place-first'),Arc('painters_river_first',('the','careful','painters','who','paint','with','the','brushes','beside','the','river','in','the','studio'),'river-first'))
def tape(s):return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=tape(s);n=len(t);bad=next((i for i in range(n//2) if t[i]!=t[n-1-i]),None);h=hashlib.sha256(t.encode()).hexdigest();rh=hashlib.sha256(t[::-1].encode()).hexdigest();return {'letters':n,'two_pointer_exact':bad is None,'first_mismatch':bad,'forward_sha256':h,'reverse_sha256':rh,'sha_equal':h==rh}
def main():
 old={tape(x['rendered']) for x in json.loads(PRIOR.read_text()).get('rendered_candidates',[])} if PRIOR.exists() else set();rows=[]
 for a in L:
  for b in R:
   if a.order!=b.order:continue
   s=' '.join(a.words)+'; '+' '.join(b.words)+'.';t=tape(s);l=0;r=len(t)-1
   while l<r and t[l]==t[r]:l+=1;r-=1
   z={'pairs':l,'center_inside_word':True,'obligation':None if l>=r else(t[l],t[r])}
   rows.append({'rendered':s,'left_arc':a.name,'right_arc':b.name,'novel_rendered':t not in old,'semantic_state':{'with_valency':True,'fixed_locative':'river','adjunct_order':a.order,'typed_prepositions':['in','beside'],'coordinated_patient':True},'audit':audit(s),'live_trace':z,'exact_admitted':z['obligation'] is None,'reader_status':'unreviewed; exactness does not certify readability','provenance':{'construction':'typed locative adjunct-order topology','finished_tape_reversal':False,'posthoc_repair':False,'catalogue_text':False}})
 exact=[x for x in rows if x['exact_admitted'] and x['novel_rendered']];out={'experiment_id':'relative-locative-order-reset-20260921','status':'completed_exact' if exact else 'completed_no_exact_closure','method':'fixed river locative with typed adjunct-order alternatives','candidate_count':len(rows),'exact_count':len(exact),'reader_eligible':False,'rendered_candidates':rows,'stats':{'longest_letters':max(x['audit']['letters'] for x in rows)},'novelty_preflight':{'hash_source':str(PRIOR),'all_rendered_new':all(x['novel_rendered'] for x in rows),'topology_reset_required':False,'family_bounded':True},'failure_and_repair':{'failure':'order-compatible locative pairs still mismatch live character obligations' if not exact else 'none','next_construction':'topology reset: couple locative order to a new transitive event frame rather than adding adjunct permutations'},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'independent_audits':['two-pointer','forward/reverse SHA-256'],'shortcuts_excluded':True}}
 RUN.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':out['status'],'candidates':len(rows),'exact':len(exact),'longest_letters':out['stats']['longest_letters'],'all_new':out['novelty_preflight']['all_rendered_new']}))
if __name__=='__main__':main()
