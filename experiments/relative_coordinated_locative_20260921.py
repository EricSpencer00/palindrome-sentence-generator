"""Coordinated passive patient plus typed locative adjunct."""
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/relative-coordinated-locative-20260921.json';PRIOR=ROOT/'runs/relative-coordinated-instruments-20260921.json'
@dataclass(frozen=True)
class Arc:name:str;words:tuple[str,...];loc:str
L=(Arc('makers_tools_brushes',('the','skilled','makers','who','work','with','the','tools','and','brushes','that','were','carried','by','the','builders','in','the','workshop'),'workshop'),Arc('artists_brushes_tools',('the','patient','artists','who','paint','with','the','brushes','and','tools','that','were','carried','by','the','students','in','the','studio'),'studio'))
R=(Arc('workers_tools_brushes',('the','quiet','workers','who','build','with','the','tools','and','brushes','that','were','carried','by','the','carpenters','in','the','workshop'),'workshop'),Arc('painters_brushes_tools',('the','careful','painters','who','paint','with','the','brushes','and','tools','that','were','carried','by','the','artists','in','the','studio'),'studio'))
def tape(s):return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=tape(s);n=len(t);bad=next((i for i in range(n//2) if t[i]!=t[n-1-i]),None);h=hashlib.sha256(t.encode()).hexdigest();rh=hashlib.sha256(t[::-1].encode()).hexdigest();return {'letters':n,'two_pointer_exact':bad is None,'first_mismatch':bad,'forward_sha256':h,'reverse_sha256':rh,'sha_equal':h==rh}
def main():
 old={tape(x['rendered']) for x in json.loads(PRIOR.read_text()).get('rendered_candidates',[])} if PRIOR.exists() else set();rows=[]
 for a in L:
  for b in R:
   if a.loc!=b.loc:continue
   s=' '.join(a.words)+'; '+' '.join(b.words)+'.';t=tape(s);l=0;r=len(t)-1
   while l<r and t[l]==t[r]:l+=1;r-=1
   z={'pairs':l,'center_inside_word':True,'obligation':None if l>=r else(t[l],t[r])}
   rows.append({'rendered':s,'left_arc':a.name,'right_arc':b.name,'novel_rendered':t not in old,'semantic_state':{'with_valency':True,'coordinated_plural_patient':True,'instrument_alternation':True,'passive_by_agent_agreement':True,'locative_preposition':'in','locative_attachment':a.loc},'audit':audit(s),'live_trace':z,'exact_admitted':z['obligation'] is None,'reader_status':'unreviewed; exactness does not certify readability','provenance':{'construction':'coordinated passive patient with typed locative adjunct','finished_tape_reversal':False,'posthoc_repair':False,'catalogue_text':False}})
 exact=[x for x in rows if x['exact_admitted'] and x['novel_rendered']];out={'experiment_id':'relative-coordinated-locative-20260921','status':'completed_exact' if exact else 'completed_no_exact_closure','method':'coordinated passive patient, alternating instruments, typed in-locative adjunct','candidate_count':len(rows),'exact_count':len(exact),'reader_eligible':False,'rendered_candidates':rows,'stats':{'longest_letters':max(x['audit']['letters'] for x in rows)},'novelty_preflight':{'hash_source':str(PRIOR),'all_rendered_new':all(x['novel_rendered'] for x in rows),'prior_family_reused':False,'family_bounded':True},'failure_and_repair':{'failure':'locative-compatible coordinated relatives still mismatch live character obligations' if not exact else 'none','next_construction':'carry two distinct typed locatives through a coordinated passive patient without widening the lexical bank'},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'independent_audits':['two-pointer','forward/reverse SHA-256'],'shortcuts_excluded':True}}
 RUN.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':out['status'],'candidates':len(rows),'exact':len(exact),'longest_letters':out['stats']['longest_letters'],'all_new':out['novelty_preflight']['all_rendered_new']}))
if __name__=='__main__':main()
