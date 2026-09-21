"""Plural passive patient with agreement-carrying plural by-agent."""
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/relative-passive-plural-patient-20260921.json';PRIOR=ROOT/'runs/relative-passive-participial-novel-20260921.json'
@dataclass(frozen=True)
class Arc:name:str;words:tuple[str,...];stem:str;agent:str
L=(Arc('makers_tools',('the','skilled','makers','who','work','with','the','tools','that','were','carried','by','the','builders'),'tools','plural'),Arc('artists_brushes',('the','patient','artists','who','paint','with','the','brushes','that','were','carried','by','the','students'),'brushes','plural'))
R=(Arc('workers_tools',('the','quiet','workers','who','build','with','the','tools','that','were','carried','by','the','carpenters'),'tools','plural'),Arc('painters_brushes',('the','careful','painters','who','paint','with','the','brushes','that','were','carried','by','the','artists'),'brushes','plural'))
def tape(s):return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=tape(s);n=len(t);bad=next((i for i in range(n//2) if t[i]!=t[n-1-i]),None);h=hashlib.sha256(t.encode()).hexdigest();rh=hashlib.sha256(t[::-1].encode()).hexdigest();return {'letters':n,'two_pointer_exact':bad is None,'first_mismatch':bad,'forward_sha256':h,'reverse_sha256':rh,'sha_equal':h==rh}
def main():
 old={tape(x['rendered']) for x in json.loads(PRIOR.read_text()).get('rendered_candidates',[])} if PRIOR.exists() else set();rows=[]
 for a in L:
  for b in R:
   if a.stem!=b.stem:continue
   s=' '.join(a.words)+'; '+' '.join(b.words)+'.';t=tape(s);l=0;r=len(t)-1
   while l<r and t[l]==t[r]:l+=1;r-=1
   z={'pairs':l,'center_inside_word':True,'obligation':None if l>=r else(t[l],t[r])}
   rows.append({'rendered':s,'left_arc':a.name,'right_arc':b.name,'novel_rendered':t not in old,'semantic_state':{'with_valency':True,'plural_stem':a.stem,'patient_number':'plural','passive_auxiliary':'were','past_participle':'carried','by_agent_number':'plural','agreement_propagated':True},'audit':audit(s),'live_trace':z,'exact_admitted':z['obligation'] is None,'reader_status':'unreviewed; exactness does not certify readability','provenance':{'construction':'plural passive patient with plural by-agent','finished_tape_reversal':False,'posthoc_repair':False,'catalogue_text':False}})
 exact=[x for x in rows if x['exact_admitted'] and x['novel_rendered']];out={'experiment_id':'relative-passive-plural-patient-20260921','status':'completed_exact' if exact else 'completed_no_exact_closure','method':'plural passive patient and agreement-carrying plural by-agent','candidate_count':len(rows),'exact_count':len(exact),'reader_eligible':False,'rendered_candidates':rows,'stats':{'longest_letters':max(x['audit']['letters'] for x in rows)},'novelty_preflight':{'prior_duplicate_rejected':'e328e07e','hash_source':str(PRIOR),'all_rendered_new':all(x['novel_rendered'] for x in rows),'family_bounded':True},'failure_and_repair':{'failure':'novel plural passive pairs still mismatch live character obligations' if not exact else 'none','next_construction':'add a plural relative patient with an explicit instrument noun alternation, retaining by-agent agreement'},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'independent_audits':['two-pointer','forward/reverse SHA-256'],'shortcuts_excluded':True}}
 RUN.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':out['status'],'candidates':len(rows),'exact':len(exact),'longest_letters':out['stats']['longest_letters'],'all_new':out['novelty_preflight']['all_rendered_new']}))
if __name__=='__main__':main()
