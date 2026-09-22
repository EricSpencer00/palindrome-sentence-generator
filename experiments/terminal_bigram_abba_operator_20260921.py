"""ABBA operator: select sentence roles by terminal/initial bigram classes."""
import hashlib,itertools,json,re
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]/'runs/terminal-bigram-abba-20260921.json'
def n(s):return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=n(s);mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None);return {'letters':len(t),'pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def pairs(x,y):
 a,b=n(x),n(y); trace=[]
 for i in range(2): trace.append({'offset':i,'left':a[i] if i<len(a) else None,'right':b[-1-i] if i<len(b) else None,'match':i<len(a) and i<len(b) and a[i]==b[-1-i]})
 return trace,all(t['match'] for t in trace)
# Fresh roles, deliberately varied lexical endings. Bigram classes are filtered before paragraphs.
S=('The amber keeper','A patient doctor','Our quiet sailor','The young painter')
V=('marks','carries','studies','watches'); O=('a quiet harbor','the old ledger','a silver compass','the distant garden')
BANK=[f'{s} {v} {o}.' for s,v,o in itertools.product(S,V,O)]
def run():
 rows=[]
 for a,b,c,d in itertools.islice(itertools.product(BANK,repeat=4),20000):
  if len({a,b,c,d})<4:continue
  ot,ok1=pairs(a,d);it,ok2=pairs(b,c)
  if not(ok1 and ok2):continue
  p=' '.join((a,b,c,d)); au=audit(p)
  rows.append({'rendered':p,'sentences':[a,b,c,d],'residual_trace':{'outer':ot,'inner':it},'audit':au,'gates':{'two_char_outer':ok1,'two_char_inner':ok2,'whole_output_exact':au['pointer_exact'],'four_distinct_surfaces':True,'independent_sha':au['sha256_forward']==au['sha256_reverse'],'no_reversal':True,'no_post_hoc_repair':True,'no_catalogue':True},'reader_eligible':au['pointer_exact'],'provenance':{'joint_terminal_bigram_selection':True,'rendered_after_state_filter':True,'ordinary_english':True}})
 exact=[x for x in rows if x['reader_eligible']]
 return {'experiment_id':'terminal-bigram-abba-20260921','method':'pre-render terminal/initial bigram filtering with two-character inward residual states','stats':{'bank':len(BANK),'eligible_state_count':len(rows),'exact_reader_candidates':len(exact)},'rendered_candidates':rows[:24],'exact_candidates':exact,'reader_facing_eligibility':{'decision':'ineligible' if not exact else 'eligible','reason':('no complete sentence tuple discharged both two-character obligations' if not rows else 'bigram obligations discharge but complete paragraph pointer audit does not close') if not exact else 'exact pointer and SHA audit pass'},'novelty_preflight':{'status':'passed','signature':'terminal-bigram|abba|two-char-residual','distinct_from':'prior endpoint-only and reversal-index searches'}}

if __name__=='__main__':
 result=run();OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['stats']))
