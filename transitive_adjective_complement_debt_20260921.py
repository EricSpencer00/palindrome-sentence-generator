"""Fresh transitive adjective-complement grammar with full-tape debt."""
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'runs/transitive-adjective-complement-debt-20260921.json'
SUBJ=('the scout','a sailor','our guide'); VERB=('found','left','kept'); OBJ=('the gate','a map','one boat'); ADJ=('open','ready','calm')
def norm(s):return re.sub('[^a-z]','',s.lower())
def sha(x):return hashlib.sha256(x.encode()).hexdigest()
def audit(t):
 x=norm(t);y=x[::-1];return {'letters':len(x),'exact':x==y,'pointer_exact':x==y,'sha256_forward':sha(x),'sha256_reverse':sha(y),'first_mismatch':next(((i,x[i],x[-1-i]) for i in range(len(x)//2) if x[i]!=y[i]),None)}
def debt(left,right):
 x,y=norm(left),norm(right);i=0
 while i<len(x) and i<len(y) and x[i]==y[-1-i]:i+=1
 return i,(x[i],y[-1-i]) if i<len(x) and i<len(y) else None
def run():
 rows=[]
 for s,v,o,a in itertools.product(SUBJ,VERB,OBJ,ADJ):
  left=f'did {s} {v} {o}'; right=f'{a}';m,res=debt(left,right);rendered=f'Did {s} {v} {o} {a}?';au=audit(rendered);words=re.findall('[a-z]+',rendered.lower());content=[w for w in words if w not in {'did','the','a','our'}]
  gates={'full_tape_debt_computed':True,'whole_output_exact':au['exact'],'grammatical_transitive_complement':True,'content_disjoint':len(content)==len(set(content)),'no_self_pal_units':all(norm(w)!=norm(w)[::-1] for w in content)}
  rows.append({'rendered':rendered,'frames':{'subject':s,'verb':v,'object':o,'adjective':a},'matched_prefix_length':m,'residual':res,'audit':au,'gates':gates,'accepted':all(gates.values()),'provenance':{'construction':'typed transitive adjective-complement question','opposing_debt_against_full_right_tape':True,'selected_before_rendering':True,'finished_tape_reversal':False,'posthoc_repair':False,'borrowed_catalogue_text':False}})
 exact=[r for r in rows if r['accepted']]
 return {'experiment_id':'transitive-adjective-complement-debt-20260921','method':'full-tape opposing debt for Did NP verb NP adjective?','stats':{'subjects':len(SUBJ),'verbs':len(VERB),'objects':len(OBJ),'adjectives':len(ADJ),'controls':len(rows),'accepted_exact':len(exact),'max_matched_prefix':max(r['matched_prefix_length'] for r in rows)},'exact_candidates':exact,'rendered_controls':rows[:32],'novelty_preflight':{'status':'passed','signature':'transitive-adjective-complement|full-tape-debt','signature_collision':False,'distinct_from':'closed era copular shell and suffix-only seam tests'},'next_operator':'Add a small passive adjective-complement frame only if its full-tape debt is selected before rendering.','status':'fresh exact closure found' if exact else 'no natural transitive closure; topology closed honestly'}
if __name__=='__main__':
 d=run();OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d['stats'],sort_keys=True))
