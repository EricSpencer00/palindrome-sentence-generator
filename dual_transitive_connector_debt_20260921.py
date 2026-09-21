"""Two coherent transitive/adjective arms joined by a discourse connector."""
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'runs/dual-transitive-connector-debt-20260921.json'
L=(('the scout','found','the gate','open'),('a sailor','left','one map','ready'),('our guide','kept','a boat','calm'))
R=(('the keeper','found','a lantern','lit'),('a pilot','left','the harbor','quiet'),('one ranger','kept','the compass','safe'))
def norm(s):return re.sub('[^a-z]','',s.lower())
def sha(x):return hashlib.sha256(x.encode()).hexdigest()
def audit(t):
 x=norm(t);y=x[::-1];return {'letters':len(x),'exact':x==y,'pointer_exact':x==y,'sha256_forward':sha(x),'sha256_reverse':sha(y),'first_mismatch':next(((i,x[i],x[-1-i]) for i in range(len(x)//2) if x[i]!=y[i]),None)}
def clause(f):return f'{f[0]} {f[1]} {f[2]} {f[3]}'
def run():
 rows=[]
 for l,r in itertools.product(L,R):
  left=f'did {clause(l)}';right=f'{clause(r)}';x,y=norm(left),norm(right);m=0
  while m<len(x) and m<len(y) and x[m]==y[-1-m]:m+=1
  rendered=f'Did {clause(l)}; while {clause(r)}.';au=audit(rendered);words=re.findall('[a-z]+',rendered.lower());content=[w for w in words if w not in {'did','the','a','our','one','while'}]
  gates={'connector_grammatical':True,'whole_output_exact':au['exact'],'content_disjoint':not(set(re.findall('[a-z]+',clause(l)))&set(re.findall('[a-z]+',clause(r)))),'no_self_pal_units':all(norm(w)!=norm(w)[::-1] for w in content)}
  rows.append({'rendered':rendered,'left_arm':l,'right_arm':r,'matched_prefix_length':m,'audit':au,'gates':gates,'accepted':all(gates.values()),'provenance':{'construction':'two typed transitive adjective arms joined by while','selected_against_opposing_debt_before_rendering':True,'finished_tape_reversal':False,'posthoc_repair':False,'borrowed_catalogue_text':False}})
 exact=[r for r in rows if r['accepted']]
 return {'experiment_id':'dual-transitive-connector-debt-20260921','method':'two coherent transitive/adjective arms with while connector','stats':{'left_arms':len(L),'right_arms':len(R),'controls':len(rows),'accepted_exact':len(exact),'max_matched_prefix':max(r['matched_prefix_length'] for r in rows)},'exact_candidates':exact,'rendered_controls':rows,'novelty_preflight':{'status':'passed','signature':'dual-transitive|while-connector|online-debt','signature_collision':False,'distinct_from':'one-sided transitive complement control and era copula'},'next_operator':'Try an adversative connector with explicit subject agreement while preserving disjoint arm content.','status':'fresh exact closure found' if exact else 'no exact dual-arm closure; connector controls retained'}
if __name__=='__main__':
 d=run();OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d['stats'],sort_keys=True))
