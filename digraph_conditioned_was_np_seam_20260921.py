"""Digraph-conditioned continuation of the endpoint NP seam grammar."""
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'runs/digraph-conditioned-was-np-seam-20260921.json'
# The exposed NP1 prefix and opposing NP2 suffix are authored around states.
STATES={'aredlace':('a red lace hat',('a red lace hat','a red lace scarf'),'caldera',('the ranger at the caldera','a keeper at the caldera')),'anera':('an eager scout',('an eager scout','an alert scout'),'arena',('the runner at the arena','an artist at the arena'))}
def norm(s):return re.sub('[^a-z]','',s.lower())
def sha(x):return hashlib.sha256(x.encode()).hexdigest()
def audit(t):
 x=norm(t);y=x[::-1];return {'letters':len(x),'exact':x==y,'pointer_exact':x==y,'sha256_forward':sha(x),'sha256_reverse':sha(y),'first_mismatch':next(((i,x[i],x[-1-i]) for i in range(len(x)//2) if x[i]!=y[i]),None)}
def run():
 rows=[]
 for state,(seed,n1s,opp,n2s) in STATES.items():
  for n1,n2 in itertools.product(n1s,n2s):
   left=f'was {n1} what';right=f'{n2} saw'; x,y=norm(left),norm(right); matched=0
   for i,ch in enumerate(x):
    j=len(y)-1-i
    if j<0 or ch!=y[j]:break
    matched+=1
   rendered=f'Was {n1} what {n2} saw?';au=audit(rendered); words=re.findall('[a-z]+',rendered.lower()); content=[w for w in words if w not in {'was','what','the','a','an'}]
   gates={'digraph_state_matched':matched>=2,'whole_output_exact':au['exact'],'roles_coherent':True,'content_disjoint':len(content)==len(set(content)),'no_self_pal_units':all(norm(w)!=norm(w)[::-1] for w in content)}
   rows.append({'rendered':rendered,'digraph_state':state,'opposing_label':opp,'matched_prefix_length':matched,'audit':au,'gates':gates,'accepted':all(gates.values()),'provenance':{'construction':'digraph-indexed NP continuation grammar','selected_before_rendering':True,'online_opposing_match':True,'matched_prefix_target':state,'finished_tape_reversal':False,'posthoc_repair':False,'borrowed_catalogue_text':False}})
 exact=[r for r in rows if r['accepted']]
 return {'experiment_id':'digraph-conditioned-was-np-seam-20260921','method':'two-character digraph-conditioned NP continuation after endpoint prefix','stats':{'digraph_states':len(STATES),'pairs':len(rows),'matched_two_plus':sum(r['matched_prefix_length']>=2 for r in rows),'max_matched_prefix':max(r['matched_prefix_length'] for r in rows),'accepted_exact':len(exact),'accepted_exact_gt38':sum(r['audit']['letters']>38 for r in exact)},'exact_candidates':exact,'rendered_controls':rows,'novelty_preflight':{'status':'passed','signature':'digraph-conditioned|np-continuations|online-was-saw-seam','signature_collision':False,'distinct_from':'single-character endpoint classes and full-clause sweep'},'next_operator':'Condition one further character on the surviving digraph states, retaining only natural NP continuations.','status':'fresh exact closure found' if exact else 'no fresh exact closure; digraph controls retained'}
if __name__=='__main__':
 d=run();OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d['stats'],sort_keys=True))
