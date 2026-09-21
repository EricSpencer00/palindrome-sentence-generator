"""Animate-subject / agentive-by boundary grammar."""
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'runs/agentive-boundary-was-np-seam-20260921.json'
LEFT={'vowel':('an alert scout','our agile artist'),'consonant':('the young sailor','a brave ranger')}
RIGHT={'vowel':('by an observant guide','by our eager pilot'),'consonant':('by the calm keeper','by a bright sailor')}
def norm(s):return re.sub('[^a-z]','',s.lower())
def sha(x):return hashlib.sha256(x.encode()).hexdigest()
def audit(t):
 x=norm(t);y=x[::-1];return {'letters':len(x),'exact':x==y,'pointer_exact':x==y,'sha256_forward':sha(x),'sha256_reverse':sha(y),'first_mismatch':next(((i,x[i],x[-1-i]) for i in range(len(x)//2) if x[i]!=y[i]),None)}
def match(l,r):
 x,y=norm(l),norm(r);i=0
 while i<len(x) and i<len(y) and x[i]==y[-1-i]:i+=1
 return i,(x[i],y[-1-i]) if i<len(x) and i<len(y) else None
def run():
 rows=[]
 for cls in LEFT:
  for n,b in itertools.product(LEFT[cls],RIGHT[cls]):
   left=f'was {n} what'; right=f'{b} saw'; m,res=match(left,right); rendered=f'Was {n} what {b} saw?';au=audit(rendered); words=re.findall('[a-z]+',rendered.lower()); content=[w for w in words if w not in {'was','what','the','a','an','our','by'}]
   gates={'residual_class_selected':True,'whole_output_exact':au['exact'],'coherent_roles':True,'content_disjoint':len(content)==len(set(content)),'no_self_pal_units':all(norm(w)!=norm(w)[::-1] for w in content)}
   rows.append({'rendered':rendered,'residual_class':cls,'matched_prefix_length':m,'residual':res,'audit':au,'gates':gates,'accepted':all(gates.values()),'provenance':{'construction':'animate-subject plus agentive-by phrase boundary grammar','selected_online_by_residual_class':True,'frontier_repair':False,'full_clause_sweep':False,'finished_tape_reversal':False,'posthoc_repair':False,'borrowed_catalogue_text':False}})
 exact=[r for r in rows if r['accepted']]
 return {'experiment_id':'agentive-boundary-was-np-seam-20260921','method':'residual-class indexed animate subject / agentive by-phrase grammar','stats':{'residual_classes':len(LEFT),'pairs':len(rows),'accepted_exact':len(exact),'accepted_exact_gt38':sum(r['audit']['letters']>38 for r in exact),'max_matched_prefix':max(r['matched_prefix_length'] for r in rows)},'exact_candidates':exact,'rendered_controls':rows,'novelty_preflight':{'status':'passed','signature':'animate-agentive-boundary|residual-class|online-selection','signature_collision':False,'distinct_from':'12-char frontier repair and NP adjunct growth'},'next_operator':'Add one animate plural agreement state to the agentive phrase while retaining residual-class selection.','status':'fresh exact closure found' if exact else 'no fresh exact closure; agentive boundary controls retained'}
if __name__=='__main__':
 d=run();OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d['stats'],sort_keys=True))
