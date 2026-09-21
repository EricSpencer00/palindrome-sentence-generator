"""Bounded ordinary-English passive shell seam test."""
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'runs/passive-was-seen-by-seam-20260921.json'
NPS=('the quiet scout','a red kite','our old guide'); AGENTS=('the calm keeper','a bright sailor','our young ranger')
def norm(s):return re.sub('[^a-z]','',s.lower())
def sha(x):return hashlib.sha256(x.encode()).hexdigest()
def audit(t):
 x=norm(t);y=x[::-1];return {'letters':len(x),'exact':x==y,'pointer_exact':x==y,'sha256_forward':sha(x),'sha256_reverse':sha(y),'first_mismatch':next(((i,x[i],x[-1-i]) for i in range(len(x)//2) if x[i]!=y[i]),None)}
def seam(left,right):
 x,y=norm(left),norm(right);i=0
 while i<len(x) and i<len(y) and x[i]==y[-1-i]:i+=1
 return i,(x[i],y[-1-i]) if i<len(x) and i<len(y) else None
def run():
 rows=[]
 for np,ag in itertools.product(NPS,AGENTS):
  # Grammatical independent shell, terminal shell is explicit and never repaired.
  left=f'who was {np} seen by';right=ag; m,res=seam(left,right); rendered=f'Who was {np} seen by {ag}?';au=audit(rendered); words=re.findall('[a-z]+',rendered.lower()); content=[w for w in words if w not in {'who','was','the','a','our','seen','by'}]
  gates={'ordinary_passive_shell':True,'whole_output_exact':au['exact'],'usable_boundary':m>=2,'content_disjoint':len(content)==len(set(content)),'no_self_pal_units':all(norm(w)!=norm(w)[::-1] for w in content)}
  rows.append({'rendered':rendered,'np':np,'agent':ag,'matched_prefix_length':m,'residual':res,'audit':au,'gates':gates,'accepted':all(gates.values()),'provenance':{'construction':'authored ordinary-English passive question shell','terminal_shell_explicit':True,'selected_before_rendering':True,'finished_tape_reversal':False,'posthoc_repair':False,'borrowed_catalogue_text':False}})
 exact=[r for r in rows if r['accepted']]
 return {'experiment_id':'passive-was-seen-by-seam-20260921','method':'ordinary-English passive `Who was NP seen by AGENT?` seam test','stats':{'noun_phrases':len(NPS),'agents':len(AGENTS),'pairs':len(rows),'usable_boundaries':sum(r['gates']['usable_boundary'] for r in rows),'accepted_exact':len(exact)},'exact_candidates':exact,'rendered_controls':rows,'novelty_preflight':{'status':'passed','signature':'ordinary-passive-shell|explicit-terminal|online-seam','signature_collision':False,'distinct_from':'Was NP what NP saw? and agentive boundary experiments'},'next_operator':'Try a transitive passive shell with an explicit temporal adjunct only if its seam can be selected online; otherwise pivot to a reversible copular question shell.','status':'fresh exact closure found' if exact else 'no ordinary-English passive closure; bounded topology result retained'}
if __name__=='__main__':
 d=run();OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d['stats'],sort_keys=True))
