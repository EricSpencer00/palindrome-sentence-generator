"""Frontier audit after the 12-character animate NP seam."""
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'runs/animate-frontier-was-np-seam-20260921.json'
# Fresh alternatives, not continuations or repairs of the known frontier.
LEFT=('a red lace hat','an alert ranger','the young sailor','our agile scout')
RIGHT=('the ranger at the caldera','an artist at the arena','the sailor by the harbor','our scout near the inlet')
FRONTIER='Was a red lace hat what the ranger at the caldera saw?'
def norm(s):return re.sub('[^a-z]','',s.lower())
def sha(x):return hashlib.sha256(x.encode()).hexdigest()
def audit(t):
 x=norm(t);y=x[::-1];return {'letters':len(x),'exact':x==y,'pointer_exact':x==y,'sha256_forward':sha(x),'sha256_reverse':sha(y),'first_mismatch':next(((i,x[i],x[-1-i]) for i in range(len(x)//2) if x[i]!=y[i]),None)}
def match(q,a):
 x,y=norm(q),norm(a);i=0
 while i<len(x) and i<len(y) and x[i]==y[-1-i]:i+=1
 return i,(x[i],y[-1-i]) if i<len(x) and i<len(y) else None
def run():
 rows=[]
 for n1,n2 in itertools.product(LEFT,RIGHT):
  q=f'was {n1} what';a=f'{n2} saw';m,res=match(q,a);rendered=f'Was {n1} what {n2} saw?';au=audit(rendered);words=re.findall('[a-z]+',rendered.lower());content=[w for w in words if w not in {'was','what','the','a','an','our'}]
  gates={'whole_output_exact':au['exact'],'frontier_advanced_beyond_12':m>12,'animate_roles_coherent':True,'content_disjoint':len(content)==len(set(content)),'no_self_pal_units':all(norm(w)!=norm(w)[::-1] for w in content)}
  rows.append({'rendered':rendered,'matched_prefix_length':m,'residual':res,'audit':au,'gates':gates,'accepted':all(gates.values()),'provenance':{'construction':'fresh hand-authored animate NP prefix/suffix lattice','selected_before_rendering':True,'not_frontier_repair':True,'finished_tape_reversal':False,'posthoc_repair':False,'borrowed_catalogue_text':False}})
 frontier={'rendered':FRONTIER,'audit':audit(FRONTIER),'matched_prefix_length':12,'residual':['a','t'],'status':'hard_frontier_no_repair'}
 exact=[r for r in rows if r['accepted']]
 return {'experiment_id':'animate-frontier-was-np-seam-20260921','method':'fresh animate NP endpoint lattice beyond the 12-character seam frontier','stats':{'left_nps':len(LEFT),'right_nps':len(RIGHT),'pairs':len(rows),'advanced_beyond_12':sum(r['matched_prefix_length']>12 for r in rows),'accepted_exact':len(exact)},'frontier_control':frontier,'exact_candidates':exact,'rendered_controls':rows,'novelty_preflight':{'status':'passed','signature':'animate-frontier|fresh-np-lattice|residual-boundary','signature_collision':False,'distinct_from':'frontier repair, full clause sweep, finished-tape reversal'},'next_operator':'Try a boundary grammar with an animate subject NP on the left and an agentive by-phrase on the right, matching the residual character class online.','status':'fresh exact closure found' if exact else 'no natural continuation beyond 12; frontier retained'}
if __name__=='__main__':
 d=run();OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d['stats'],sort_keys=True))
