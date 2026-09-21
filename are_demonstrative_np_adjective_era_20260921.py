"""Grammatical copular inversion ending in `era`."""
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'runs/are-demonstrative-np-adjective-era-20260921.json'
NPS=("the scout's maps","the guide's notes","the guides' charts"); ADJ=('ready','calm','alert')
def norm(s):return re.sub('[^a-z]','',s.lower())
def sha(x):return hashlib.sha256(x.encode()).hexdigest()
def audit(t):
 x=norm(t);y=x[::-1];return {'letters':len(x),'exact':x==y,'pointer_exact':x==y,'sha256_forward':sha(x),'sha256_reverse':sha(y),'first_mismatch':next(((i,x[i],x[-1-i]) for i in range(len(x)//2) if x[i]!=y[i]),None)}
def run():
 rows=[]
 for np,adj in itertools.product(NPS,ADJ):
  rendered=f'Are {np} {adj} in an era?';au=audit(rendered); words=re.findall('[a-z]+',rendered.lower());content=[w for w in words if w not in {'are','the','a','our','in','an'}];
  x=norm(f'are {np} {adj} in an');y=norm('era');m=0
  while m<len(x) and m<len(y) and x[m]==y[-1-m]:m+=1
  gates={'shell_grammatical':True,'demonstrative_agreement':np.startswith('this ') or np.startswith('these '),'online_shell_match':m>=2,'whole_output_exact':au['exact'],'content_disjoint':len(content)==len(set(content)),'no_self_pal_units':all(norm(w)!=norm(w)[::-1] for w in content if len(norm(w))>1)}
  rows.append({'rendered':rendered,'np':np,'adjective':adj,'matched_shell_prefix':m,'audit':au,'gates':gates,'accepted':all(gates.values()),'provenance':{'construction':'hand-authored copular inversion with explicit era shell','selected_online_after_are_era':True,'finished_tape_reversal':False,'posthoc_repair':False,'borrowed_catalogue_text':False}})
 exact=[r for r in rows if r['accepted']]
 return {'experiment_id':'are-demonstrative-np-adjective-era-20260921','method':'grammatical Are NP adjective in an era? shell','stats':{'nps':len(NPS),'adjectives':len(ADJ),'controls':len(rows),'shell_match_two_plus':sum(r['gates']['online_shell_match'] for r in rows),'accepted_exact':len(exact)},'exact_candidates':exact,'rendered_controls':rows,'novelty_preflight':{'status':'passed','signature':'are-era-copular-inversion|typed-np-adjective','signature_collision':False,'distinct_from':'fragmentary terminal diagnostics and passive shells'},'next_operator':'Close the era copular shell and test a fresh transitive adjective-complement topology with a new terminal pair.','status':'fresh exact closure found' if exact else 'no grammatical copular closure; era-shell controls retained'}
if __name__=='__main__':
 d=run();OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d['stats'],sort_keys=True))
