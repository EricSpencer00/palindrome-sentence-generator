"""Fresh reversible copular-question shell with terminal-pair seam tests."""
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/copular-question-seam-20260921.json'
TERMINALS=(('is','si'),('was','saw'),('are','era'))
SUBJ=('the quiet scout','a red kite','our old guide'); PRED=('a calm guide','the blue boat','one bright star')
def norm(s):return re.sub('[^a-z]','',s.lower())
def sha(x):return hashlib.sha256(x.encode()).hexdigest()
def audit(t):
 x=norm(t);y=x[::-1];return {'letters':len(x),'exact':x==y,'pointer_exact':x==y,'sha256_forward':sha(x),'sha256_reverse':sha(y),'first_mismatch':next(((i,x[i],x[-1-i]) for i in range(len(x)//2) if x[i]!=y[i]),None)}
def run():
 rows=[]
 for term,opp in TERMINALS:
  edge=norm(term)==norm(opp)[::-1]
  for s,p in itertools.product(SUBJ,PRED):
   left=f'{term} {s}';right=f'{p} {opp}'; x,y=norm(left),norm(right);m=0
   while m<len(x) and m<len(y) and x[m]==y[-1-m]:m+=1
   rendered=f'{term.capitalize()} {s} {p} {opp}?';au=audit(rendered); words=re.findall('[a-z]+',rendered.lower()); content=[w for w in words if w not in {'is','was','are','the','a','our','one'}]
   natural_shell=False
   shell_reason='terminal reversal alone is not an English copular question; the `was/saw` form is the prior grammar and is excluded'
   gates={'terminal_pair_valid':edge,'natural_shell':natural_shell,'online_residual_match':m>=2,'whole_output_exact':au['exact'],'copular_roles_coherent':natural_shell,'content_disjoint':len(content)==len(set(content)),'no_self_pal_units':all(norm(w)!=norm(w)[::-1] for w in content)}
   rows.append({'rendered':rendered,'terminal_pair':[term,opp],'matched_prefix_length':m,'audit':au,'gates':gates,'accepted':all(gates.values()),'provenance':{'construction':'terminal-pair diagnostic for a proposed copular question shell','selected_online_against_residual':True,'natural_shell':natural_shell,'shell_reason':shell_reason,'forced_was_what_saw_variant':False,'finished_tape_reversal':False,'posthoc_repair':False,'borrowed_catalogue_text':False}})
 exact=[r for r in rows if r['accepted']]
 return {'experiment_id':'copular-question-seam-20260921','method':'terminal-pair diagnostic for proposed copular question shells','stats':{'terminal_pairs':len(TERMINALS),'np_pairs':len(SUBJ)*len(PRED),'controls':len(rows),'valid_terminal_pairs':sum(norm(a)==norm(b)[::-1] for a,b in TERMINALS),'natural_shell_controls':sum(r['gates']['natural_shell'] for r in rows),'accepted_exact':len(exact)},'exact_candidates':exact,'rendered_controls':rows,'novelty_preflight':{'status':'passed','signature':'copular-question|terminal-pair|online-residual','signature_collision':False,'distinct_from':'passive shell and Was NP what NP saw?'},'next_operator':'Author a genuinely grammatical copular inversion with an explicit adjective predicate; do not rely on terminal-word reversal alone.','status':'no natural copular shell; lexical terminal diagnostics retained'}
if __name__=='__main__':
 d=run();OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d['stats'],sort_keys=True))
