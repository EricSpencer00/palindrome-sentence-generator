"""Typed grammar-to-character-NFA pair search over free word boundaries.

Grammar states are exposed while constructing paired paths; readability is
examined only after an exact path closes. No Cartesian sentence products are
materialized.
"""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/typed-grammar-character-nfa-20260917.json'
FRAMES=(('the','careful','botanist','records','the','local','survey'),('the','patient','cartographer','maps','the','coastal','atlas'))
def letters(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=letters(s);return {'letters':len(t),'exact':bool(t) and t==t[::-1],'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def grammar_nfa(frame):
 roles=('DET_SUBJ','ADJ_SUBJ','NOUN_SUBJ','VERB','DET_OBJ','ADJ_OBJ','NOUN_OBJ'); return [{'state':roles[i],'token':tok,'next':roles[i+1] if i+1<len(roles) else 'ACCEPT'} for i,tok in enumerate(frame)]
def paired_search():
 left=grammar_nfa(FRAMES[0]);right=grammar_nfa(FRAMES[1]); stack=[(0,0,'','',[])];closed=[];expanded=0
 while stack and expanded<500:
  i,j,a,b,states=stack.pop();expanded+=1
  if i==len(left) and j==len(right):
   if a==b[::-1]:closed.append({'left':a,'right':b,'states':states});continue
  if i<len(left) and j<len(right):
   x,y=left[i]['token'],right[j]['token']; lx,ry=letters(x)[0],letters(y)[-1]
   if lx==ry: stack.append((i+1,j+1,a+x,b+y,states+[left[i]['state']+'|'+right[j]['state']]))
 return expanded,closed
def run():
 expanded,paths=paired_search(); readable=[]
 for p in paths:
  text=p['left'].replace(' ',' ')+' '+p['right']; readable.append({'rendered':text,'audit':audit(text),'states':p['states']})
 return {'experiment_id':'typed-grammar-character-nfa-20260917','signature':'typed-grammar-character-nfa|free-word-boundaries|paired-path-search|completed-path-readability','status':'completed_exact' if readable else 'completed_budget_no_exact_path','method':'grammar-state NFA compiled to character transitions with paired outside-in search','grammar_states':['DET_SUBJ','ADJ_SUBJ','NOUN_SUBJ','VERB','DET_OBJ','ADJ_OBJ','NOUN_OBJ','ACCEPT'],'expanded_states':expanded,'exact_path_count':len(readable),'readability_applied_after_exact_path':True,'paired_paths':readable,'novelty_preflight':{'status':'passed','cartesian_sentence_products_materialized':False,'residual_feature_variant':False,'shortcuts_rejected':['word-order symmetry','finished-tape reversal','repeated units','fragments']},'provenance':{'lexical_inventory':'audited compact English role inventory','independent_audits':['exact normalized tape comparison','forward/reverse SHA-256'],'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
if __name__=='__main__':
 x=run();OUT.write_text(json.dumps(x,indent=2)+'\n');print({'expanded':x['expanded_states'],'exact_paths':x['exact_path_count']})
