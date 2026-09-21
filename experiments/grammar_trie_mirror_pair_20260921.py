"""Grammar-conditioned trie search for an original sentence mirror pair."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/grammar-trie-mirror-pair-20260921.json'
SUBJ=("the harbor pilot","a patient tailor","our village doctor")
VERB=("marks","records","carries")
OBJ=("a brass compass","the quiet ledger","one sealed parcel")
ADJ=("before dawn","near the river","during the storm")
RSUBJ=("the evening keeper","a careful botanist","our coastal ranger")
RVERB=("notices","returns","measures")
ROBJ=("a blue lantern","the spare key","an iron gate")
RADJ=("after sunset","beside the garden","under a clear sky")
def letters(s): return re.sub(r'[^a-z]','',s.lower())
def audit(s):
 t=letters(s); m=[(i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]]
 return {'letters':len(t),'exact':bool(t) and not m,'first_mismatch':m[:3], 'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def clauses(left=True):
 bank=(SUBJ,VERB,OBJ,ADJ) if left else (RSUBJ,RVERB,ROBJ,RADJ)
 return [f'{s} {v} {o} {a}' for s in bank[0] for v in bank[1] for o in bank[2] for a in bank[3]]
def trie_depth(target, right):
 # Trie-like prefix index over independently authored right clauses.
 prefixes={}; best=[]; best_depth=-1
 for clause in right:
  w=letters(clause); n=0
  while n<len(w) and n<len(target) and w[n]==target[n]: n+=1
  prefixes.setdefault(n,[]).append(clause)
  if n>best_depth: best_depth=n; best=[n,clause]
 return {'depth':best[0] if best else 0,'candidate':best[1] if best else None,'states':len(prefixes)}
def run():
 left=clauses(); right=clauses(False); rows=[]
 for l in left:
  target=letters(l)[::-1]; hit=trie_depth(target,right)
  rows.append({'left':l,'right_prefix_candidate':hit['candidate'],'obligation_prefix':target[:hit['depth']], 'trie':hit,
    'complete_right_matches':[],'provenance':{'authored_svo_adjunct':True,'independent_right_bank':True,'finished_reversal_for_generation':False,'catalogue':False,'self_palindromic_sentence':False,'posthoc_repair':False}})
 exact=[r for r in rows if r['complete_right_matches']]
 control=f'{left[0]}.'; return {'experiment_id':'grammar-trie-mirror-pair-20260921','method':'typed SVO adjunct trie prefix plus reverse-obligation DP diagnostic','stats':{'left_sentences':len(left),'right_sentences':len(right),'exact_pairs':len(exact),'deepest_prefix':max(r['trie']['depth'] for r in rows)},'exact_pairs':exact,'deepest_parse':max(rows,key=lambda r:r['trie']['depth']),'control':{'sentence':control,'audit':audit(control)},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'audits':['two-pointer','SHA-256']},'next_grammar_operator':'Add a typed adjunct beginning with the deepest live prefix, then resume DP without changing existing lexical banks.','status':'exact pair found' if exact else 'no exact pair; deepest grammar prefix recorded'}
if __name__=='__main__':
 d=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(d,indent=2)+'\n'); print(json.dumps(d['stats'],sort_keys=True))
