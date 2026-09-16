"""Hand-authored complete-clause bank with cross-boundary tape alignment."""
from pathlib import Path
import hashlib,json,sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from llm_palindrome.wordpair_graph import WordPair,tape

BANK=[
 WordPair('Mara carries warm bread to the river','At the river, Owen listens for bells'),
 WordPair('The careful pilot studies cloud maps','Beyond the clouds, a quiet engine waits'),
 WordPair('Children gather bright shells by moonlight','At moonlight, the patient tide returns'),
 WordPair('A gardener shelters young cedar shoots','Near the cedar, small finches settle'),
 WordPair('Old friends share stories beside fire','By the fire, new stories begin'),
]
def ptr(a,b):
 i,j=0,len(b)-1
 while i<len(a) and j>=0 and a[i]==b[j]:i+=1;j-=1
 return {'exact':i==len(a) and j<0,'matched':i,'residual_left':a[i:],'residual_right':b[:j+1]}
def main():
 best=[]; closures=[]; calls=0
 def visit(path,l,r,used):
  nonlocal best,calls
  calls+=1
  if len(tape(l+' '+r))>len(tape(' '.join(x.left for x in best)+' '+' '.join(x.right for x in best))):best=path
  q=ptr(tape(l),tape(r))
  if q['exact']:closures.append(path[:])
  if len(path)==len(BANK):return
  for k,p in enumerate(BANK):
   if k in used or p.valency!='clause' or any(w in tape(l+' '+r) for w in tape(p.left).split() if len(w)>3):continue
   visit(path+[p],l+' '+p.left,p.right+' '+r,used|{k})
 for k,p in enumerate(BANK):visit([p],p.left,p.right,{k})
 text=' '.join(p.left for p in best)+' '+' '.join(p.right for p in reversed(best)); lt=tape(text)
 # Independently recompute pointer and digest.
 left=tape(' '.join(p.left for p in best));right=tape(' '.join(p.right for p in reversed(best))); audit=ptr(left,right)
 out={'method':'hand_authored_clause_cross_boundary_v1','candidate':{'text':text,'letters':len(lt),'exact':audit['exact'],'admitted':audit['exact'] and len(lt)>=100,'pointer_audit':audit,'hash':hashlib.sha256(lt.encode()).hexdigest()},'closures':len(closures),'expansions':calls,'provenance':{'bank':'five fresh complete clauses, semantic valency=clause','alignment':'cross-word residual character ledger; no word-order reversal'},'novelty_preflight':'digest emitted for comparison against prior run artifacts','next_repair':'Replace the residual suffix at the first mismatch with a semordnilap-compatible ordinary verb-object clause while preserving non-repeating content words.'}
 path=Path(__file__).parents[1]/'runs/hand-authored-clause-breakthrough-2026-09-16.json';path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'output':str(path),'letters':len(lt),'exact':audit['exact']}))
if __name__=='__main__':main()
