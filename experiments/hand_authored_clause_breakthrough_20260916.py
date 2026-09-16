"""Hand-authored complete-clause bank with cross-boundary tape alignment."""
from pathlib import Path
import hashlib,json,sys,re
sys.path.insert(0,str(Path(__file__).parents[1]))
from llm_palindrome.wordpair_graph import WordPair,tape
from llm_palindrome.admission import mechanical_admission_checks

BANK=[
 WordPair('Mara carries warm bread to the river','At the shore, Owen listens for bells'),
 WordPair('The careful pilot studies cloud maps','Beyond the hills, a quiet engine waits'),
 WordPair('Children gather bright shells by moonlight','At twilight, the patient tide returns'),
 WordPair('A gardener shelters young cedar shoots','Near the grove, small finches settle'),
 WordPair('Old friends share stories beside fire','By the hearth, new tales begin'),
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
   if k in used or p.valency!='clause': continue
   prior_words=set(re.findall(r"[a-z]+", (l+' '+r).lower()))
   pair_words=set(re.findall(r"[a-z]+", (p.left+' '+p.right).lower()))
   # Enforce the same all-different content-word policy used by admission;
   # repeated mirrored nouns are not a valid constructive shortcut.
   if prior_words & {w for w in pair_words if len(w)>3}: continue
   visit(path+[p],l+' '+p.left,p.right+' '+r,used|{k})
 for k,p in enumerate(BANK):visit([p],p.left,p.right,{k})
 # Keep every bank member as an intact, reader-facing clause.  Periods are
 # outside the tape, so this improves prose presentation without weakening
 # the letter-level audit.
 text='. '.join(p.left for p in best)+'. '+'. '.join(p.right for p in reversed(best))+'.'; lt=tape(text)
 # Independently recompute pointer and digest.
 left=tape(' '.join(p.left for p in best));right=tape(' '.join(p.right for p in reversed(best))); audit=ptr(left,right)
 checks=mechanical_admission_checks(text,min_letters=39,max_letters=1000)
 rendered_hash=hashlib.sha256(lt.encode()).hexdigest(); reverse_hash=hashlib.sha256(lt[::-1].encode()).hexdigest()
 out={'method':'hand_authored_clause_cross_boundary_v1','candidate':{'text':text,'letters':len(lt),'exact':audit['exact'],'admitted':all(checks.values()),'pointer_audit':audit,'hash':rendered_hash,'reverse_hash':reverse_hash,'hash_equal':rendered_hash==reverse_hash,'mechanical_checks':checks},'closures':len(closures),'expansions':calls,'provenance':{'bank':'five fresh complete clauses, semantic valency=clause','alignment':'cross-word residual character ledger; no word-order reversal','all_different_content_words':True},'novelty_preflight':'digest emitted for comparison against prior run artifacts','next_repair':'Replace the residual suffix at the first mismatch with a semordnilap-compatible ordinary verb-object clause while preserving non-repeating content words.'}
 path=Path(__file__).parents[1]/'runs/hand-authored-clause-breakthrough-2026-09-16.json';path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'output':str(path),'letters':len(lt),'exact':audit['exact']}))
if __name__=='__main__':main()
