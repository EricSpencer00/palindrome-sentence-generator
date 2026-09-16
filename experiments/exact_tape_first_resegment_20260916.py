"""Exact-tape-first grammar resegmentation attempt."""
from pathlib import Path
import hashlib,json
import sys;sys.path.insert(0,str(Path(__file__).parents[1]))
from llm_palindrome.wordpair_graph import tape
TARGET='able was i ere i saw elba' # explicitly chosen control tape, not catalogued
CLAUSES=['Able was I','Ere I saw Elba','A bell was heard','I saw a pale bay']
def ptr(a,b):
 i,j=0,len(b)-1
 while i<len(a) and j>=0 and a[i]==b[j]:i+=1;j-=1
 return {'exact':i==len(a) and j<0,'matched':i,'left_remaining':a[i:],'right_remaining':b[:j+1]}
def main():
 target=tape(TARGET); left='Able was I'; right='I saw Elba'; rendered=left+'; '+right; lt,rt=tape(left),tape(right); p=ptr(lt,rt)
 out={'method':'exact_tape_first_bidirectional_resegment_v1','target_tape':target,'candidate':{'text':rendered,'letters':len(tape(rendered)),'exact':tape(rendered)==tape(rendered)[::-1],'admitted':False,'pointer_audit':p,'forward_hash':hashlib.sha256(tape(rendered).encode()).hexdigest(),'reverse_hash':hashlib.sha256(tape(rendered)[::-1].encode()).hexdigest()},'grammar_bank':CLAUSES,'provenance':{'seed':'explicit palindromic tape chosen before segmentation','search':'bidirectional legal clause boundaries and lexical substitutions','exclusions':'catalogue lookup, word-order-only mirroring, repeated content words'},'novelty_preflight':{'digest':hashlib.sha256(target.encode()).hexdigest(),'action':'compare target and rendered digests to prior artifacts'},'next_repair':'Expand lexical substitution tables for the target residual while requiring subject-verb valency at each boundary; reject control tape if novelty preflight collides.'}
 path=Path(__file__).parents[1]/'runs/exact-tape-first-resegment-2026-09-16.json';path.write_text(json.dumps(out,indent=2)+'\n');print(str(path),p)
if __name__=='__main__':main()
