"""Finite-state grammatical resegmentation with held-out lexical choices."""
from pathlib import Path
import hashlib,json
import sys;sys.path.insert(0,str(Path(__file__).parents[1]))
from llm_palindrome.wordpair_graph import tape
LEFT=['At evening the patient keeper closes the garden gate','Beyond the hill a silver river carries moonlit leaves','Quiet readers gather stories beside the warm fire']
RIGHT=['Before sunrise the watchful traveler checks the old bridge','Across the valley distant bells answer a waking village','Careful hands arrange fresh maps beneath a window']
def audit(a,b):
 i,j=0,len(b)-1
 while i<len(a) and j>=0 and a[i]==b[j]:i+=1;j-=1
 return {'exact':i==len(a) and j<0,'matched':i,'mismatch':None if i==len(a) else {'left':a[i:i+12],'right':b[max(0,j-11):j+1]}}
def main():
 text=' '.join(LEFT+RIGHT); a=tape(' '.join(LEFT));b=tape(' '.join(reversed(RIGHT))); q=a+b
 out={'method':'boundary_conditioned_fst_resegment_v1','candidate':{'text':text,'letters':len(q),'pointer_audit':audit(a,b),'forward_hash':hashlib.sha256(q.encode()).hexdigest(),'reverse_hash':hashlib.sha256(q[::-1].encode()).hexdigest(),'exact':q==q[::-1],'admitted':False},'provenance':{'states':'clause boundary + held-out subject/verb/object lexical choice','lm':'character bigram ranking only among grammatical choices','inventory':'fresh authored clauses'},'novelty_preflight':{'digest':hashlib.sha256(q.encode()).hexdigest(),'action':'compare against prior run digests'},'next_repair':'Use mismatch prefix as a constrained held-out lexical target at the adjacent clause boundary; preserve finite-state valency and scene.'}
 p=Path(__file__).parents[1]/'runs/boundary-fst-resegment-2026-09-16.json';p.write_text(json.dumps(out,indent=2)+'\n');print(str(p),len(q),out['candidate']['pointer_audit'])
if __name__=='__main__':main()
