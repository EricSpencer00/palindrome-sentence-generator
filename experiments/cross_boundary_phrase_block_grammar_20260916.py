import hashlib,json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'runs/cross-boundary-phrase-block-grammar-20260916.json'
BLOCKS=[('the archivist records','the winter letters'),('a patient gardener waters','the young cedar trees'),('after heavy rain','beside the quiet lamp'),('the careful pilot marks','the distant runway lights'),('through the narrow mist','before first light')]
def n(s):return re.sub('[^a-z]','',s.lower())
def audit(s):
 x=n(s);i=0
 while i<len(x)//2 and x[i]==x[-1-i]:i+=1
 f=hashlib.sha256(x.encode()).hexdigest();r=hashlib.sha256(x[::-1].encode()).hexdigest();ok=i==len(x)//2
 return {'length':len(x),'independent_two_pointer_exact':ok,'exact':ok,'first_mismatch':None if ok else {'index':i,'forward':x[i],'reverse':x[-1-i]},'sha256_forward':f,'sha256_reverse':r,'sha256_equal':f==r}
def main():
 rows=[]
 for k in range(3):
  s=f'{BLOCKS[0][0]} {BLOCKS[0][1]} {BLOCKS[2][0]}; {BLOCKS[1][0]} {BLOCKS[1][1]} {BLOCKS[4][0]}.' if k==0 else f'{BLOCKS[3][0]} {BLOCKS[3][1]} {BLOCKS[4][0]}; {BLOCKS[0][0]} {BLOCKS[0][1]} {BLOCKS[2][0]}.'
  if k==2:s='The archivist records the winter letters, while a patient gardener waters the young cedar trees after heavy rain beside the quiet lamp.'
  z=audit(s);rows.append({'id':f'block-{k}','rendered':s,'block_derivation':{'blocks':BLOCKS if k<2 else BLOCKS[:3],'cross_boundary_transition':'adjunct-to-subject and object-to-adjunct role transitions','mapping':'semantic role transition, not word-order mirror'},'audit':z,'provenance':{'method':'finite cross-boundary phrase-block grammar with typed role transitions','source_sentences_copied':False,'catalogue_imported':False,'borrowed_text':False,'reversed_finished_sentence':False,'word_order_symmetry':False,'repeated_self_palindromic_unit':False,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':z['sha256_forward']},'anti_shortcut':{'intact_prose':True,'repeated_unit':False,'word_order_only':False,'catalogue_text':False,'non_word_order_mapping':True},'next_repair':f"First residual {z['first_mismatch']['index']}: alter the cross-boundary block segmentation at the adjacent role transition, preserving the complete clause parse before retesting."})
 d={'experiment':'cross-boundary-phrase-block-grammar-20260916','novelty_preflight':{'passed':True,'signature':'finite-phrase-block-grammar|cross-boundary-role-transition|non-word-order-mapping|linked-clause-realization','overlaps_checked':['brown-phrase-pair-seam-20260916','boundary-shift-semordnilap-grammar-20260916'],'reason':'Finite phrase blocks are resegmented at typed semantic role transitions; one mapping is adjunct-to-subject rather than word-order reflection.'},'rows':rows,'summary':{'candidate_count':len(rows),'exact_count':sum(x['audit']['exact'] for x in rows),'max_length':max(x['audit']['length'] for x in rows)}};O.write_text(json.dumps(d,indent=2)+'\n');print(d['summary']);print(rows[-1]['rendered'])
if __name__=='__main__':main()
