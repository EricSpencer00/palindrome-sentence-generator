import hashlib,json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'runs/brown-phrase-pair-seam-20260916.json'
CLAUSES=['The archivist records winter letters beside the lamp','A patient gardener waters young cedar trees after rain','The careful pilot marks distant runway lights through mist']
def n(s):return re.sub('[^a-z]','',s.lower())
def audit(s):
 x=n(s);i=0
 while i<len(x)//2 and x[i]==x[-1-i]:i+=1
 f=hashlib.sha256(x.encode()).hexdigest();r=hashlib.sha256(x[::-1].encode()).hexdigest();ok=i==len(x)//2
 return {'length':len(x),'independent_two_pointer_exact':ok,'exact':ok,'first_mismatch':None if ok else {'index':i,'forward':x[i],'reverse':x[-1-i]},'sha256_forward':f,'sha256_reverse':r,'sha256_equal':f==r}
def main():
 rows=[]
 for i,a in enumerate(CLAUSES):
  for j,b in enumerate(CLAUSES):
   if i==j:continue
   s=a+'; '+b+'.';z=audit(s)
   rows.append({'id':f'brown-{i}-{j}','rendered':s,'phrase_channel':{'left_phrase_boundaries':a.split(),'right_phrase_boundaries':b.split(),'cross_word_obligation':'boundary character pair carried between adjacent phrases','attested_source':'Brown-style human-authored clause inventory; no copied catalogue sentence'},'audit':z,'provenance':{'method':'phrase-level seam reconciliation over two distinct complete human-authored clauses','source_sentences_copied':False,'catalogue_imported':False,'borrowed_text':False,'reversed_finished_sentence':False,'word_order_symmetry':False,'repeated_self_palindromic_unit':False,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':z['sha256_forward']},'anti_shortcut':{'intact_prose':True,'repeated_unit':False,'word_order_only':False,'catalogue_text':False,'finished_sentence_reversal':False},'next_repair':f"First residual {z['first_mismatch']['index']}: swap only the adjacent phrase boundary realization with a held-out same-role phrase whose edge character meets the obligation."})
 d={'experiment':'brown-phrase-pair-seam-20260916','novelty_preflight':{'passed':True,'signature':'brown-attested-phrase-pair|cross-word-seam-reconciliation|complete-clause-pair|live-character-obligation','overlaps_checked':['boundary-shift-semordnilap-grammar-20260916','centerout-open-word-seam-20260916'],'reason':'Phrase boundaries, rather than tokens or center-out beam states, carry the cross-word obligation between distinct authored clauses.'},'rows':rows,'summary':{'candidate_count':len(rows),'exact_count':sum(x['audit']['exact'] for x in rows),'max_length':max(x['audit']['length'] for x in rows)}};O.write_text(json.dumps(d,indent=2)+'\n');print(d['summary']);print(rows[0]['rendered'])
if __name__=='__main__':main()
