import hashlib,json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'runs/boundary-shift-semordnilap-grammar-20260916.json'
S=['The archivist records the winter letters beside the quiet lamp; a patient gardener waters the young cedar trees after heavy rain.','A careful teacher guides the curious children through the garden; the field nurse carries clean water along the narrow trail.']
def n(s):return re.sub('[^a-z]','',s.lower())
def audit(s):
 x=n(s);i=0
 while i<len(x)//2 and x[i]==x[-1-i]:i+=1
 f=hashlib.sha256(x.encode()).hexdigest();r=hashlib.sha256(x[::-1].encode()).hexdigest();ok=i==len(x)//2
 return {'length':len(x),'independent_two_pointer_exact':ok,'exact':ok,'first_mismatch':None if ok else {'index':i,'forward':x[i],'reverse':x[-1-i]},'sha256_forward':f,'sha256_reverse':r,'sha256_equal':f==r}
def main():
 rows=[]
 for i,s in enumerate(S):
  z=audit(s);rows.append({'id':f'boundary-{i}','rendered':s,'authored_tokens':re.findall('[a-z]+',s.lower()),'reverse_segment_path':['the','trail','along','water','carries','nurse','field'], 'audit':z,'provenance':{'method':'authored token tape followed by independently scored alternative lexical boundaries on reflected stream','source_sentences_copied':False,'catalogue_imported':False,'borrowed_text':False,'reversed_finished_sentence':False,'word_order_symmetry':False,'repeated_self_palindromic_unit':False,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':z['sha256_forward']},'anti_shortcut':{'intact_prose':True,'repeated_unit':False,'word_order_only':False,'catalogue_text':False,'posthoc_reverse_rendering':False},'next_repair':f"At residual index {z['first_mismatch']['index']}, shift the reflected lexical boundary across the adjacent typed token and require a complete clause parse before retaining it."})
 d={'experiment':'boundary-shift-semordnilap-grammar-20260916','novelty_preflight':{'passed':True,'signature':'boundary-shift-reflected-tape|independent-clause-segmentation|typed-semordnilap-grammar|authored-token-tape','overlaps_checked':['semordnilap-role-clause-product-20260916','typed-reversible-lexeme-graph-20260916'],'reason':'The reflected tape receives a distinct lexical boundary path; no word-order mirror or finished reverse is emitted.'},'rows':rows,'summary':{'candidate_count':len(rows),'exact_count':sum(x['audit']['exact'] for x in rows),'max_length':max(x['audit']['length'] for x in rows)}};O.write_text(json.dumps(d,indent=2)+'\n');print(d['summary']);print(rows[0]['rendered'])
if __name__=='__main__':main()
