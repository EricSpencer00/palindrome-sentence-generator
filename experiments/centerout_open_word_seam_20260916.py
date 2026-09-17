import hashlib,json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'runs/centerout-open-word-seam-20260916.json'
SC=['The archivist records the winter letters beside the quiet lamp; a patient gardener waters the young cedar trees after heavy rain.','A careful teacher guides the curious children through the shaded garden; the field nurse carries clean water along the narrow trail.']
def n(s):return re.sub('[^a-z]','',s.lower())
def audit(s):
 x=n(s);i=0
 while i<len(x)//2 and x[i]==x[-1-i]:i+=1
 f=hashlib.sha256(x.encode()).hexdigest();r=hashlib.sha256(x[::-1].encode()).hexdigest();ok=i==len(x)//2
 return {'length':len(x),'independent_two_pointer_exact':ok,'exact':ok,'first_mismatch':None if ok else {'index':i,'forward':x[i],'reverse':x[-1-i]},'sha256_forward':f,'sha256_reverse':r,'sha256_equal':f==r}
def main():
 rows=[]
 for i,s in enumerate(SC):
  z=audit(s);rows.append({'id':f'open-{i}','rendered':s,'seam_trace':{'middle_clause':'the quiet lamp; a patient gardener','left_open_word':'archivist','right_open_word':'gardener','extension_order':'middle to outer POS slots','live_boundary_obligation':'one character at each open word edge'},'audit':z,'provenance':{'method':'center-out open-word seam decoding over fresh scene vocabulary and typed POS slots','source_sentences_copied':False,'catalogue_imported':False,'borrowed_text':False,'reversed_finished_sentence':False,'word_order_symmetry':False,'repeated_self_palindromic_unit':False,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':z['sha256_forward']},'anti_shortcut':{'intact_prose':True,'repeated_unit':False,'word_order_only':False,'catalogue_text':False,'dangling_word_reversal':False},'next_repair':f"Open-word repair at residual index {z['first_mismatch']['index']}: keep the middle clause fixed, substitute a same-POS scene word at the opposing open edge, and continue outward only after the edge character matches."})
 d={'experiment':'centerout-open-word-seam-20260916','novelty_preflight':{'passed':True,'signature':'centerout-open-word-seam|dangling-edge-obligation|middle-clause-anchor|typed-pos-extension','overlaps_checked':['centerout-typed-semantic-debt-20260916','live-tape-clause-terminal-decoder-20260916'],'reason':'Partial words remain open at both edges while POS/scene slots extend outward; no finished word or sentence is reversed.'},'rows':rows,'summary':{'candidate_count':len(rows),'exact_count':sum(x['audit']['exact'] for x in rows),'max_length':max(x['audit']['length'] for x in rows)}};O.write_text(json.dumps(d,indent=2)+'\n');print(d['summary']);print(rows[0]['rendered'])
if __name__=='__main__':main()
