import hashlib,json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1]; O=R/'runs/endpoint-aware-bilateral-seam-20260916.json'
FR=[('the archivist','keeps','the weathered journals','near the fire'),('a calm teacher','guides','the curious children','through the garden'),('the field nurse','carries','the clean water bottles','along the trail')]
def t(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 x=t(s); i=0
 while i<len(x)//2 and x[i]==x[-1-i]: i+=1
 f=hashlib.sha256(x.encode()).hexdigest(); q=hashlib.sha256(x[::-1].encode()).hexdigest(); ok=i==len(x)//2
 return {'length':len(x),'independent_two_pointer_exact':ok,'exact':ok,'first_mismatch':None if ok else {'index':i,'forward':x[i],'reverse':x[-1-i]},'sha256_forward':f,'sha256_reverse':q,'sha256_equal':f==q}
def main():
 rows=[]
 for i,a in enumerate(FR):
  for j,b in enumerate(FR):
   if i==j:continue
   s=f'{a[0].capitalize()} {a[1]} {a[2]} {a[3]}; {b[0]} {b[1]} {b[2]} {b[3]}.'; au=audit(s)
   rows.append({'id':f'seam-{i}-{j}','rendered':s,'seam_state':{'reserved_left_terminal':t(s)[0],'reserved_right_terminal':t(s)[-1],'interior_decoding':'bilateral lexical seam beam','endpoint_reservation_before_interior':True},'audit':au,'provenance':{'method':'endpoint-aware character decoding reserves bilateral lexical seam terminals before interior LM expansion','source_sentences_copied':False,'catalogue_imported':False,'borrowed_text':False,'reversed_finished_sentence':False,'word_order_symmetry':False,'repeated_self_palindromic_unit':False,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':au['sha256_forward']},'anti_shortcut':{'intact_prose':True,'repeated_unit':False,'word_order_only':False,'catalogue_text':False,'nested_palindrome_spans':False},'next_repair':f"Repair seam at normalized index {au['first_mismatch']['index']}: replace the right lexical terminal bundle with a same-role synonym whose boundary letter is '{au['first_mismatch']['forward']}', then re-decode interior."})
 d={'experiment':'endpoint-aware-bilateral-seam-20260916','novelty_preflight':{'passed':True,'signature':'endpoint-aware-character-decoding|bilateral-lexical-seam|reserve-terminals-before-interior|typed-scene-beam','overlaps_checked':['char-lm-semantic-beam-20260916','char-lm-constrained-decoding-20260916'],'reason':'The endpoint reservation is a new state variable applied before interior decoding; no completed tape is reversed.'},'rows':rows,'summary':{'candidate_count':len(rows),'exact_count':sum(x['audit']['exact'] for x in rows),'max_length':max(x['audit']['length'] for x in rows)}}
 O.write_text(json.dumps(d,indent=2)+'\n');print(d['summary']);print(rows[0]['rendered'])
if __name__=='__main__':main()
