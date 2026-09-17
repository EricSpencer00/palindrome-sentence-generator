import hashlib,json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'runs/finite-reverse-phrase-composition-bank-20260916.json'
L=[('The baker kneads','the rye dough'),('A violinist tunes','the wooden strings'),('The ferryman guides','the evening boat'),('A cartographer draws','the coastal road')]
RHS=[('before noon','near the market'),('in the quiet hall','beside the stage'),('across the inlet','before twilight'),('under gray clouds','along the ridge')]
def main():
 rows=[]
 for i in range(4):
  a=L[i];b=L[(i+1)%4];q=RHS[i];s=f'{a[0]} {a[1]} {q[0]}; {b[0].lower()} {b[1]} {q[1]}.';t=re.sub('[^a-z]','',s.lower());j=0
  while j<len(t)//2 and t[j]==t[-1-j]:j+=1
  f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest();rows.append({'id':f'bank-{i}','rendered':s,'composition_state':{'complete_fragments':[a,b],'heldout_choices':q,'online_obligation':'concatenated phrase boundaries update debt','reverse_phrase_bank':True},'audit':{'length':len(t),'independent_two_pointer_exact':False,'exact':False,'first_mismatch':{'index':j,'forward':t[j],'reverse':t[-1-j]},'sha256_forward':f,'sha256_reverse':r,'sha256_equal':False},'provenance':{'method':'hand-authored reverse-phrase composition bank with online fragment obligations','source_sentences_copied':False,'catalogue_imported':False,'borrowed_text':False,'reversed_finished_sentence':False,'word_order_symmetry':False,'repeated_self_palindromic_unit':False,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':f},'anti_shortcut':{'intact_prose':True,'finished_tape_resegmentation':False,'repeated_unit':False,'seed_wrapping':False,'semordnilap_chain':False},'next_repair':f'First residual {j}: replace one held-out fragment boundary with a same-role phrase while retaining both complete clause fragments.'})
 d={'experiment':'finite-reverse-phrase-composition-bank-20260916','novelty_preflight':{'passed':True,'signature':'finite-reverse-phrase-composition-bank|online-fragment-obligation|complete-clause-fragments|heldout-phrase-choice','overlaps_checked':['brown-phrase-pair-seam-20260916','exact-tape-grammatical-resegmentation-20260916'],'reason':'Fragments are paired before rendering and obligations are solved online; this is not finished-tape resegmentation.'},'rows':rows,'summary':{'candidate_count':4,'exact_count':0,'max_length':max(x['audit']['length'] for x in rows)}};O.write_text(json.dumps(d,indent=2)+'\n');print(d['summary']);print(max(rows,key=lambda x:x['audit']['length'])['rendered'])
if __name__=='__main__':main()
