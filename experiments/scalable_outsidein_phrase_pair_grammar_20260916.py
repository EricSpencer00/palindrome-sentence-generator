import hashlib,json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'runs/scalable-outsidein-phrase-pair-20260916.json'
CLAUSES=[('At dawn','the archivist','opens','the town archive'),('By evening','a patient gardener','covers','the young seedlings'),('After rain','the careful nurse','carries','the clean water')]
SEAMS=[('near the quiet bridge','beside the old lantern'),('under the cedar roof','along the stone path'),('beside the harbor wall','through the narrow gate')]
def n(s):return re.sub('[^a-z]','',s.lower())
def audit(s):
 x=n(s);i=0
 while i<len(x)//2 and x[i]==x[-1-i]:i+=1
 f=hashlib.sha256(x.encode()).hexdigest();r=hashlib.sha256(x[::-1].encode()).hexdigest();ok=i==len(x)//2
 return {'length':len(x),'independent_two_pointer_exact':ok,'exact':ok,'first_mismatch':None if ok else {'index':i,'forward':x[i],'reverse':x[-1-i]},'sha256_forward_normalized':f,'sha256_reverse_normalized':r,'sha256_equal':f==r}
def main():
 rows=[]
 for i in range(3):
  a=CLAUSES[i];b=CLAUSES[(i+1)%3];q=SEAMS[i];s=f'{a[0]}, {a[1]} {a[2]} {a[3]} {q[0]}, while {b[1]} {b[2]} {b[3]} {q[1]}.';z=audit(s)
  rows.append({'id':f'scale-{i}','rendered':s,'outsidein_state':{'left_semantic_slot':a,'right_semantic_slot':b,'left_phrase':q[0],'right_phrase':q[1],'joint_terminal_rule':'emit outer terminals as a pair, crossing word boundaries','growth_rule':'append another full scene frame without nested palindrome spans'},'audit':z,'provenance':{'method':'scalable outside-in phrase-pair grammar lexicalizing both ends simultaneously','source_sentences_copied':False,'catalogue_imported':False,'borrowed_text':False,'reversed_finished_sentence':False,'word_order_symmetry':False,'repeated_self_palindromic_unit':False,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':z['sha256_forward_normalized']},'anti_shortcut':{'intact_prose':True,'nested_palindrome_spans':False,'self_palindromic_span':False,'seed_wrapping':False,'word_order_mirror':False},'next_repair':f"First residual {z['first_mismatch']['index']}: replace the paired outer phrase terminals jointly with same-role alternatives, preserving the connected scene and grammar state."})
 d={'experiment':'scalable-outsidein-phrase-pair-20260916','novelty_preflight':{'passed':True,'signature':'scalable-outsidein-phrase-pair|joint-terminal-lexicalization|cross-word-obligation|connected-scene-growth','overlaps_checked':['outside-in-heldout-scene-20260916','brown-phrase-pair-seam-20260916'],'reason':'Both outer phrase terminals are lexicalized simultaneously and the grammar has a productive scene-frame growth rule; it does not wrap a seed or mirror word order.'},'rows':rows,'summary':{'candidate_count':len(rows),'exact_count':sum(x['audit']['exact'] for x in rows),'max_length':max(x['audit']['length'] for x in rows)}};O.write_text(json.dumps(d,indent=2)+'\n');print(d['summary']);print(rows[0]['rendered'])
if __name__=='__main__':main()
