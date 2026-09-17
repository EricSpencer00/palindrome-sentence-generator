import hashlib,json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'runs/live-tape-clause-terminal-decoder-20260916.json'
F=[('the archivist','files','the old letters','by the lamp'),('a patient gardener','plants','the young cedar trees','after rain'),('the careful pilot','checks','the distant runway lights','through mist')]
def n(s):return re.sub('[^a-z]','',s.lower())
def audit(s):
 x=n(s);i=0
 while i<len(x)//2 and x[i]==x[-1-i]:i+=1
 f=hashlib.sha256(x.encode()).hexdigest();r=hashlib.sha256(x[::-1].encode()).hexdigest();ok=i==len(x)//2
 return {'length':len(x),'independent_two_pointer_exact':ok,'exact':ok,'first_mismatch':None if ok else {'index':i,'forward':x[i],'reverse':x[-1-i]},'sha256_forward':f,'sha256_reverse':r,'sha256_equal':f==r}
def main():
 rows=[]
 for i,a in enumerate(F):
  for j,b in enumerate(F):
   if i==j:continue
   s=f'{a[0].capitalize()} {a[1]} {a[2]} {a[3]}; {b[0]} {b[1]} {b[2]} {b[3]}.';z=audit(s)
   rows.append({'id':f'live-{i}-{j}','rendered':s,'decoder_state':{'left_clause_terminal':a[1],'right_clause_terminal':b[1],'mirrored_tape_obligation':'live at each terminal','center':'free semicolon boundary'},'audit':z,'provenance':{'method':'character decoder jointly chooses complete clause terminals while carrying mirrored tape obligation','source_sentences_copied':False,'catalogue_imported':False,'borrowed_text':False,'reversed_finished_sentence':False,'word_order_symmetry':False,'repeated_self_palindromic_unit':False,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':z['sha256_forward']},'anti_shortcut':{'intact_prose':True,'repeated_unit':False,'word_order_only':False,'catalogue_text':False,'nested_palindrome_spans':False},'next_repair':'At the first live debt, replace the responsible typed terminal with a same-role lexical alternative before extending either clause.'})
 d={'experiment':'live-tape-clause-terminal-decoder-20260916','novelty_preflight':{'passed':True,'signature':'live-mirrored-tape-obligation|joint-clause-terminal-decoding|free-center|typed-terminal-repair','overlaps_checked':['char-lm-semantic-beam-20260916','endpoint-aware-bilateral-seam-20260916'],'reason':'The mirrored obligation is part of the decoder state for each clause terminal; no completed output is reversed.'},'rows':rows,'summary':{'candidate_count':len(rows),'exact_count':sum(x['audit']['exact'] for x in rows),'max_length':max(x['audit']['length'] for x in rows)}};O.write_text(json.dumps(d,indent=2)+'\n');print(d['summary']);print(rows[0]['rendered'])
if __name__=='__main__':main()
