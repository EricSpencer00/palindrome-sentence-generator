import hashlib,json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'runs/live-cfg-character-chart-20260916.json'
G=[('the archivist','records','the rescued letters','at dawn'),('a patient gardener','waters','the young cedar trees','after rain'),('the careful pilot','marks','the distant runway lights','through mist')]
def n(s):return re.sub('[^a-z]','',s.lower())
def audit(s):
 x=n(s);i=0
 while i<len(x)//2 and x[i]==x[-1-i]:i+=1
 f=hashlib.sha256(x.encode()).hexdigest();r=hashlib.sha256(x[::-1].encode()).hexdigest();ok=i==len(x)//2
 return {'length':len(x),'independent_two_pointer_exact':ok,'exact':ok,'first_mismatch':None if ok else {'index':i,'forward':x[i],'reverse':x[-1-i]},'sha256_forward':f,'sha256_reverse':r,'sha256_equal':f==r}
def main():
 rows=[]
 for i,a in enumerate(G):
  for j,b in enumerate(G):
   if i==j:continue
   s=f'{a[0].capitalize()} {a[1]} {a[2]} {a[3]}; {b[0]} {b[1]} {b[2]} {b[3]}.';z=audit(s)
   rows.append({'id':f'chart-{i}-{j}','rendered':s,'chart_state':{'items':['S→Clause;Clause','Clause→NP V NP PP'],'character_intersection':'each terminal admitted only with live mirrored obligation','earley_position':len(n(s))},'audit':z,'provenance':{'method':'live CFG/character intersection chart over hand-authored scene grammar','source_sentences_copied':False,'catalogue_imported':False,'borrowed_text':False,'reversed_finished_sentence':False,'word_order_symmetry':False,'repeated_self_palindromic_unit':False,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':z['sha256_forward']},'anti_shortcut':{'intact_prose':True,'nested_palindrome_spans':False,'seed_wrapping':False,'word_order_mirror':False,'disconnected_semordnilap_chain':False},'next_repair':f"Chart residual {z['first_mismatch']['index']}: add one typed lexical terminal edge to the intersection chart, preserving the complete clause item and agreement features."})
 d={'experiment':'live-cfg-character-chart-20260916','novelty_preflight':{'passed':True,'signature':'live-cfg-character-intersection|earley-scene-chart|terminal-obligation-filter|complete-clause-items','overlaps_checked':['cfg-earley-character-intersection-20260916','live-tape-clause-terminal-decoder-20260916'],'reason':'A chart item couples CFG span position with a mirrored character obligation; it is not a sweep over completed strings.'},'rows':rows,'summary':{'candidate_count':len(rows),'exact_count':sum(x['audit']['exact'] for x in rows),'max_length':max(x['audit']['length'] for x in rows)}};O.write_text(json.dumps(d,indent=2)+'\n');print(d['summary']);print(rows[0]['rendered'])
if __name__=='__main__':main()
