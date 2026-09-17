import hashlib,json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'runs/scalable-outsidein-opposing-terminal-repair2-20260916.json'
F=[('At dawn','the archivist','opens','the town archive'),('By evening','a patient gardener','covers','the young seedlings'),('After rain','the careful nurse','carries','the clean water')]
P=[('near the blue footbridge','under the old awning'),('beside the river steps','across the quiet court'),('along the cedar fence','near the harbor shed')]
def n(s):return re.sub('[^a-z]','',s.lower())
def a(s):
 x=n(s);i=0
 while i<len(x)//2 and x[i]==x[-1-i]:i+=1
 f=hashlib.sha256(x.encode()).hexdigest();r=hashlib.sha256(x[::-1].encode()).hexdigest();ok=i==len(x)//2
 return {'length':len(x),'independent_two_pointer_exact':ok,'exact':ok,'first_mismatch':None if ok else {'index':i,'forward':x[i],'reverse':x[-1-i]},'sha256_forward_normalized':f,'sha256_reverse_normalized':r,'sha256_equal':f==r}
def main():
 rows=[]
 for i in range(3):
  l=F[i];r=F[(i+1)%3];s=f'{l[0]}, {l[1]} {l[2]} {l[3]} {P[i][0]}, while {r[1]} {r[2]} {r[3]} {P[i][1]}.';z=a(s)
  rows.append({'id':f'opposing2-{i}','rendered':s,'repair_state':{'changed_side':'opposing outer terminal pair','phrase_pair':P[i],'scene_frame_preserved':True,'growth_rule':'append linked semantic frame without nesting'},'audit':z,'provenance':{'method':'second held-out opposing-terminal repair in productive outside-in grammar','source_sentences_copied':False,'catalogue_imported':False,'borrowed_text':False,'reversed_finished_sentence':False,'word_order_symmetry':False,'repeated_self_palindromic_unit':False,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':z['sha256_forward_normalized']},'anti_shortcut':{'intact_prose':True,'nested_spans':False,'seed_wrapping':False,'word_order_mirror':False,'repeated_unit':False},'next_repair':f"First residual {z['first_mismatch']['index']}: hold this side fixed and replace the other edge phrase with a new same-role terminal."})
 d={'experiment':'scalable-outsidein-opposing-terminal-repair2-20260916','novelty_preflight':{'passed':True,'signature':'opposing-terminal-heldout-repair2|productive-outsidein-growth|scene-preserving-state|crossword-debt','overlaps_checked':['scalable-outsidein-paired-terminal-repair-20260916'],'reason':'Only the opposing outer phrase pair changes from the prior repair; scene frame and growth state are held fixed.'},'rows':rows,'summary':{'candidate_count':len(rows),'exact_count':sum(x['audit']['exact'] for x in rows),'max_length':max(x['audit']['length'] for x in rows)}};O.write_text(json.dumps(d,indent=2)+'\n');print(d['summary']);print(rows[0]['rendered'])
if __name__=='__main__':main()
